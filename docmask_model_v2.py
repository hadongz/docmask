import tensorflow as tf
import keras
import os
from docmask_dataset import DocMaskDataset
from utils import cat_model_summary
from loss import HybridLossV3, HybridLossV2
from metrics import boundary_iou, boundary_f1

os.environ["KERAS_BACKEND"] = "tensorflow"

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
    dropout_rate=0.1,
    activation="swish",
):
    """Enhanced convolution block with advanced features"""
    x = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer="he_normal",
    )(block_input)
    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    
    if dropout_rate > 0:
        x = keras.layers.SpatialDropout2D(dropout_rate)(x)
    
    # Residual connection with projection if needed
    if block_input.shape[-1] != num_filters:
        shortcut = keras.layers.Conv2D(
            num_filters, 1, padding="same", 
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(1e-4)
        )(block_input)
        shortcut = keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = block_input
        
    x = keras.layers.Add()([x, shortcut])
    
    return x

def channel_attention(inputs, reduction=8):
    """Squeeze-and-excitation channel attention module"""
    channels = inputs.shape[-1]
    
    # Squeeze
    se = keras.layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    
    # Excite
    se = keras.layers.Dense(
        channels // reduction,
        activation="swish",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(se)
    se = keras.layers.Dense(
        channels,
        activation="sigmoid",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(se)
    
    return keras.layers.Multiply()([inputs, se])

def attention_gate(x, g, filters, compression=2):
    """Improved attention gate with channel compression"""
    theta_x = keras.layers.Conv2D(filters//compression, 1, padding='same')(x)
    phi_g = keras.layers.Conv2D(filters//compression, 1, padding='same')(g)
    
    if g.shape[1] < x.shape[1]:
        phi_g = keras.layers.UpSampling2D(
            size=(x.shape[1] // g.shape[1], x.shape[2] // g.shape[2]),
            interpolation="bilinear"
        )(phi_g)
    
    f = keras.layers.Add()([theta_x, phi_g])
    f = keras.layers.Activation('relu')(f)
    
    psi_f = keras.layers.Conv2D(1, 1, padding='same')(f)
    psi_f = keras.layers.UpSampling2D(
        size=(x.shape[1] // psi_f.shape[1], x.shape[2] // psi_f.shape[2]),
        interpolation="bilinear"
    )(psi_f)
    
    alpha = keras.layers.Activation('sigmoid')(psi_f)
    alpha = keras.layers.Conv2D(1, 1, activation='sigmoid', kernel_initializer='he_normal')(alpha)
    
    # Channel-wise scaling
    channel_scale = keras.layers.GlobalAveragePooling2D(keepdims=True)(alpha)
    channel_scale = keras.layers.Dense(1, activation='swish', kernel_initializer='he_normal')(channel_scale)
    alpha = keras.layers.Multiply()([alpha, channel_scale])
    
    return keras.layers.Multiply()([x, alpha])

def aspp_module(inputs, output_stride=16, l2_weight=1e-4):
    """Improved ASPP module with optimized dilation rates"""
    shape = tf.keras.backend.int_shape(inputs)
    
    if output_stride == 16:
        rates = [6, 12, 18]
    else:  # For output_stride = 8
        rates = [12, 24, 36]
    
    # 1x1 conv branch
    conv1x1 = keras.layers.Conv2D(
        256, 1, padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_weight)
    )(inputs)
    conv1x1 = keras.layers.BatchNormalization()(conv1x1)
    conv1x1 = keras.layers.Activation("swish")(conv1x1)
    
    # Atrous branches
    atrous_convs = []
    for rate in rates:
        x = keras.layers.SeparableConv2D(
            256, 3, padding='same',
            dilation_rate=rate,
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
        )(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("swish")(x)
        atrous_convs.append(x)
    
    # Global average pooling branch
    gap = keras.layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    gap = keras.layers.Conv2D(
        256, 1, padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_weight)
    )(gap)
    gap = keras.layers.BatchNormalization()(gap)
    gap = keras.layers.Activation("swish")(gap)
    gap = keras.layers.UpSampling2D(
        size=(shape[1], shape[2]),
        interpolation='bilinear'
    )(gap)
    
    # Concatenate and compress
    x = keras.layers.Concatenate()([conv1x1] + atrous_convs + [gap])
    x = keras.layers.Conv2D(
        256, 1, padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(l2_weight)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("swish")(x)
    x = channel_attention(x)
    
    return x

def dynamic_weighting(x, skip_processed):
    # Generate multi-scale weights
    scale1 = keras.layers.Conv2D(1, 1, padding="same", kernel_initializer="he_normal")(x)
    scale2 = keras.layers.SeparableConv2D(1, 3, dilation_rate=2, padding="same")(x)
    
    # Fuse scales
    skip_weights = keras.layers.Add()([scale1, scale2])
    skip_weights = keras.layers.BatchNormalization()(skip_weights)
    skip_weights = keras.layers.Activation("swish")(skip_weights)
    skip_weights = keras.layers.Activation("sigmoid")(skip_weights)
    skip_weights = keras.layers.SpatialDropout2D(0.1)(skip_weights)
    
    # Apply weights with residual connection
    weighted = keras.layers.Multiply()([skip_processed, skip_weights])
    skip_processed = keras.layers.Add()([weighted, skip_processed])  # Residual add
    
    return skip_processed

def docmask_model_v2(image_size=224, num_classes=1, use_attn=True):
    """
    Improved document segmentation and classification model
    
    Args:
        image_size: Input image size (assumed square)
        num_classes: Number of classification classes (1 for binary)
        use_attn: Whether to use attention gates
        
    Returns:
        Keras model with segmentation and classification outputs
    """
    # Input layer
    input_layer = keras.layers.Input((image_size, image_size, 3), name="img_input")
    x = keras.layers.Rescaling(1.0 / 255.0)(input_layer)
    
    # Encoder (MobileNetV3Large with fixed weights initially)
    backbone = keras.applications.MobileNetV3Large(
        input_tensor=x,
        include_top=False,
        weights="imagenet",
        include_preprocessing=False
    )

    # Unfreeze last few layers for fine-tuning
    unfreeze_layers = [
        'expanded_conv_2',
        'expanded_conv_6', 
        'expanded_conv_8', 
        'expanded_conv_10', 
        'expanded_conv_11', 
        'expanded_conv_12'
    ]
    for layer in backbone.layers:
        if any([kw in layer.name for kw in unfreeze_layers]):
            layer.trainable = True
        else:
            layer.trainable = False
    
    # Enhanced skip connections with more optimal layer choices
    skip_connections = [
        backbone.get_layer("re_lu").output,                   # 112x112
        backbone.get_layer("expanded_conv_2_depthwise").output,   # 56x56
        backbone.get_layer("expanded_conv_6_depthwise").output,   # 14x14
        backbone.get_layer("expanded_conv_12_depthwise").output,  # 7x7
    ]
    
    bridge = backbone.output  # 7x7 with 960 channels
    
    # ASPP (Atrous Spatial Pyramid Pooling) for better context understanding
    x = aspp_module(bridge)
    
    # Decoder with attention gates
    decoder_filters = [256, 128, 64, 32]
    
    for i, (skip, filters) in enumerate(zip(reversed(skip_connections), decoder_filters)):
        # Calculate upsampling factor based on current and target feature map size
        stride = (skip.shape[1] // x.shape[1], skip.shape[2] // x.shape[2])
        
        if stride[0] > 1 or stride[1] > 1:
            x = keras.layers.Conv2DTranspose(filters, 3, strides=stride, padding="same")(x)
        
        # Process skip connection
        skip_processed = keras.layers.Conv2D(filters, 1, padding="same", kernel_initializer="he_normal")(skip)
        skip_processed = keras.layers.BatchNormalization()(skip_processed)
        skip_processed = keras.layers.Activation("swish")(skip_processed)
        skip_processed = keras.layers.SpatialDropout2D(0.05)(skip_processed)

        # Apply attention gate if requested
        if use_attn:
            skip_processed = attention_gate(skip_processed, x, filters)
        
        # Add dynamic weighting based on decoder features
        skip_processed = dynamic_weighting(x, skip_processed)

        # Combine features
        x = keras.layers.Concatenate()([x, skip_processed])
        x = convolution_block(x, filters, dropout_rate=0.2)
        x = convolution_block(x, filters, dropout_rate=0.2)
    
    # Final upsampling to original image size if needed
    if x.shape[1] < image_size:
        x = keras.layers.UpSampling2D(
            size=(image_size // x.shape[1], image_size // x.shape[2]), 
            interpolation="bilinear"
        )(x)
    
    # Segmentation head
    segmentation_output = keras.layers.Conv2D(
        1, 3, padding="same", activation="sigmoid", name="segmentation_output"
    )(x)
    
    # Classification head with feature pyramid
    # Use both high-level (bridge) and mid-level features
    cls_features = [
        keras.layers.GlobalAveragePooling2D()(bridge),
        keras.layers.GlobalAveragePooling2D()(skip_connections[1])  # Mid-level feature
    ]
    
    cls_x = keras.layers.Concatenate()(cls_features)
    cls_x = keras.layers.Dense(256, activation="relu")(cls_x)
    cls_x = keras.layers.BatchNormalization()(cls_x)
    cls_x = keras.layers.Dropout(0.5)(cls_x)
    cls_x = keras.layers.Dense(128, activation="relu")(cls_x)
    cls_x = keras.layers.Dropout(0.3)(cls_x)
    classification_output = keras.layers.Dense(
        num_classes, activation="sigmoid", name="classification_output"
    )(cls_x)
    
    model = keras.Model(backbone.input, [segmentation_output, classification_output])
    
    return model

def train_v2(model, batch_size=16, epoch=100, use_simple_metrics=True):
    """
    Create a complete training pipeline optimized for a small dataset (1000 images)
    """
    # Define losses and weights
    segmentation_loss = HybridLossV2()
    losses = {
        "segmentation_output": segmentation_loss,
        "classification_output": "binary_crossentropy"
    }
    
    loss_weights = {
        "segmentation_output": 1.0,
        "classification_output": 0.5
    }

    train_size = 1500 * 0.8 // batch_size
    total_steps = epoch * train_size
    warmup_steps = int(0.1 * total_steps)
    decay_steps = total_steps - warmup_steps 
    lr_scheduler = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        alpha=0.01
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler, clipnorm=1.0)
    
    # Metrics
    if use_simple_metrics:
        metrics = {
            "segmentation_output": [
                "accuracy"
            ],
            "classification_output": [
                "accuracy"
            ]
        }
    else:
        metrics = {
            "segmentation_output": [
                tf.keras.metrics.MeanIoU(num_classes=2),
                boundary_iou,
                boundary_f1
            ],
            "classification_output": [
                tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
                tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
                tf.keras.metrics.PrecisionAtRecall(0.9)
            ]
        }
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    print(model.summary())
    print("======== TRAIN DOCMASK V2 MODEL ========")
    model.summary(print_fn=cat_model_summary)
    dataset = DocMaskDataset(txt_path="./labels/train_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=batch_size)
    train_ds, val_ds = dataset.load()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, start_from_epoch=50, verbose=1),
        keras.callbacks.ModelCheckpoint(monitor="val_segmentation_output_loss", filepath='./output/model_{epoch:02d}_{val_segmentation_output_loss:.4f}.keras', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, update_freq='epoch', write_graph=True, write_images=True),
        keras.callbacks.CSVLogger("./logs/training.csv", separator=",", append=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks=callbacks, verbose=1)
    model.save("./output/final_model.keras")