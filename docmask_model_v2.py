import tensorflow as tf
import numpy as np
import keras
import os
from docmask_dataset import DocMaskDataset
from utils import cat_model_summary

os.environ["KERAS_BACKEND"] = "tensorflow"

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
    dropout_rate=0.1,
):
    """Enhanced convolution block with optional dropout and residual connections"""
    x = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    
    # Add dropout for regularization
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)
    
    # Add residual connection if input and output shapes match
    if block_input.shape[-1] == num_filters:
        x = keras.layers.Add()([x, block_input])
        
    return x

def attention_gate(x, g, filters):
    """Attention gate to focus on relevant features"""
    theta_x = keras.layers.Conv2D(filters, 1, padding='same')(x)
    phi_g = keras.layers.Conv2D(filters, 1, padding='same')(g)
    
    # Upsample g if needed to match x's dimensions
    if g.shape[1] < x.shape[1]:
        phi_g = keras.layers.UpSampling2D(
            size=(x.shape[1] // g.shape[1], x.shape[2] // g.shape[2]),
            interpolation="bilinear"
        )(phi_g)
    
    f = keras.layers.Activation('relu')(keras.layers.Add()([theta_x, phi_g]))
    psi_f = keras.layers.Conv2D(1, 1, padding='same')(f)
    
    alpha = keras.layers.Activation('sigmoid')(psi_f)
    return keras.layers.Multiply()([x, alpha])

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
    
    # Data normalization
    x = keras.layers.Lambda(lambda x: x / 255.0)(input_layer)
    
    # Encoder (MobileNetV3Large with fixed weights initially)
    backbone = keras.applications.MobileNetV3Large(
        input_tensor=x,
        include_top=False,
        weights="imagenet",
        include_preprocessing=False
    )
    
    # Unfreeze last few layers for fine-tuning
    for layer in backbone.layers[-30:]:
        layer.trainable = True
    
    # Enhanced skip connections with more optimal layer choices
    skip_connections = [
        backbone.get_layer("re_lu").output,                   # 112x112
        backbone.get_layer("expanded_conv_2_depthwise").output,   # 56x56
        backbone.get_layer("expanded_conv_6_depthwise").output,   # 14x14
        backbone.get_layer("expanded_conv_12_depthwise").output,  # 7x7
    ]
    
    bridge = backbone.output  # 7x7 with 960 channels
    
    # ASPP (Atrous Spatial Pyramid Pooling) for better context understanding
    aspp_outputs = []
    aspp_rates = [1, 6, 12, 18]
    
    for rate in aspp_rates:
        aspp_outputs.append(convolution_block(
            bridge, 
            num_filters=256, 
            kernel_size=3, 
            dilation_rate=rate
        ))
    
    # Global pooling branch
    global_features = keras.layers.GlobalAveragePooling2D()(bridge)
    global_features = keras.layers.Reshape((1, 1, 960))(global_features)
    global_features = keras.layers.Conv2D(256, 1, padding='same')(global_features)
    global_features = keras.layers.UpSampling2D(
        size=(bridge.shape[1], bridge.shape[2]),
        interpolation='bilinear'
    )(global_features)
    
    aspp_outputs.append(global_features)
    x = keras.layers.Concatenate()(aspp_outputs)
    x = convolution_block(x, 256)
    
    # Decoder with attention gates
    decoder_filters = [256, 128, 64, 32]
    
    for i, (skip, filters) in enumerate(zip(reversed(skip_connections), decoder_filters)):
        # Calculate upsampling factor based on current and target feature map size
        stride = (skip.shape[1] // x.shape[1], skip.shape[2] // x.shape[2])
        
        if stride[0] > 1 or stride[1] > 1:
            x = keras.layers.Conv2DTranspose(
                filters, 3, strides=stride, padding="same"
            )(x)
        
        # Process skip connection
        skip_processed = keras.layers.Conv2D(filters, 1, padding="same")(skip)
        
        # Apply attention gate if requested
        if use_attn:
            skip_processed = attention_gate(skip_processed, x, filters)
            
        # Combine features
        x = keras.layers.Concatenate()([x, skip_processed])
        x = convolution_block(x, filters, dropout_rate=0.1)
        x = convolution_block(x, filters, dropout_rate=0.1)
    
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

@keras.saving.register_keras_serializable()
def hybrid_loss_v2(y_true, y_pred):
    """
    Enhanced hybrid loss function for segmentation with improved numerical stability
    Combines Dice loss, Focal BCE, and Edge-aware loss
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Larger smooth factor for better stability
    smooth = 1e-6
    
    # Clip predictions more aggressively
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
    
    # 1. Dice Loss with more stable implementation
    # Cast to float32 to ensure precision
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    
    # Handle empty masks case
    dice_coef = tf.where(
        union > smooth,
        (2. * intersection + smooth) / (union + smooth),
        tf.ones_like(intersection)  # If both true and pred are empty, dice is 1
    )
    dice_loss = 1.0 - dice_coef
    dice_loss = tf.reduce_mean(dice_loss)
    
    # 2. Focal BCE with stability improvements
    alpha = 0.75  # Weight for positive class
    gamma = 2.0   # Focusing parameter
    
    # Calculate focal weights with improved stability
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    pt = tf.clip_by_value(pt, smooth, 1.0 - smooth)  # Ensure pt is not too close to 0 or 1
    
    focal_weight = tf.pow(1 - pt, gamma)
    
    # Apply alpha weighting
    alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    # Calculate focal BCE
    bce = -tf.math.log(pt)
    focal_bce = focal_weight * alpha_weight * bce
    focal_bce_loss = tf.reduce_mean(focal_bce)
    
    # 3. Edge Loss with safe normalization
    # Add axis parameter to sobel operator for clarity
    edge_true = tf.image.sobel_edges(y_true)
    edge_pred = tf.image.sobel_edges(y_pred)
    
    # Calculate magnitude of gradients
    edge_true_mag = tf.sqrt(tf.reduce_sum(tf.square(edge_true), axis=-1) + smooth)
    edge_pred_mag = tf.sqrt(tf.reduce_sum(tf.square(edge_pred), axis=-1) + smooth)
    
    # Safe normalization with checks for zero tensors
    edge_true_max = tf.maximum(tf.reduce_max(edge_true_mag), smooth)
    edge_pred_max = tf.maximum(tf.reduce_max(edge_pred_mag), smooth)
    
    norm_edge_true = edge_true_mag / edge_true_max
    norm_edge_pred = edge_pred_mag / edge_pred_max
    
    # L1 loss is more stable than L2 for edge loss
    edge_loss = tf.reduce_mean(tf.abs(norm_edge_true - norm_edge_pred))
    
    # Check for NaN values before combining
    dice_loss = tf.debugging.check_numerics(dice_loss, "Dice loss contains NaN")
    focal_bce_loss = tf.debugging.check_numerics(focal_bce_loss, "Focal BCE loss contains NaN")
    edge_loss = tf.debugging.check_numerics(edge_loss, "Edge loss contains NaN")
    
    # Combine losses with adjusted weights
    total_loss = 0.4 * focal_bce_loss + 0.4 * dice_loss + 0.2 * edge_loss
    
    # Verify final loss is valid
    # total_loss = tf.debugging.check_numerics(total_loss, "Total loss contains NaN")
    
    return total_loss

def train_v2(model, batch_size=16, epoch=100, use_simple_metrics=True):
    """
    Create a complete training pipeline optimized for a small dataset (1000 images)
    """
    
    # Define losses and weights
    losses = {
        "segmentation_output": hybrid_loss_v2,
        "classification_output": "binary_crossentropy"
    }
    
    loss_weights = {
        "segmentation_output": 1.0,
        "classification_output": 0.5
    }
    
    # Adjusted learning rate schedule for smaller dataset
    # One cycle should complete in ~30-40 epochs for 1000 images with batch_size=16
    steps_per_epoch = 1000 // batch_size  # ~63 steps per epoch
    decay_steps = steps_per_epoch * 35  # ~2200 steps (35 epochs)
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=decay_steps,
        alpha=1e-5
    )
    
    # Adam optimizer with appropriate weight decay for small dataset
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        weight_decay=2e-5  # Slightly increased for smaller dataset
    )
    
    # Metrics
    if use_simple_metrics:
        metrics = {
            "segmentation_output": ["accuracy"],
            "classification_output": ["accuracy"]
        }
    else:
        metrics = {
            "segmentation_output": [
                keras.metrics.IoU(num_classes=2, target_class_ids=[1]),
                keras.metrics.Recall(),
                keras.metrics.Precision()
            ],
            "classification_output": [
                keras.metrics.BinaryAccuracy(),
                keras.metrics.AUC(),
                keras.metrics.Precision(),
                keras.metrics.Recall()
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
    model.summary(print_fn=cat_model_summary)
    dataset = DocMaskDataset(txt_path="./labels/train_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=batch_size)
    train_ds, val_ds = dataset.load()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, start_from_epoch=20, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath='./output/model_{epoch:02d}_{val_loss:.4f}.keras', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir='./tensorboard'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger("./logs/training.csv", separator=",", append=False)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks=callbacks, verbose=1)
    model.save("./output/final_model.keras")