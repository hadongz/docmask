import tensorflow as tf
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
):
    x = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.ReLU()(x)

def docmask_model(image_size=224):
    # Encoder (MobileNetV3Large)
    input_layer = keras.layers.Input((224, 224, 3), name="img_input")
    backbone = keras.applications.MobileNetV3Large(
        input_tensor=input_layer,
        include_top=False,
        weights="imagenet",
        include_preprocessing=False
    )
    
    # Skip connections (verified layer names)
    skip_connections = [
        backbone.get_layer("expanded_conv_2_depthwise").output,   # 56x56
        backbone.get_layer("expanded_conv_6_depthwise").output,   # 14x14
        backbone.get_layer("expanded_conv_12_depthwise").output,  # 7x7
    ]
    
    bridge = backbone.output  # 7x7 with 960 channels (not 672)

    # Decoder
    decoder_filters = [256, 128, 64]
    x = bridge
    
    for i, (skip, filters, stride) in enumerate(zip(reversed(skip_connections), decoder_filters, [1, 2, 4])):
        x = keras.layers.Conv2DTranspose(filters, 3, strides=stride, padding="same")(x)
        skip = keras.layers.Conv2D(filters, 1, padding="same")(skip)
        x = keras.layers.Concatenate()([x, skip])
        x = convolution_block(x, filters)

    # Final upsampling to original image size
    x = keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)  # 56x56 -> 224x224
    x = convolution_block(x, 32)
    segmentation_output = keras.layers.Conv2D(1, 1, activation="sigmoid", name="segmentation_output")(x)

    x2 = keras.layers.GlobalAveragePooling2D()(bridge)
    x2 = keras.layers.Dense(128, activation="relu")(x2)
    x2 = keras.layers.Dropout(0.5)(x2)  # Regularization
    classification_output = keras.layers.Dense(1, activation="sigmoid", name="classification_output")(x2)
    
    return keras.Model(backbone.input, [segmentation_output, classification_output])

@keras.saving.register_keras_serializable()
def hybrid_loss(y_true, y_pred):
    # 1. Dice Loss
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_loss = 1 - (2. * intersection + 1e-7) / (union + 1e-7)
    
    # 2. BCE
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # 3. Edge Loss using TensorFlow's Sobel operator
    edge_true = tf.abs(tf.image.sobel_edges(y_true))
    edge_pred = tf.abs(tf.image.sobel_edges(y_pred))
    edge_loss = tf.reduce_mean(tf.square(edge_true - edge_pred))
    
    return 0.5 * bce_loss + 0.4 * dice_loss + 0.1 * edge_loss

def train(model, epoch=10, need_compile=True):
    print(model.summary())
    print("======== TRAIN DOCMASK V1 MODEL ========")
    model.summary(print_fn=cat_model_summary)
    dataset = DocMaskDataset(txt_path="./labels/train_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=32)
    train_ds, val_ds = dataset.load()

    if need_compile:
        loss = {
            "segmentation_output": hybrid_loss, 
            "classification_output": keras.losses.binary_crossentropy
        }
        loss_weights = {
            "segmentation_output": 1.0, 
            "classification_output": 1.0
        }
        metrics = {
            "segmentation_output": ["accuracy"],
            "classification_output": ["accuracy"]
        }
        model.compile(optimizer="adam", loss=loss, loss_weights=loss_weights, metrics=metrics)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, start_from_epoch=100, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath='./output/model_{epoch:02d}_{val_loss:.4f}.keras', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir='./tensorboard'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger("./logs/training.csv", separator=",", append=False)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks=callbacks, verbose=1)
    model.save("./output/final_model.keras")
