import tensorflow as tf
import keras
import os
import cv2
import numpy as np
from docmask_dataset import DocMaskDataset

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

def unet_like_model(image_size=224):
    # Encoder (MobileNetV3Large)
    backbone = keras.applications.MobileNetV3Large(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False
    )
    
    # Verified skip connections
    skip_connections = [
        backbone.get_layer("expanded_conv_2_depthwise").output,   # 56x56 (72 channels)
        backbone.get_layer("expanded_conv_6_depthwise").output,   # 14x14 (240 channels)
        backbone.get_layer("expanded_conv_12_depthwise").output,  # 7x7 (672 channels)
    ]
    
    # Bridge layer (last encoder output)
    bridge = backbone.output  # Expected to be 7x7 with 672 channels

    # Decoder configuration
    decoder_filters = [256, 128, 64]  # Filter sizes for each decoder stage
    x = bridge
    
    # Reverse skip connections to go from deep to shallow
    for i, (skip, filters, stride) in enumerate(zip(reversed(skip_connections), decoder_filters, [1, 2, 4])):
        # Step 1: Upsample previous output
        x = keras.layers.Conv2DTranspose(filters, 3, strides=stride, padding="same")(x)
        
        # Step 2: Adjust skip connection channels to match decoder features
        skip = keras.layers.Conv2D(filters, 1, padding="same")(skip)
        
        # Step 3: Concatenate with processed skip connection
        x = keras.layers.Concatenate()([x, skip])
        
        # Step 4: Process combined features
        x = convolution_block(x, filters)

    # Final upsampling to original image size
    x = keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)  # 56x56 -> 224x224
    x = convolution_block(x, 32)
    outputs = keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
    
    return keras.Model(backbone.input, outputs)

@keras.saving.register_keras_serializable()
def hybrid_loss(y_true, y_pred, alpha=0.333, beta=0.334):
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
    
    return alpha * bce_loss + beta * dice_loss + (1 - alpha - beta) * edge_loss

def visualize_feature_maps(features, orig_img):
    features = np.array(features)
    feature_maps = features[0]

    normalized_feature_maps = []
    for i in range(feature_maps.shape[-1]):
        feature_map = feature_maps[:, :, i]
        
        normalized_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)
        
        normalized_map = (normalized_map * 255).astype(np.uint8)
        normalized_feature_maps.append(normalized_map)

    normalized_feature_maps = np.array(normalized_feature_maps)

    average_map = np.mean(normalized_feature_maps, axis=0).astype(np.uint8)
    # average_map = cv2.resize(average_map, (224, 224))
    cv2.imshow("Average Feature Map", average_map)
    cv2.imshow("Original Image", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict(model):
    for i in range(len(os.listdir("./predict_datasets"))):
        img = cv2.imread(f"./predict_datasets/receipt-test-{i}.jpg")
        img = cv2.resize(img, (224, 224))
        img_show = np.copy(img)
        img = img.astype(np.float32)
        img2 = img
        img2 = np.expand_dims(img2, axis=0)

        result = model(img2, training=False)
        result = result[0].numpy()
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imshow("Average Feature Map", result)
        cv2.imshow("Original Image", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def train(model, loss, epoch=10):
    print(model.summary())
    dataset = DocMaskDataset(txt_path="./labels/receipt_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=32)
    train_ds, val_ds = dataset.load()
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath='./output/model_{epoch:02d}_{val_loss:.6f}.keras', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks=callbacks)
    model.save("./output/final_model.keras")

def cat_model_summary(s):
    with open('model_summary.txt','w') as f:
        print(s, file=f)

def debug_model(model):
    print(model.summary())
    dataset = DocMaskDataset(txt_path="./labels/receipt_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=1)
    train_ds, val_ds = dataset.load()
    for x, y in train_ds.take(3):
        image = x[0].numpy()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        output = model(x)
        output = output[0].numpy()
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        output = cv2.resize(output, (224, 224))
        cv2.imshow("Image", image)
        cv2.imshow("Mask", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__": 
    # model = docmask_model()
    # model = keras.models.load_model("./output/unet_model.keras")
    model = unet_like_model()
    # model.summary(print_fn=cat_model_summary)
    train(model, hybrid_loss, epoch=200)
    # debug_model(model)
    # predict(model)