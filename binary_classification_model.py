import keras
import os
import cv2
import numpy as np
from binary_dataset import BinaryClassDataset

os.environ["KERAS_BACKEND"] = "tensorflow"

def create_model():
    base_model = keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3), 
        alpha=1.0, 
        include_top=False,
        dropout_rate=0.5,
        include_preprocessing=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = keras.models.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

def train():
    dataset = BinaryClassDataset(txt_path="./train_labels.txt", img_size=224, img_folder="./train_datasets/")
    train_ds, val_ds = dataset.get_data()

    model = create_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath='./output/model_{epoch:02d}_{val_loss:.6f}.keras', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    ]
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
    model.save("./output/final_model.keras")

def predict():
    model = keras.models.load_model("./output/final_model.keras")
    for i in range(len(os.listdir("./test_datasets"))):
        img = cv2.imread(f"./test_datasets/receipt-test-{i}.jpg")
        img = cv2.resize(img, (224, 224))
        img_show = np.copy(img)
        img = img.astype(np.float32)
        img2 = img / 255.0
        img2 = np.expand_dims(img2, axis=0)
        result = model(img2, training=False)

        print(result)

        cv2.imshow("Image", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def predict_video():
    model = keras.models.load_model("./output/final_model.keras")
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("show")
    while True:
        ret, img = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_show = np.copy(img)
        img = img.astype(np.float32)
        img2 = img / 255.0
        img2 = np.expand_dims(img2, axis=0)
        result = model(img2, training=False)
        classification = result[0].numpy()[0]
        cv2.putText(img_show, f"{classification}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("show", img_show)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

if __name__ == "__main__":
    # test()
    # train()
    predict_video()