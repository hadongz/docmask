import tensorflow as tf
import numpy as np
import keras
import os
import cv2
import argparse
from docmask_dataset import DocMaskDataset
from utils import cat_model_summary

from docmask_model import docmask_model, hybrid_loss, train
from docmask_model_v2 import docmask_model_v2, hybrid_loss_v2, train_v2

os.environ["KERAS_BACKEND"] = "tensorflow"

def debug_model(model):
    print(model.summary())
    model.summary(print_fn=cat_model_summary)
    dataset = DocMaskDataset(txt_path="./labels/train_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=1)
    train_ds, val_ds = dataset.load()
    for x, y in train_ds.take(3):
        image = x["img_input"][0].numpy()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(y["classification_output"])

        output = model(x)
        mask = output[0][0].numpy()
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask = cv2.resize(mask, (224, 224))

        class_pred = output[1][0].numpy()
        cv2.putText(image, f"{class_pred}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Image", image)
        cv2.imshow("Mask", mask)
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
        mask = result[0][0].numpy()
        class_pred = result[1][0].numpy()
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.putText(img_show, f"{class_pred}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("Average Feature Map", mask)
        cv2.imshow("Original Image", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def init_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command", help="Subcommands")
    train = subparser.add_parser("train", help="train the model")
    train.add_argument("-e", "--epoch", type=int, default=500)
    train.add_argument("--no_compile", action="store_false")

    train_v2 = subparser.add_parser("train_v2", help="train the v2 model")
    train_v2.add_argument("-e", "--epoch", type=int, default=100)

    debug = subparser.add_parser("debug", help="debug the model")
    predict = subparser.add_parser("predict", help="predict the model")

    for subparser in [train, train_v2, debug, predict]:
        subparser.add_argument("-mp", "--model_path", help="model path")
    return parser


if __name__ == "__main__": 
    parser = init_args()
    args = parser.parse_args()

    if args.model_path:
        model = keras.models.load_model(args.model_path)
    else:
        model = docmask_model()

    if args.command == "train":
        epoch = args.epoch
        need_compile = args.no_compile
        train(model, epoch=epoch, need_compile=need_compile)
    elif args.command == "train_v2":
        model = docmask_model_v2()
        epoch = args.epoch
        train_v2(model, epoch=epoch)
    elif args.command == "debug":
        debug_model(model)
    elif args.command == "predict":
        predict(model)