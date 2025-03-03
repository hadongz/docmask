import tensorflow as tf
import numpy as np
import keras
import os
import cv2
import argparse
from docmask_dataset import DocMaskDataset
from utils import cat_model_summary

from docmask_model import docmask_model, hybrid_loss, train
from docmask_model_v2 import docmask_model_v2, train_v2, boundary_f1, boundary_iou
from loss import HybridLossV3, HybridLossV2

os.environ["KERAS_BACKEND"] = "tensorflow"

def debug_model(model):
    print(model.summary())
    dataset = DocMaskDataset(txt_path="./labels/train_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=1)
    train_ds, val_ds = dataset.load()
    for x, y in train_ds.take(10):
        image = x["img_input"][0].numpy()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output = model(x)
        pred_mask = output[0][0].numpy()
        pred_mask = cv2.normalize(pred_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (224, 224))
        true_mask = y["segmentation_output"]
        true_mask = true_mask[0].numpy()
        true_mask = cv2.normalize(true_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        class_pred = output[1][0].numpy()
        cv2.putText(image, f"{class_pred}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        segmentation_loss = HybridLossV2()
        segment_true = y["segmentation_output"]
        classification_true = y["classification_output"]
        segment_pred = output[0]
        classification_pred = output[1][0]
        segment_loss = segmentation_loss(segment_true, segment_pred)
        classification_loss = keras.losses.binary_crossentropy(classification_true, classification_pred)
        print("SEGMENTATION LOSS ", segment_loss.numpy())
        print("CLASSIFICATION LOSS ", classification_loss.numpy())

        cv2.imshow("image", image)
        cv2.imshow("pred_mask", pred_mask)
        cv2.imshow("true_mask", true_mask)
        cv2.moveWindow("pred_mask", 0, 224) 
        cv2.moveWindow("true_mask", 224, 224) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_model(model):
    print(model.summary())
    dataset = DocMaskDataset(txt_path="./labels/test_labels.txt", img_size=224, img_folder="./test_datasets/", batch_size=1)
    test_ds = dataset.load_test()
    for x, y in test_ds:
        image = x["img_input"][0].numpy()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output = model(x)
        pred_mask = output[0][0].numpy()
        pred_mask = cv2.normalize(pred_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (224, 224))
        true_mask = y["segmentation_output"]
        true_mask = true_mask[0].numpy()
        true_mask = cv2.normalize(true_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        class_pred = output[1][0].numpy()
        cv2.putText(image, f"{class_pred}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        segmentation_loss = HybridLossV2()
        segment_true = y["segmentation_output"]
        classification_true = y["classification_output"]
        segment_pred = output[0]
        classification_pred = output[1][0]
        segment_loss = segmentation_loss(segment_true, segment_pred)
        classification_loss = keras.losses.binary_crossentropy(classification_true, classification_pred)
        print("SEGMENTATION LOSS ", segment_loss.numpy())
        print("CLASSIFICATION LOSS ", classification_loss.numpy())

        cv2.imshow("image", image)
        cv2.imshow("pred_mask", pred_mask)
        cv2.imshow("true_mask", true_mask)
        cv2.moveWindow("pred_mask", 0, 224) 
        cv2.moveWindow("true_mask", 224, 224) 
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
        cv2.moveWindow("Average Feature Map", 0, 224) 
        cv2.imshow("Original Image", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def predict_video(model):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("show")
    cv2.namedWindow("img")

    while True:
        ret, img = cam.read()
        if not ret:
            print("failed to grab frame")
            break
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

        cv2.imshow("show", mask)
        cv2.moveWindow("show", 0, 224) 
        cv2.imshow("img", img_show)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def init_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command", help="Subcommands")
    train = subparser.add_parser("train", help="train the model")
    train.add_argument("-e", "--epoch", type=int, default=500)
    train.add_argument("--no_compile", action="store_false")

    train_v2 = subparser.add_parser("train_v2", help="train the v2 model")
    train_v2.add_argument("-e", "--epoch", type=int, default=100)

    debug = subparser.add_parser("debug", help="debug the model")
    test = subparser.add_parser("test", help="test the model")
    predict = subparser.add_parser("predict", help="predict the model")
    predict_video = subparser.add_parser("predict_video", help="predict the model in video")

    for subparser in [train, train_v2, debug, test, predict, predict_video]:
        subparser.add_argument("-mp", "--model_path", help="model path")
        subparser.add_argument("--v1", action='store_true')
        subparser.add_argument("--v2", action='store_true')
    return parser


if __name__ == "__main__": 
    parser = init_args()
    args = parser.parse_args()

    if args.model_path:
        model = keras.models.load_model(args.model_path, safe_mode=False)
    elif args.v1:
        model = docmask_model()
    elif args.v2:
        model = docmask_model_v2()

    if args.command == "train":
        epoch = args.epoch
        need_compile = args.no_compile
        train(model, epoch=epoch, need_compile=need_compile)
    elif args.command == "train_v2":
        epoch = args.epoch
        train_v2(model, epoch=epoch)
    elif args.command == "debug":
        debug_model(model)
    elif args.command == "test":
        test_model(model)
    elif args.command == "predict":
        predict(model)
    elif args.command == "predict_video":
        predict_video(model)