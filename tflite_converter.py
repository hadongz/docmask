import tensorflow as tf
import numpy as np
import keras
import cv2
import os
import argparse
from docmask_model import hybrid_loss
from loss import HybridLossV2, HybridLossV3

def predict_tflite_model(model_name):
    interpreter = tf.lite.Interpreter(f"./saved_model/{model_name}.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(os.listdir("./predict_datasets"))):
        img = cv2.imread(f"./predict_datasets/receipt-test-{i}.jpg")
        img = cv2.resize(img, (224, 224))
        img_show = np.copy(img)
        img = img.astype(np.float32)
        img2 = img
        img2 = np.expand_dims(img2, axis=0)
        
        interpreter.set_tensor(input_details[0]["index"], img2)
        interpreter.invoke()
        output_data = []
        
        for output in output_details:
            output_data.append(interpreter.get_tensor(output['index']))

        output = output_data[1][0]
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        output = cv2.resize(output, (224, 224))
        cv2.imshow("Image", img_show)
        cv2.imshow("Mask", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def predict_tflite_model_video(model_name):
    cam = cv2.VideoCapture(0)
    interpreter = tf.lite.Interpreter(f"./saved_model/{model_name}.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
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

        interpreter.set_tensor(input_details[0]["index"], img2)
        interpreter.invoke()
        output_data = []
        
        for output in output_details:
            output_data.append(interpreter.get_tensor(output['index']))

        output = output_data[1][0]
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        output = cv2.resize(output, (224, 224))

        class_label = output_data[0][0]
        cv2.putText(img_show, f"{class_label}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("show", output)
        cv2.imshow("img", img_show)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

def convert_to_tflite(model_name):
    model = keras.models.load_model(f"./saved_model/{model_name}.keras", safe_mode=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(f"./saved_model/{model_name}.tflite", 'wb') as f:
        f.write(tflite_model)

def init_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    convert = subparsers.add_parser("convert", help="convert model to tflite")
    predict_model = subparsers.add_parser("predict_model", help="predict tflite model to images")
    predict_video = subparsers.add_parser("predict_video", help="preidct tflite model to cam")
    for subparser in [convert, predict_model, predict_video]:
        subparser.add_argument("-m", "--model_name", required=True, help="model name")
    return parser

if __name__ == "__main__":
    parser = init_args()
    args = parser.parse_args()
    model_name = args.model_name
    if args.command == "convert":
        convert_to_tflite(model_name)
        predict_tflite_model(model_name)
        predict_tflite_model_video(model_name)
    elif args.command == "predict_model":
        predict_tflite_model(model_name)
    elif args.command == "predict_video":
        predict_tflite_model_video(model_name)
