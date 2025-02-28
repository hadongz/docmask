import pandas as pd
import numpy as np
import cv2
import argparse
import os
import json

def convert_parquet(filename, img_count):
    df = pd.read_parquet(f"./parquets/{filename}", engine="pyarrow")

    for i in range(len(df["pixel_values"])):
        image_bytes = df["pixel_values"][i]["bytes"]
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        ds_count = i + img_count
        cv2.imwrite(f"./parquet_images/dataset-{ds_count}.jpg", image)

def rename_dataset_filename(new_name, count):
    files = os.listdir("./parquet_images")
    for file in files:
        os.rename(f"./parquet_images/{file}", f"./parquet_images/{new_name}-{count}.jpg")
        count += 1

def convert_studio_label_json_to_txt(file_name, output_file, is_non_receipt=False):
    with open(f"{file_name}", "r") as json_file:
        json_array = json.load(json_file)
    with open(output_file, 'w') as f:
        for entry in json_array:
            # Process image name
            img_path = entry['img'].replace('\\/', '/')  # Normalize path separators
            filename = os.path.basename(img_path)
            parts = filename.split('-')
            new_filename = '-'.join(parts[1:])  # Remove the first part
            
            # Extract keypoints and original dimensions
            keypoints = entry['kp-1']
            if len(keypoints) == 4:
                original_width = keypoints[0]['original_width']
                original_height = keypoints[0]['original_height']
                
                # Calculate absolute coordinates
                coordinates = []
                for kp in keypoints:
                    x_percent = kp['x']
                    y_percent = kp['y']
                    x_abs = (x_percent / 100) * original_width
                    y_abs = (y_percent / 100) * original_height
                    coordinates.append(f"{round(x_abs, 8):.8f}")
                    coordinates.append(f"{round(y_abs, 8):.8f}")
                
                # Create the line for the txt file
                line = (
                    f"{new_filename},"
                    f"mask-{new_filename},"
                    # f"{','.join(coordinates)}\n"
                    f"1\n"
                )
            elif len(keypoints) != 4 and is_non_receipt:
                line = (
                    f"{new_filename},"
                    f"mask-{new_filename},"
                    f"0\n"
                    f"{','.join([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}\n"
                )
            else:
                continue
            
            f.write(line)

def init_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command", help="subcommands")
    convert_parquet = subparser.add_parser("convert_parquet", help="convert parquets image")
    convert_parquet.add_argument("--filename", required=True)
    convert_parquet.add_argument("--start_count", type=int, required=True)

    rename_files = subparser.add_parser("rename_file", help="rename image file")
    rename_files.add_argument("--filename", required=True)
    rename_files.add_argument("--start_count", type=int, required=True)

    convert_label = subparser.add_parser("convert_label_studio_json", help="rename image file")
    convert_label.add_argument("--filename", required=True)
    convert_label.add_argument("--output", required=True)
    convert_label.add_argument("--is_non_receipt", action="store_true")

    return parser

if __name__ == "__main__":
    parser = init_args()
    args = parser.parse_args()

    if args.command == "convert_parquet":
        convert_parquet(args.filename, args.start_count)
    elif args.command == "rename_file":
        rename_dataset_filename(args.filename, args.start_count)
    elif args.command == "convert_label_studio_json":
        convert_studio_label_json_to_txt(args.filename, args.output, args.is_non_receipt)