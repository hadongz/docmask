import pandas as pd
import numpy as np
import cv2
import argparse
import os
import json
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

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

def clean_metadata_in_folder(folder_path):
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            file_path = os.path.join(folder_path, filename)
            
            # Run exiftool to remove all metadata (overwrite original)
            try:
                subprocess.run(['exiftool', '-all=', '-overwrite_original', file_path], check=True)
                print(f"Processed: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {str(e)}")

def resize_and_compress_image(input_path, output_path, max_size_mb=1, max_dimension=1024):
    # Calculate max file size in bytes
    max_size_bytes = max_size_mb * 1024 * 1024

    # Open the image
    img = Image.open(input_path)
    img_format = img.format

    # Skip if already under the size limit
    if os.path.getsize(input_path) <= max_size_bytes:
        return

    # Resize while maintaining aspect ratio
    width, height = img.size
    if max(width, height) > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        img = img.resize((new_width, new_height), Image.LANCZOS)

    # Handle JPEG and PNG formats
    if img_format == "JPEG":
        # Adjust quality for JPEGs
        quality = 85  # Start with high quality
        buffer = BytesIO()
        while True:
            buffer.seek(0)  # Reset buffer position
            img.save(buffer, "JPEG", quality=quality, optimize=True)
            if buffer.tell() <= max_size_bytes or quality <= 10:
                break
            quality -= 5  # Reduce quality incrementally
    elif img_format == "PNG":
        # Optimize PNGs (lossless compression)
        buffer = BytesIO()
        img.save(buffer, "PNG", optimize=True)
        # If still too large, resize further (PNGs are harder to compress)
        if buffer.tell() > max_size_bytes:
            img = img.resize((new_width // 2, new_height // 2), Image.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, "PNG", optimize=True)
    else:
        raise ValueError(f"Unsupported format: {img_format}")

    # Overwrite the original file
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())

def resize_images_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                resize_and_compress_image(file_path, file_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def create_binary_mask(image_path, polygons, output_mask_path, original_width, original_height):
    # Read image to get current dimensions
    img = Image.open(image_path)
    current_width, current_height = img.size
    img.close()

    # Create a blank mask at original dimensions
    mask = np.zeros((original_height, original_width), dtype=np.uint8)

    # Draw all polygons on the mask (scaled to original dimensions)
    for polygon in polygons:
        # Convert percentage coordinates to pixel values
        scaled_points = [
            (int((x / 100) * original_width), int((y / 100) * original_height)) 
            for x, y in polygon
        ]
        points = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 255)

    # Resize mask to match current image dimensions (if resized)
    if (original_width, original_height) != (current_width, current_height):
        mask = cv2.resize(
            mask,
            (current_width, current_height),
            interpolation=cv2.INTER_NEAREST  # Preserve binary mask
        )

    # Save the mask
    cv2.imwrite(output_mask_path, mask)

def process_label_studio_export(json_path, images_dir, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for item in data:
        _image_name = item['data']['image'].split('/')[-1].split('-')
        image_name = '-'.join(_image_name[1:])
        image_path = os.path.join(images_dir, image_name)
        mask_path = os.path.join(output_dir, f"mask-{image_name}")

        # Extract original dimensions and polygons
        original_width = None
        original_height = None
        polygons = []
        for annotation in item['annotations']:
            for result in annotation['result']:
                if result['type'] == 'polygonlabels':
                    # Get original dimensions from the first annotation
                    if original_width is None:
                        original_width = result['original_width']
                        original_height = result['original_height']
                    # Extract polygon points
                    polygon = result['value']['points']
                    polygons.append(polygon)

        # Create mask
        if polygons:
            create_binary_mask(
                image_path,
                polygons,
                mask_path,
                original_width,
                original_height
            )
            print(f"Mask saved: {mask_path}")
        else:
            print(f"No annotations for: {image_name}")


def visualize_mask(image_path, mask_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    
    # Overlay mask on image (red overlay)
    masked = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masked[mask == 255] = [255, 0, 0]
    
    plt.imshow(masked)
    plt.show()

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