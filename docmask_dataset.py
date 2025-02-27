import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
import numpy as np
import tensorflow_models as tfm
import math

class DocMaskDataset:
    def __init__(self, txt_path="./labels/mask_labels.txt", img_size=224, img_folder="./train_datasets/", val_ratio=0.2, batch_size=32):
        """
        Initialize the document dataset.
        
        Args:
            txt_path: Path to text file containing image paths and corner coordinates
            img_size: Target image size (default: 224 for MobileNetV3Small)
            img_folder: Base folder for images
            val_ratio: Validation split ratio
            batch_size: Batch size for training
        """
        self.txt_path = txt_path
        self.img_folder = img_folder
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_size = 0
        self.val_size = 0

    def decode_img(self, img_path, img_size, channels=3):
        """Decode and preprocess image"""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=channels)
        img = tf.image.resize(img, [img_size, img_size])
        return img
    
    def analysis_line(self, line):
        """Process a single line from the dataset file"""
        parts = tf.strings.split(line, ",")
        img_path = self.img_folder + parts[0]
        mask_path = self.img_folder + parts[1]
        img = self.decode_img(img_path, self.img_size)
        mask = self.decode_img(mask_path, self.img_size, channels=1)
        class_label = tf.strings.to_number(parts[2])
        return ({"img_input": img}, {"segmentation_output": mask, "classification_output": class_label})
    
    def augment(self, img, labels):
        """Apply augmentation to image and adjust corner coordinates"""
        
        # img = data[0]["img_input"]
        # labels = data[1]
        img = img["img_input"]
        mask = labels["segmentation_output"]
        class_label = labels["classification_output"]

        # Random brightness
        if tf.random.uniform([]) < 0.5:
            img = tf.image.random_brightness(img, 0.2)
        
        # Random contrast
        if tf.random.uniform([]) < 0.3:
            img = tf.image.random_contrast(img, 0.8, 1.2)
        
        # Random hue
        if tf.random.uniform([]) < 0.3:
            img = tf.image.random_hue(img, 0.1)
        
        # Cutout augmentation (simulates occlusion)
        if tf.random.uniform([]) < 0.3:
            cutout_size = tf.random.uniform(shape=(), minval=10, maxval=40, dtype=tf.int32)
            img = tfm.vision.augment.cutout(img, cutout_size, 255)
        
        # Flip augmentation
        if tf.random.uniform([]) < 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)

        # Random padding
        if tf.random.uniform([]) < 0.5:
            # Randomly sample padding sizes
            top = tf.random.uniform((), minval=0, maxval=80, dtype=tf.int32)
            bottom = tf.random.uniform((), minval=0, maxval=80, dtype=tf.int32)
            left = tf.random.uniform((), minval=0, maxval=80, dtype=tf.int32)
            right = tf.random.uniform((), minval=0, maxval=80, dtype=tf.int32)
            
            # Define padding
            padding = [[top, bottom], [left, right], [0, 0]]
            
            # Apply zero padding
            img_size = self.img_size
            img = tf.pad(img, padding, mode="REFLECT")
            mask = tf.pad(mask, padding, mode="CONSTANT")
            img = tf.image.resize(img, [img_size, img_size])
            mask = tf.image.resize(mask, [img_size, img_size])
        
        # Rotation augmentation (small angles only)
        if tf.random.uniform([]) < 0.8:
            angle = tf.random.uniform([], minval=-90, maxval=90)
            img = self.img_rotate(img, angle)
            mask = self.img_rotate(mask, angle, fill_mode='constant')

        mask /= 255.0
        return ({"img_input": img}, {"segmentation_output": mask, "classification_output": class_label})

    def img_rotate(self, image, degrees, fill_mode='reflect'):
        # Convert from degrees to radians.
        degrees_to_radians = math.pi / 180.0
        radians = tf.cast(degrees * degrees_to_radians, tf.float32)

        original_ndims = tf.rank(image)
        image = tfm.vision.augment.to_4d(image)

        image_height = tf.cast(tf.shape(image)[1], tf.float32)
        image_width = tf.cast(tf.shape(image)[2], tf.float32)
        transforms = tfm.vision.augment._convert_angles_to_transform(angles=radians, image_width=image_width, image_height=image_height)
        # In practice, we should randomize the rotation degrees by flipping
        # it negatively half the time, but that's done on 'degrees' outside
        # of the function.
        image = tfm.vision.augment.transform(image, transforms=transforms, fill_mode=fill_mode)
        return tfm.vision.augment.from_4d(image, original_ndims)
    
    def configure_for_performance(self, ds):
        """Configure dataset for optimal performance"""
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(self.augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    
    def generate_masks(self):
        import cv2

        with open(self.txt_path, "r") as fr:
            lines = fr.readlines()
            lines = [line.strip() for line in lines]
            self.data_size = len(lines)

        for i in range(self.data_size):
            img_name = lines[i].split(",")[0]
            img_path = self.img_folder + img_name
            img = cv2.imread(img_path)
            img_height = img.shape[0]
            img_width = img.shape[1]
            corners = lines[i].split(",")[1:]
            corners = [int(float(corner)) for corner in corners]
            mask_name = "mask-" + img_name
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            if np.mean(np.array(corners)) != 1:
                cv2.fillPoly(mask, [np.array(corners).reshape(-1, 2)], color=(255, 255, 255))
            cv2.imwrite(f"./train_datasets/{mask_name}", mask)

    
    def load(self):
        # Create dataset from lines
        with open(self.txt_path, "r") as fr:
            lines = fr.readlines()
            lines = [line.strip() for line in lines]
            self.data_size = len(lines)

        np.random.shuffle(lines)
        file_lines = ops.convert_to_tensor(lines, dtype=dtypes.string)
        dataset = tf.data.Dataset.from_tensor_slices(file_lines)
        
        # Split into train and validation
        self.val_size = int(self.data_size * self.val_ratio)
        dataset = dataset.shuffle(self.data_size, reshuffle_each_iteration=False)
        dataset_train = dataset.skip(self.val_size)
        dataset_val = dataset.take(self.val_size)
        
        # Process datasets
        train_ds = dataset_train.map(self.analysis_line, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = dataset_val.map(self.analysis_line, num_parallel_calls=tf.data.AUTOTUNE)

        return self.configure_for_performance(train_ds), self.configure_for_performance(val_ds)

if __name__ == "__main__":
    import cv2

    dataset = DocMaskDataset(txt_path="./labels/train_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=1)
    train_ds, val_ds = dataset.load()

    for x, y in train_ds.take(10):
        image = x["img_input"][0].numpy().astype(np.uint8)
        mask = y["segmentation_output"][0].numpy().astype(np.uint8)
        mask *= 255
        classification = y["classification_output"][0].numpy()
        cv2.putText(mask, f"{classification}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", image)
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()