import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
import numpy as np
import tensorflow_models as tfm

class BinaryClassDataset:
    def __init__(self, txt_path, img_size=224, img_folder="", val_ratio=0.2, batch_size=32):
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
    
    @staticmethod
    def decode_img(img_path, img_size):
        """Decode and preprocess image"""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        return img / 255.0
    
    def analysis_line(self, line):
        """Process a single line from the dataset file"""
        parts = tf.strings.split(line, ",")
        img_path = self.img_folder + parts[0]
        img = self.decode_img(img_path, self.img_size)
        return img, tf.strings.to_number(parts[1])
    
    @staticmethod
    def augment(img, label):
        """Apply augmentation to image and adjust corner coordinates"""
        # Convert preprocessed image back to 0-255 range for augmentations
        class_label = label
        img = img * 255.0
        
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
            cutout_size = tf.random.uniform(shape=(), minval=10, maxval=50, dtype=tf.int32)
            img = tfm.vision.augment.cutout(img, cutout_size, 255)
        
        # Rotation augmentation (small angles only)
        if tf.random.uniform([]) < 0.5:
            angle = tf.random.uniform([], minval=-20, maxval=20)
            img = tfm.vision.augment.rotate(img, angle)
        
        # Random padding
        if tf.random.uniform([]) < 0.5:
            orig_width = img.shape[0]
            orig_height = img.shape[1]
            img = BinaryClassDataset.random_padding(img)
            img = tf.image.resize(img, [orig_width, orig_height])

        return img / 255.0, class_label
    
    @staticmethod
    def random_padding(img, max_ratio=0.5):
        ratio = tf.random.uniform(shape=[4], minval=0., maxval=max_ratio)
        img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
        size_change = tf.round(ratio * tf.cast(tf.concat([img_shape[0:2], img_shape[0:2]], axis=0), dtype=dtypes.float32))
        new_height = img_shape[0] + size_change[0] + size_change[2]
        new_width = img_shape[1] + size_change[1] + size_change[3]
        img = tf.image.pad_to_bounding_box(
            img,
            tf.cast(size_change[0], dtype=tf.int32),
            tf.cast(size_change[1], dtype=tf.int32),
            tf.cast(new_height, dtype=tf.int32),
            tf.cast(new_width, dtype=tf.int32),
        )
        
        return img
    
    def configure_for_performance(self, ds, aug=False):
        """Configure dataset for optimal performance"""
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        if aug:
            ds = ds.map(self.augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
            
    def get_data(self):
        """Get training and validation datasets"""
        with open(self.txt_path, "r") as fr:
            lines = fr.readlines()
            lines = [line.strip() for line in lines]
            self.data_size = len(lines)
        
        # Create dataset from lines
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

        return self.configure_for_performance(train_ds, True), self.configure_for_performance(val_ds, True)
    
if __name__ == "__main__":
    import cv2

    dataset = BinaryClassDataset(txt_path="./train_labels.txt", img_size=224, img_folder="./train_datasets/", batch_size=1)
    train_ds, val_ds = dataset.get_data()

    for x, y in train_ds.take(5):
        image = x[0]
        image = cv2.normalize(image.numpy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(y)