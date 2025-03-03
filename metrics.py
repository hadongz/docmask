import tensorflow as tf
import keras
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

@keras.saving.register_keras_serializable()
def boundary_iou(y_true, y_pred, dilation_radius=3):
    """Calculate Intersection-over-Union for mask boundaries"""
    # Ensure input is 4D (batch, height, width, channels)
    y_true = tf.ensure_shape(y_true, (None, None, None, 1))
    y_pred = tf.ensure_shape(y_pred, (None, None, None, 1))

    # Create 4D erosion kernel [height, width, in_channels, out_channels]
    kernel = tf.ones((dilation_radius, dilation_radius, 1), dtype=tf.float32)

    # Erosion operations with proper data_format
    y_true_eroded = tf.nn.erosion2d(
        value=y_true,
        filters=kernel,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',  # Channels-last format
        dilations=[1, 1, 1, 1],
        name='y_true_erosion'
    )

    y_pred_eroded = tf.nn.erosion2d(
        value=y_pred,
        filters=kernel,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',  # Channels-last format
        dilations=[1, 1, 1, 1],
        name='y_pred_erosion'
    )

    # Calculate boundary regions
    y_true_boundary = y_true - y_true_eroded
    y_pred_boundary = y_pred - y_pred_eroded

    # Compute IoU
    intersection = tf.reduce_sum(y_true_boundary * y_pred_boundary)
    union = tf.reduce_sum(y_true_boundary) + tf.reduce_sum(y_pred_boundary) - intersection
    
    return (intersection + 1e-6) / (union + 1e-6)

@keras.saving.register_keras_serializable()
def boundary_f1(y_true, y_pred, threshold=0.5):
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    edges_true = tf.image.sobel_edges(y_true)
    edges_pred = tf.image.sobel_edges(y_pred_bin)
    
    tp = tf.reduce_sum(edges_true * edges_pred)
    fp = tf.reduce_sum(edges_pred) - tp
    fn = tf.reduce_sum(edges_true) - tp
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return 2 * (precision * recall) / (precision + recall + 1e-6)