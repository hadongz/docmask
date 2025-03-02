import tensorflow as tf
import keras
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

class HybridLossV3(keras.losses.Loss):
    def __init__(self, alpha=0.7, gamma=2.5, edge_weight=0.3, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha  # Dice vs Focal balance
        self.gamma = gamma  # Focus on hard examples
        self.edge_weight = edge_weight  # Reduced due to attention gates
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """Receive segmentation masks directly (no [0] indexing needed)"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, self.smooth, 1.0 - self.smooth)

        dice_loss = self.dice_loss(y_true, y_pred)
        focal_loss = self.focal_loss(y_true, y_pred)
        boundary_loss = self.boundary_awareness(y_true, y_pred)

        return (self.alpha * dice_loss) + ((1 - self.alpha) * focal_loss) + (self.edge_weight * boundary_loss) 

    def dice_loss(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        loss = 1.0 - (2.*intersection + self.smooth) / (union + self.smooth)
        return tf.reduce_mean(loss)

    def focal_loss(self, y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        log_pt = tf.math.log(pt + self.smooth)
        focal_weight = tf.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * log_pt
        return tf.reduce_mean(loss)

    def boundary_awareness(self, y_true, y_pred):
        edges_true = tf.sqrt(tf.reduce_sum(tf.square(tf.image.sobel_edges(y_true)), axis=-1) + self.smooth)
        edges_pred = tf.sqrt(tf.reduce_sum(tf.square(tf.image.sobel_edges(y_pred)), axis=-1) + self.smooth)
        weighted_smoothness = edges_true * edges_pred
        return tf.reduce_mean(weighted_smoothness)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "edge_weight": self.edge_weight,
            "smooth": self.smooth
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class HybridLossV2(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def hybrid_loss_v2(self, y_true, y_pred):
        """
        Improved hybrid loss function for segmentation with better convergence properties
        Combines Dice loss, Focal BCE, and Edge-aware loss with adjusted weights and stability
        improvements to help escape plateaus.
        """
        # Ensure inputs are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Larger smooth factor for better stability
        smooth = 1e-5
        
        # Clip predictions to prevent extreme values
        y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
        
        # 1. Dice Loss with improved implementation
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        
        # Handle empty masks case
        dice_coef = tf.where(
            union > smooth,
            (2. * intersection + smooth) / (union + smooth),
            tf.ones_like(intersection)  # If both true and pred are empty, dice is 1
        )
        dice_loss = 1.0 - dice_coef
        dice_loss = tf.reduce_mean(dice_loss)
        
        alpha = 0.85
        
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        pt = tf.clip_by_value(pt, smooth, 1.0 - smooth)
        
        gamma = tf.where(pt > 0.7, 1.5, 2.5) 
        focal_weight = tf.pow(1 - pt, gamma)
        
        # Apply alpha weighting
        alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        # Calculate focal BCE
        bce = -tf.math.log(pt)
        focal_bce = focal_weight * alpha_weight * bce
        focal_bce_loss = tf.reduce_mean(focal_bce)
        
        # 3. Edge Loss with simpler implementation (no normalization)
        edge_true = tf.image.sobel_edges(y_true)
        edge_pred = tf.image.sobel_edges(y_pred)
        
        # Calculate magnitude of gradients
        edge_true_mag = tf.sqrt(tf.reduce_sum(tf.square(edge_true), axis=-1) + smooth)
        edge_pred_mag = tf.sqrt(tf.reduce_sum(tf.square(edge_pred), axis=-1) + smooth)
        
        # Simpler L1 edge loss without complex normalization
        edge_loss = tf.reduce_mean(tf.abs(edge_true_mag - edge_pred_mag))
        
        # Combine losses with adjusted weights (emphasizing Dice loss more)
        lambda_dice = 0.7
        lambda_focal = 0.25 
        lambda_edge = 0.05
        total_loss = (lambda_dice * dice_loss) + (lambda_focal * focal_bce_loss) + (lambda_edge * edge_loss)
        
        return total_loss

    def call(self, y_true, y_pred):
        return self.hybrid_loss_v2(y_true, y_pred)

    def get_config(self):
        config = super().get_config()
        return config