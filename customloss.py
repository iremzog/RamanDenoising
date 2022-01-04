import tensorflow as tf

def custom_loss(alpha, thr_ratio):
    def peak_mse_combined_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

        thr = tf.reduce_max(y_true, axis=1, keepdims=True) * thr_ratio
        weights = y_true > thr
        weights = tf.cast(weights, tf.float32)
        y_true_peaks = tf.math.multiply(weights, y_true)
        y_pred_peaks = tf.math.multiply(weights, y_pred)

        mse_peak = tf.keras.losses.mean_squared_error(y_true_peaks, y_pred_peaks)

        return mse + mse_peak * alpha

    return peak_mse_combined_loss
