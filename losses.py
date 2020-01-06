import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty_loss(gradients, gradient_penalty_weight=10):
    gradients_squared = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_squared, axis=tf.range(1, tf.rank(gradients_squared)))

    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * tf.square(1 - gradient_l2_norm)
    return tf.reduce_mean(gradient_penalty)
