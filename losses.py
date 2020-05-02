import tensorflow as tf

def wasserstein_loss(y_true, y_pred, reduction=True):
    if reduction:
        return tf.reduce_mean(y_true * y_pred)
    else:
        return y_true * y_pred

def gradient_penalty_loss(average_pred, average_samples, weight=10, reduction=True):
    gradients = tf.gradients(average_pred, average_samples)[0]

    gradients_squared = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_squared, axis=tf.range(1, tf.rank(gradients_squared)))

    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)

    gradient_penalty = weight * tf.square(1 - gradient_l2_norm)

    if reduction:
        return tf.reduce_mean(gradient_penalty)
    else:
        return tf.reduce_mean(gradient_penalty, axis=-1)

def epsilon_penalty_loss(real_pred, weight=0.001):
    return weight * tf.square(real_pred)

def labels_loss(y_true, y_pred, n_labels, weight=1.0):
    y_true = tf.one_hot(y_true, depth=n_labels)
    return weight * tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
