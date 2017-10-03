import tensorflow as tf



def conv2d(inputs, filters, kernel_size=3, strides=1, padding="SAME", name="conv_2d"):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                            name=name)

def batch_normalization(x, is_train, scope='bn'):
    with tf.variable_scope(scope):
        n_out = x.shape[3]
        batch_mean, batch_variance = tf.nn.moments(x, axes=[0, 1, 2])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta')
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_variance])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_variance)
        
        mean, var = tf.cond(is_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_variance)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

def relu(x):
    return tf.nn.relu(x)

def bn_relu_conv(x, is_train, filters, kernel_size, strides=1):
    x = batch_normalization(x, is_train)
    x = relu(x)
    x = conv2d(x, filters, kernel_size, strides)
    return x

def max_pooling(x, kernel_size=3, strides=2, padding='SAME'):
    return tf.nn.max_pool(x,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, strides, strides, 1],
                          padding='SAME')
    
def dense(x, num_labels):
    x = tf.layers.dense(x, num_labels, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
    return x