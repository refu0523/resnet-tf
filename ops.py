import tensorflow as tf



def conv2d(inputs, filters, kernel_size=3, strides=1, padding = "SAME", name="conv_2d"):
	return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,\
	 padding=padding,name=name)

def batch_normalization(x, n_out=4, is_train=True, scope='bn'):
	with tf.variable_scope(scope):
		batch_mean, batch_variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta')
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

def relu(x):
	return tf.nn.relu(x)

def bn_relu_conv(x, filters, kernel_size, strides):
	x = batch_normalization(x)
	x = relu(x)
	x = conv2d(x, filters, kernel_size, strides)
	return x

