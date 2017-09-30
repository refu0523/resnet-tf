import tensorflow as tf
from ops import *
image_row = 32
image_col = 32
num_class = 10
X = tf.placeholder(tf.float32, [None,image_row,image_col,3])
Y = tf.placeholder(tf.float32, [None, num_class])
is_train = tf.placeholder(tf.bool)

def residual_block(input, filters, strides, first_block=False):
    with tf.variable_scope('a'):
        # first block of first layer already done bn->relu in conv1
        if(first_block):
            x = conv(input, filters, kernel_size=1)
        else:
            x = bn_relu_conv(input, filters, kernel_size=1)
            
    with tf.variable_scope('b'):
        x = bn_relu_conv(x, filters, kernel_size=3)
        
    with tf.variable_scope('c'):
        x = bn_relu_conv(x, filters*4, kernel_size=1)      
    
    with tf.variable_scope('shortcut'):
        if input.shape[3] != filters:
            short_cut = conv(input, filters*4, strides, kernel_size=1)
        else:
            short_cut = input
    return x + short_cut


num_filters = 64
with tf.variable_scope('conv1', reuse=reuse):
    x = conv(X, filters=num_filters, kernel_size=7, strides=2, name='conv_1')
    x = bn(x,is_train=is_train)
    x = relu(x)
    
with tf.variable_scope('conv2', reuse=reuse):
    x = max_pooling(x, kernel_size=3, strides=2)
    x = residual_block(x, num_filters, strides=1, first_block=True)
    x = residual_block(x, num_filters, strides=1, first_block=False)
    x = residual_block(x, num_filters, strides=1, first_block=False)

x = bn(x)
x = relu(x)
global_pool = tf.reduce_mean(x, [1, 2])
x = tf.layers.dense(x, num_class, activation=tf.nn.softmax)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(0,10):
        

   
