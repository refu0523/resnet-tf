import tensorflow as tf
from ops import *
from cifar10_reader import *

class ResNet():
    def __init__(self, num_blocks, batch_size=32, epoch=100, img_row=32, img_col=32, depth=3, num_class=10, num_filters=64):
        self.batch_size = batch_size
        self.epoch = epoch
        self.img_row = img_row
        self.img_col = img_col
        self.depth = depth
        self.num_blocks = num_blocks
        self.num_class = num_class  
        self.num_filters = num_filters
        self.X = tf.placeholder(tf.float32, [None, img_row, img_col, depth])
        self.Y = tf.placeholder(tf.float32, [None, num_class])
        self.is_training = tf.placeholder(tf.bool)

    def residual_block(self, input, filters, strides, first_block=False):
        with tf.variable_scope('a'):
            # first block of first layer already done bn->relu in conv1
            if(first_block):
                conv1_1 = conv2d(input, filters, kernel_size=1)
            else:
                conv1_1 = bn_relu_conv(input, self.is_training, filters, kernel_size=1, strides=strides)
                
        with tf.variable_scope('b'):
            conv3_3 = bn_relu_conv(conv1_1, self.is_training, filters, kernel_size=3)
            
        with tf.variable_scope('c'):
            residual = bn_relu_conv(conv3_3, self.is_training, filters*4, kernel_size=1)      
       
        with tf.variable_scope('shortcut'):
            #if shape are different, use 1x1 conv to adjust input dimeension 
            if input.shape[-1] != residual.shape[-1]:
                short_cut = conv2d(input, residual.shape[-1], kernel_size=1, strides=strides)
            else:
                short_cut = input

        return residual + short_cut

    def stage(self, stage, x, filters):
        with tf.variable_scope('conv_{0}'.format(stage)):
            if stage == 1: # first block
                x = max_pooling(x, kernel_size=3, strides=2)
            for i in range(0, self.num_blocks[stage-1]):
                with tf.variable_scope('block_{0}'.format(i+1)):
                    if i == 0 and stage == 1 : # first layer of first block
                        x = self.residual_block(x, filters, strides=1, first_block=True)
                    elif i == 0:  # first layer 
                        x = self.residual_block(x, filters, strides=2)
                    else:
                        x = self.residual_block(x, filters, strides=1)
            return x

    def model(self, x):
        num_filters=self.num_filters

        with tf.variable_scope('conv0'):
            x = conv2d(x, filters=num_filters, kernel_size=7, strides=2, name='conv_1')
            x = batch_normalization(x, self.is_training)
            x = relu(x)

        for i in range(1, len(self.num_blocks)+1):
            x = self.stage(i, x, num_filters)
            num_filters *= 2 

        x = batch_normalization(x, self.is_training)
        x = relu(x)
        x = tf.reduce_mean(x, [1, 2])
        x = dense(x, self.num_class)
        return x

    def fit(self, X_train, y_train, X_test, y_test):
        batch_size = self.batch_size

        #loss
        x = self.model(self.X)
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=self.Y)
        loss = tf.reduce_mean(entropy) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        #accuracy
        preds = tf.nn.softmax(x)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(0,self.epoch):
                #train
                p = np.random.permutation(len(X_train))
                X_train, y_train = X_train[p], y_train[p]
                batch_index = len(X_train)//batch_size
                total_accuracy = 0.0
                total_loss = 0.0
                for i in range(0, batch_index):
                    X_batch =  X_train[i*batch_size:(i + 1) * batch_size]
                    y_batch = y_train[i*batch_size:(i + 1) * batch_size]
                    _, loss_batch, accuracy_batch =sess.run([optimizer, loss, accuracy], 
                    	feed_dict ={self.X: X_batch, self.Y: y_batch, self.is_training: True})
                    total_accuracy += accuracy_batch
                    total_loss += loss_batch
                print('EPOCH:%2d   accuracy:%4f  loss:%4f' % (epoch, total_accuracy/len(X_train), total_loss))
                
                #validate
                #if epoch%5 == 0:
                p = np.random.permutation(len(X_test))
                X_test, y_test = X_test[p], y_test[p]
                batch_index = len(X_test)//batch_size
                total_accuracy = 0.0
                total_loss = 0.0
                for i in range(0, batch_index):
                    X_batch =  X_test[i*batch_size:(i + 1) * batch_size]
                    y_batch = y_test[i*batch_size:(i + 1) * batch_size]
                    _, loss_batch, accuracy_batch =sess.run([optimizer, loss, accuracy],
                    	feed_dict ={self.X: X_batch, self.Y: y_batch, self.is_training: False})
                    total_accuracy += accuracy_batch
                    total_loss += loss_batch
                print('EPOCH:%2d   accuracy:%4f  loss:%4f' % (epoch, total_accuracy/len(X_test), total_loss))
