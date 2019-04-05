import tensorflow as tf
import numpy as np
import sys

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer

CIFAR_10_WEIGHTS_PATH = "./keras_cifar10_trained_model_weights.h5"
SINGLE_TRANSFORM_PER_BATCH = True

def load_cifar_encoder(encode_dim=256, weights_path=CIFAR_10_WEIGHTS_PATH):
    cifar10 = cifar10_encoder(encode_dim=encode_dim)
    cifar10.load_weights(weights_path, by_name=True)
    return cifar10


def cifar10_encoder(encode_dim=256):
    model = Sequential()
#     model.add(InputLayer(input_tensor=input_placeholder,
#                      input_shape=(32, 32, 3)))

    model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_1', input_shape=(32, 32, 3)))
    model.add(Activation('relu', name='activation_1'))
    model.add(Conv2D(32, (3, 3), name='conv2d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv2d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(Conv2D(64, (3, 3), name='conv2d_4'))
    model.add(Activation('relu', name='activation_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
    model.add(Dropout(0.25, name='dropout_2'))

    model.add(Flatten(name='flatten_1'))
    model.add(Dense(512, name='dense_1'))
    model.add(Activation('relu', name='activation_5'))
    model.add(Dropout(0.5, name='dropout_3'))
    model.add(Dense(encode_dim, name='dense_encode'))
    model.add(Activation('linear', name='encoding'))

    return model


# In[13]:

def cifar_cnn(input_placeholder, encode_dim=256, reuse=False):
    # Basic CIFAR CNN
    conv = input_placeholder
    with tf.variable_scope('conv_f32', reuse=reuse) as scope:
        for _ in range(2):
            conv = tf.layers.conv2d(conv, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=[2, 2])
        dropout = tf.layers.dropout(pool, rate=0.25)
        input_placeholder = dropout

    conv = input_placeholder
    with tf.variable_scope('conv_f64', reuse=reuse) as scope:
        for _ in range(2):
            conv = tf.layers.conv2d(conv, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=[2, 2])
        dropout = tf.layers.dropout(pool, rate=0.25)
        input_placeholder = dropout

    with tf.variable_scope('dense', reuse=reuse) as scope:
        flatten = tf.layers.flatten(input_placeholder)
        dense = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
        dropout = tf.layers.dropout(pool, rate=0.5)
        input_placeholder = dropout

    # Instead of softmax, do linear encoding
    with tf.variable_scope('encode', reuse=tf.AUTO_REUSE) as scope:
        encoding = tf.layers.dense(input_placeholder, encode_dim)

    return encoding

class SiameseEncoder(object):

    def __init__(self, 
                 cnn_func, 
                 margin,
                 learning_rate=5e-6,
                 momentum=0.9,
                 decay=5e-4,
                 weights_path=CIFAR_10_WEIGHTS_PATH):
        
        # self.init_sess()
        self.weights_path = weights_path

        self.learning_rate = learning_rate
        self.margin = margin

        self.image_1 = tf.placeholder(tf.float32, [None, 32, 32, 3], name="image_1")
        self.image_2 = tf.placeholder(tf.float32, [None, 32, 32, 3], name="image_2")
        self.labels = tf.placeholder(tf.float32, [None], name="labels") # 0 for negative, 1 for positive

        self.cifar_encoder = load_cifar_encoder(weights_path=weights_path)
        self.encoding_1 = self.cifar_encoder(self.image_1)
        self.encoding_2 = self.cifar_encoder(self.image_2)
        
#         self.encoding_1 = cnn_func(self.image_1, reuse=tf.AUTO_REUSE)
#         self.encoding_2 = cnn_func(self.image_2, reuse=tf.AUTO_REUSE)
        
        with tf.variable_scope('training', reuse=tf.AUTO_REUSE) as scope:
            self.l2_distance_squared = tf.square(tf.norm(tf.reshape(self.encoding_1 - self.encoding_2, (tf.shape(self.labels)[0], -1)), axis=-1))
            self.positives_loss = tf.reduce_mean(self.labels * self.l2_distance_squared)
            self.negatives_loss = tf.reduce_mean((1 - self.labels) * tf.maximum(0., margin**2 - self.l2_distance_squared))
            self.loss = self.positives_loss + self.negatives_loss 
            self.update_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.loss)

    def init_sess(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def train(self,
              train_data,
              transform=None,
              transforms=None,
              s=50, 
              epochs=10):

        n = train_data.shape[0]
        n_range = np.arange(n)
        step = n // 1000
        for e in range(epochs):
            print("EPOCH:", e)
            # print("hello")
            perm = np.random.permutation(n)
            data = train_data[perm]
            
            chunk_size = s * 3
            for i in range(0, (n // chunk_size) * chunk_size, chunk_size):
                
                
                # Generate positive pairs
                originals_p = data[i:i+s,:,:,:]

                if transforms is None:
                    positives = transform(originals_p)
                else:
                    if SINGLE_TRANSFORM_PER_BATCH:
                        # Single transform per batch
                        r = np.random.choice(len(transforms))
                        transform = transforms[r]

                        positives = self.sess.run(transform, feed_dict={
                                                        input_transform_ph: originals_p
                                                    }
                                         )
                    else:
                        # Different transform per positive in batch
                        positives = []
                        for k in range(s):
                            r = np.random.choice(len(transforms))
                            transform = transforms[r]

                            positive = originals_p[k,:,:,:]
                            positive = self.sess.run(transform, feed_dict={
                                                        self.input_transform_ph: positive
                                                    }
                                         )
                            positives.append(positive)

                        positives = np.concatenate(positives, axis=0)

                
                # Generate negative pairs
                originals_n = data[i+s:i+s*2,:,:,:]
                negatives = data[i+s*2:i+s*3,:,:,:]

                # One step of SGD
                pos, neg, loss, _ = self.sess.run([self.positives_loss, self.negatives_loss, self.loss, self.update_op],
                              feed_dict={
                                    self.image_1: np.concatenate((originals_p, originals_n), axis=0),
                                    self.image_2: np.concatenate((positives, negatives), axis=0),
                                    self.labels: np.concatenate((np.ones(s), np.zeros(s)))
                                }
                              )
                if np.isnan(loss):
                    print("[ERROR:] Epoch %d, Batch %d/%d, loss = %f, pos = %f, neg = %f" % (e, i, n, loss, pos, neg))
                    break

                if i % step == 0:
                    print("%d/%d samples: loss = %f" % (i, n, loss))
                    # print(l2)
    

    def encode(self, x):
        encoding = self.sess.run(self.encoding_1, feed_dict={self.image_1: x})
        return encoding
    
    def init_weights(self):
        self.cifar_encoder.load_weights(self.weights_path, by_name=True)


# In[14]:

# tf transforms
import estimation_strats
from collections import OrderedDict
def init_transforms():
    # Declare transform tensors
    input_transform_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])

    transforms = []

    for t_name, strat_param in transform_names.items():
        strat = estimation_strats.get_strat(t_name, strat_param)
        transform_tensor = strat.generate_samples(input_transform_ph,
                                                  1,
                                                  (32, 32, 3))
        transforms.append(transform_tensor)

    return [tf.reshape(transform_tensor, (-1,) + (32, 32, 3)) for transform_tensor in transforms], input_transform_ph

transform_names_ = [
    ("uniform", 0.064),
    ("translate", 0.45),
    ("rotate", 0.018),
    ("pixel_scale", 0.17),
    ("crop_resize", 0.04),
    ("brightness", 0.09),
#     ("hue", 0.22),
    ("contrast", 0.55),
#     ("saturation", 0.15),
    ("gaussian_noise", 0.095),
    ("poisson_noise", 0.0065),
    # ("jpeg_compression", 37.5), 
    # ("gamma", 0.35)
]

transform_names = OrderedDict({})
for name, param in transform_names_:
    transform_names[name] = param


TRANSFORMS, input_transform_ph = init_transforms()


# In[ ]:

NP_TRANSFORMS = [
    skimage.filters.gaussian(x_random, sigma=c) 0 to 0.55
    skimage.exposure.adjust_gamma(x_random, gamma=c) 0.73 to 1
]


# In[ ]:

def perform_transform(t, images):
    if t < len(transform_names):
        transform_tensor = TRANSFORMS[t]
        return sess.run(transform_tensor, feed_dict={input_transform_ph: images})
    else:
        t =  t - len(transform_names)
        

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.

    excluded_t = int(sys.argv[1])

    # In[24]:
    transforms = TRANSFORMS[:excluded_t] + TRANSFORMS[excluded_t + 1:] if excluded_t != -1 else TRANSFORMS
    t_names = list(transform_names.keys())
    if excluded_t != -1:
        names = t_names[:excluded_t] + t_names[excluded_t + 1:]
        print("Training on the these transforms:", str(names))

    encoder_name = "./encoder_all.h5" if excluded_t == -1 else "./encoder_no_%s.h5" % t_names[excluded_t]
    

    encoder = SiameseEncoder(cifar_cnn, margin=np.sqrt(10), learning_rate=1e-4)
    encoder.init_sess()
    encoder.init_weights()

    encoder.train(x_train, transforms=all_but_brightness, s=32, epochs=100)
    encoder.cifar_encoder.save_weights(encoder_name)

if __name__ == '__main__':
    main()



