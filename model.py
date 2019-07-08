from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

class Models :
    '''
        generator & discriminator model and loss
    '''
    def __init__ (self, call_last_weight = False) :
        self.call_last_weight = call_last_weight
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def make_generator_model (self) :
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (5, 5), strides = (2, 2), padding = 'same', input_shape = [250, 250, 1]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(64))
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', activation = 'tanh'))
        
        return model

    def make_discriminator_model(self) :
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', input_shape = [250, 250, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output) :
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output) :
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)