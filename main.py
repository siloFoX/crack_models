from data_augmentation import DataAugmentation as DA
from preprocess import DataCollect as DC
from model import Models

import tensorflow as tf
import numpy as np
import time
import pickle
import os

from IPython import display 

def save_generated_images() :
    pass

def train_each_step (Model, noise, original, generator, discriminator) :
    noise = np.reshape(noise, newshape = (1, 250, 250, 1))
    original = np.reshape(original, newshape = (1, 250, 250, 1))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape :
        generated_images = generator(noise)
        
        real_output = discriminator(original, training = True)
        fake_output = discriminator(generated_images, training = True)

        gen_loss = Model.generator_loss(fake_output)
        dis_loss = Model.discriminator_loss(real_output, fake_output)

    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_dis = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

    Model.optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    Model.optimizer.apply_gradients(zip(gradient_dis, discriminator.trainable_variables))

def train(Model, epochs, generator, discriminator) :
    dataloader = DA()
    noise, original = dataloader.rotated_data()

    length = noise.shape[0]

    for epoch in range(epochs) :
        print("epoch {} is working" .format(epoch + 1))
        print("Progress|", "*" * (epoch + 1), " " * (49 - epoch), "|{}%".format(2 * (epoch + 1)))

        start_time = time.time()

        for idx in range(length) :
            train_each_step(Model = Model, noise = noise[idx], original = original[idx], generator = generator, discriminator = discriminator)

        display.clear_output(wait = True)

        print("Time for epoch {} is {} sec" .format(epoch + 1, time.time() - start_time))
            

def main() :
    EPOCHS = 50

    Model = Models()

    generator = Model.make_generator_model()
    discriminator = Model.make_discriminator_model()

    train(Model = Model, epochs = EPOCHS, generator = generator, discriminator = discriminator)

    raw_noise = np.array(DC().preprocessed_set()[0], dtype = np.float32)

    predictions = generator(raw_noise, training = False)

    with open("prediction.pickle", "wb") as f :
        pickle(predictions, f)

if __name__ == '__main__' :
    main()


'''
WARNING: Logging before flag parsing goes to stderr.
W0707 22:00:56.441749 17464 deprecation.py:323] From C:\Users\ladna\Anaconda3\lib\site-packages\tensorflow\python\ops\nn_impl.py:182: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
'''