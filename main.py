from data_augmentation import DataAugmentation as DA
from preprocess import DataCollect as DC
from model import Models

import tensorflow as tf
import numpy as np
import pickle
import time
import cv2
import os

from IPython import display 

make_noisy =  True

def save_generated_images() :
    pass

def make_data_noisy (data, scale = 1) :
    shape = data.shape
    noisy = np.random.randn(shape[0], shape[1], shape[2], shape[3])

    data += noisy * scale

    return data

def train_each_step (Model, noise, original, generator, discriminator) :
    batch_epoch = 10
    noise = np.reshape(noise, newshape = (1, 250, 250, 1))
    original = np.reshape(original, newshape = (1, 250, 250, 1))

    for _ in range(batch_epoch) :
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
    time_sum = 0.

    if make_noisy :
        noise = make_data_noisy(noise)

    length = noise.shape[0]

    for epoch in range(epochs) :
        print("epoch {} is working" .format(epoch + 1))
        print("Progress|", "*" * (epoch + 1), " " * (49 - epoch), "|{}%".format(2 * (epoch + 1))) 

        start_time = time.time()

        for idx in range(length) :
            train_each_step(Model = Model, noise = noise[idx], original = original[idx], generator = generator, discriminator = discriminator)

        display.clear_output(wait = True)

        time_tmp = time.time() - start_time
        time_sum += time_tmp

        print("Time for epoch {} is {} sec" .format(epoch + 1, time_tmp))

    display.clear_output(wait = True)

    print("================================================")
    print("Training complete!!")
    print("Time for all epoch({}) is {} sec" .format(epochs, time_sum))
    print("================================================")        

def main() :
    EPOCHS = 50

    Model = Models()

    generator = Model.make_generator_model()
    discriminator = Model.make_discriminator_model()

    train(Model = Model, epochs = EPOCHS, generator = generator, discriminator = discriminator)

    raw_noise = np.array(DC().preprocessed_set()[0], dtype = np.float32)
    raw_noise = make_data_noisy(raw_noise)
    predictions = np.array(generator(raw_noise, training = False))

    for prediction in predictions :
        cv2.imshow("predictions", prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    with open("prediction.pickle", "wb") as f :
        pickle.dump(predictions, f)

if __name__ == '__main__' :
    main()