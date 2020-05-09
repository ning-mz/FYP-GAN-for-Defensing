 
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
import tensorflow as tf
import models.gan_models as models

#set GPU computing
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#training hyper perameters
EPOCHS = 15000
seed_dim = 100
num_examples_to_generate = 16
BUFFER_SIZE = 70000
BATCH_SIZE = 256
DATASET_TYPE = 'MNIST'

#initialize the generator and discriminator of defenseGAN
generator = models.generator()
discriminator = models.dicriminator()
 
#define loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#optimizator
generator_optimizer = tf.keras.optimizers.Adam(5e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)


def prepare_dataset():
    #load training dataset
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = np.vstack((train_images, test_images))
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # [-1, 1]
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset

@tf.function
def train_batch(images):
    seed = tf.random.normal([BATCH_SIZE, seed_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(seed, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        #print(fake_output)

        gen_loss = models.generator_loss(fake_output, cross_entropy)
        disc_loss = models.discriminator_loss(real_output, fake_output, cross_entropy)


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs, global_step, checkpoint, checkpoint_prefix):
    #get the seed for generator test image
    seed = tf.random.normal([num_examples_to_generate, seed_dim])

    for epoch in range(global_step.numpy(), epochs):
        global_step.assign_add(1)
        start = time.time()

        for image_batch in dataset:
            train_batch(image_batch)
 
        #save model for every 500 iters
        if (epoch + 1) % 500 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        #save image for every 100 iters
        if (epoch + 1) % 100 == 0:
            display.clear_output(wait=True)
            save_images(generator,
                             epoch + 1,
                             seed)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    #generate the image of lateset epoch
    display.clear_output(wait=True)


def generate_and_save_images(model, epoch, test_input_seed):
    predictions = model(test_input_seed, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def save_images(model, epoch, input_seed):

    predictions = model(input_seed, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    if DATASET_TYPE == 'MNIST':
        dir_exist = os.path.exists('./visiable/DefenseGAN-MNIST')
        if dir_exist is False:
            os.makedirs('./visiable/DefenseGAN-MNIST')
        plt.savefig('./visiable/DefenseGAN-MNIST/image_at_epoch_{:04d}.png'.format(epoch))


    elif DATASET_TYPE == 'FMNIST':
        dir_exist = os.path.exists('./visiable/DefenseGAN-FMNIST')
        if dir_exist is False:
            os.makedirs('./visiable/DefenseGAN-FMNIST')
        plt.savefig('./visiable/DefenseGAN-FMNIST/image_at_epoch_{:04d}.png'.format(epoch))


def train_gan(dataset_type, epochs):   
    global_step = tf.Variable(0, trainable=False, name='global_step')
    #load check point

    if epochs is not None:
        EPOCHS = epochs

    if dataset_type == 'MNIST':
        checkpoint_dir = './defenseGAN/MNIST'
        DATASET_TYPE = 'MNIST'
    elif dataset_type == 'FMNIST':
        checkpoint_dir = './defenseGAN/FMNIST'
        DATASET_TYPE = 'FMNIST'
    else:
        print('Need to set the dataset type:"MNIST" or "FMNIST"!')
        return

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 global_step=global_step)

    check_exist = tf.train.get_checkpoint_state(checkpoint_dir)
    if check_exist and check_exist.model_checkpoint_path:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        ckptFlag = True
        print('Pre trained DefenseGAN model detected and loaded, continue training.')
        if global_step < EPOCHS:
            train_dataset = prepare_dataset()
            train(train_dataset, EPOCHS, global_step, checkpoint, checkpoint_prefix)
    else:
        print('Start train DefenseGAN')
        train_dataset = prepare_dataset()
        train(train_dataset, EPOCHS, global_step, checkpoint, checkpoint_prefix)

    print('DefenseGAN trainning on {} finished. Total train interation is: {}'.format(dataset_type, epochs))


def get_trained_gen(dataset_type): 
    global_step = tf.Variable(0, trainable=False, name='global_step')

    if dataset_type == 'MNIST':
        checkpoint_dir = './defenseGAN/MNIST'
    elif dataset_type == 'FMNIST':
        checkpoint_dir = './defenseGAN/FMNIST'
    else:
        print('Need to set the dataset type:"MNIST" or "FMNIST"!')
        return

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 global_step=global_step)
    check_exist = tf.train.get_checkpoint_state(checkpoint_dir)

    if check_exist and check_exist.model_checkpoint_path:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        ckptFlag = True
        print('Defense GAN Model loaded: '+dataset_type)

    else:
        print('No pre-trained DefenseGAN model detected, will start train it now')
        train_gan(dataset_type, epochs=15000)

    return generator


def get_tmp_gen(train_dataset, epochs):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    checkpoint_dir = './Tmp-GAN'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 global_step=global_step)
    print('Training Tmp-GAN on given dataset...')
    train(train_dataset, epochs, global_step, checkpoint, checkpoint_prefix)
    return generator
