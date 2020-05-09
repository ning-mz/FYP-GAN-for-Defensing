from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ReLU, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import numpy as np
import matplotlib.pyplot as plt
import defenseGAN as gan
import models.cnn_models as cnn_models
import evaluation as evaluation
import math

#set GPU ON
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


EPHOCS = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28
BUFFER_SIZE = 30000
BATCH_SIZE = 256


#The funciton for show examples
def visiable_images(model,model_type,test_input, title, save=False):
  predictions = model.predict(test_input)
  fig = plt.figure(figsize=(7,7))
  plt.suptitle('{}'.format(title), fontsize=20)
    
  for i in range(test_input.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(test_input[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
    plt.title('Class:{}, {:0.2f}%'.format(np.argmax(predictions[i])
                , np.max(predictions[i])))

  if save:
    plt.show()
  else:
    dir_exist = os.path.exists('./visiable/black_box/'+model_type)
    if dir_exist is False:
      os.makedirs('./visiable/black_box/'+model_type)
    plt.savefig('./visiable/black_box/'+model_type+'/{}.png'.format(title))


def fgsm_adversarial_advanced(model,input_image,epsilon,rand,alpha=0.02,rate=1.0):
  adv_nums = int(input_image.shape[0] * rate)
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
   
  target_image = input_image[:adv_nums]
  input_label = model.predict_classes(target_image)
  if rand:
    noise = tf.convert_to_tensor(
      np.random.uniform(-alpha, alpha, (target_image.numpy().shape))
      #np.clip(alpha * np.sign(np.random.randn(*target_image.numpy().shape)),-1.0,1.0)
      )
    target_image = tf.clip_by_value(
            target_image + tf.cast(noise, dtype=tf.float32),
            -1.0, 1.0)
    epsilon = epsilon - alpha

  with tf.GradientTape() as tape:
    tape.watch(target_image)
    prediction = model(target_image)
    loss = loss_object(input_label, prediction)
  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, target_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)

  adv = target_image + epsilon * signed_grad
  input_image_np = input_image.numpy()
  input_image_np[:adv_nums] = adv.numpy()
  
  adv_all = tf.convert_to_tensor(input_image_np)

  print('\nAdversarial examples generated:')
  print('Total number of images is: {}'.format(input_image.shape[0]))
  if rand:
    print('\tRand + FGSM be used.')
  print('Number for adversarial examples is: {}, epsilon is: {}'.format(adv_nums, epsilon))

  return adv, adv_all


def display_images(model, image, description):
  label = model.predict_classes(image)
  confidence = np.max(model.predict(image))
  plt.figure()
  plt.imshow(image[0].numpy().reshape(28,28))
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()


def prepare_data(hold_out, data_set_type):
  #Data prepatation
  train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
  validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
  attack_image_generator = ImageDataGenerator(rescale=1./255) #Generator for out attack data

  #load data of fashion mnist data
  
  if data_set_type == 'MNIST':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
  elif data_set_type == 'FMNIST':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
  
  train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
  train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
  #train_images = train_images / 255


  test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
  test_images = (test_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
  #test_images = test_images / 255

  #prepare the sub data set for training substitute model
  images_sub = test_images[: hold_out]
  labels_sub_o = test_labels[: hold_out]
  #make the label array can be rewrite
  labels_sub = labels_sub_o.copy()
  #test_labels_sub.flags.writeable = True

  test_images = test_images[hold_out :]
  test_labels = test_labels[hold_out :]

  return train_images, train_labels, test_images, test_labels, images_sub, labels_sub


def get_trained_cnn(data_set_type,model_type, batchNorm, train_images, train_labels, test_images, test_labels, Retrain=False): 
  '''
  return the trained cnn model
  if the model is not trained previously, train the model
  if the model was trained, load the model
  '''
  #Load model amd compile
  model = cnn_models.get_model(model_type, batchNorm)
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  #see details of target model
  model.summary()

  checkpoint_dir = './black_box/'+data_set_type+'/'+model_type
  checkpoint_prefix = os.path.join(checkpoint_dir,model_type)
  checkpoint = tf.train.Checkpoint(targetModel=model)
  ckptFlag = False

  #check whether there exist checkpoint
  check_exist = tf.train.get_checkpoint_state(checkpoint_dir)
  if check_exist and check_exist.model_checkpoint_path:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    ckptFlag = True
    print('Trained CNN Model Loaded, model is: {}, dataset is: {}'.format(model_type,data_set_type))
  else:
    model.fit(train_images,train_labels,
          batch_size=BATCH_SIZE,
          epochs=EPHOCS,
          validation_data=(test_images,test_labels))

    checkpoint.save(file_prefix=checkpoint_prefix)

  model.trainable = False
  return model


def train_sub_model(target_model,sub_model_type,batchNorm,aug_times,images_sub, 
                    labels_sub, test_images, test_labels):
  #function for augmentate adversary training of substitue model
  sub_model = cnn_models.get_model(sub_model_type, batchNorm)
  sub_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  print('Substitute Model Summary:')
  sub_model.summary()

  #train the substitute model
  for num in range(aug_times):
    sub_model.fit(images_sub, labels_sub,
          batch_size=BATCH_SIZE,
          epochs=10,
          validation_data=(test_images,test_labels))

    if num < aug_times-1:
      print("Augmenting substitute training data. Time: {}".format(num))  
      previous_num = images_sub.shape[0] - 1
      X_sub = np.vstack([images_sub, images_sub])
      Y_sub = np.hstack([labels_sub, labels_sub])
      
      for index, image in enumerate(images_sub):
        image = image[None, ...]
        image = tf.convert_to_tensor(image)
        image = tf.cast(image, tf.float32)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
          tape.watch(image)
          prediction = sub_model(image)
          num = labels_sub[index].astype(int)
          #print(prediction[:,num])
          #loss = prediction[:,num]
          loss = loss_object(prediction[:,num],prediction)

        gradient_rec = tape.gradient(loss, image)
        signed_grad = tf.sign(gradient_rec)

        aug_image = image + 0.1*signed_grad

        X_sub[previous_num + index] = aug_image[0]
        Y_sub[previous_num + index] = target_model.predict_classes(aug_image)
        #np.argmax(target_model(aug_image), axis=1)

      images_sub = X_sub
      labels_sub = Y_sub

      #visiable_images(target_model, images_sub[images_sub.shape[0]-17:images_sub.shape[0]-1], 'Adversarial Examples for Target Model',save=False)


  #train the substitute model
  print('Train substitute model finished.')
  sub_model.trainable = False
  return sub_model


def do_rec_images(target_model, generator, adv_images, rr, rec_L, rec_batch_size):
  num_of_images = adv_images.shape[0]
  batch_nums = int(math.ceil(float(num_of_images/rec_batch_size)))
  print('Reconstructing adversarial...')

  result = []

  for num in range(batch_nums): 
    #print('Rec batch is:', num)
    ind_start = num * rec_batch_size
    ind_end = min(num_of_images, (num+1) * rec_batch_size)
    count = ind_end - ind_start
    rec_images_batch = tf.reshape(adv_images[ind_start:ind_end], [count, np.prod(adv_images.shape[1:])])
    rec_images_batch = tf.tile(rec_images_batch, [1,rr])
    rec_images_batch = tf.reshape(rec_images_batch, [count*rr, 28, 28, 1])
  
    #rec_iamges_batch = tf.tile(adv_images, [1, rr, 1, 1])
    #visiable_images(target_model, rec_images_batch[0:16], 'test', save=False)

    #batch_nums = int(math.ceil(float(num_of_images / rr)))
        

    #for num in range(batch_nums):
    #lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-1, 300, 0.001, staircase=True)
    #rec_optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.7, name='rec_optimizer')
    rec_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1, beta_1=0.5, epsilon=1e-5)
    #generate random seed for reconstruct
    z_hat = tf.Variable(initial_value=tf.random.normal([rr*count, 100]),
                    name='z_hat',
                    trainable=True,
                    dtype=tf.float32,
                    shape=[rr*count, 100]
                )

    # def get_loss():
    #   rec_loss = tf.reduce_sum(tf.reduce_mean(tf.square(rec_image - adv_images), axis=1))
    #   return rec_loss

    for i in range(0,rec_L):
      with tf.GradientTape() as rec_type:
        rec_type.watch(z_hat)
        rec_image = generator(z_hat, training=False)
      
        rec_loss = tf.reduce_sum(tf.reduce_mean(
                    tf.square(rec_image-rec_images_batch), 
                    axis=range(1, len(rec_image.get_shape()))))
    
        #rec_gradients = rec_optimizer.minimize(get_loss,[z_hat])
      rec_gradients = rec_type.gradient(rec_loss, [z_hat])
      rec_optimizer.apply_gradients(zip(rec_gradients, [z_hat]))

    rec_done_images = generator(z_hat, training=False)

        
    for i in range(count):
      ind = i*rr
      adv_batch = rec_images_batch[ind:ind+rr]
      rec_done_batch = rec_done_images[ind:ind+rr]
    
  
      loss = tf.reduce_mean(tf.square(rec_done_batch-adv_batch),
                          axis=range(1, len(rec_done_batch.get_shape())))

      ind = tf.argmin(loss)  
      result.append(rec_done_batch[ind])
        

  images = tf.stack(result)
  #visiable_images(target_model, images, 'Rec Images of Adversarial Examples', save=False)
  return images


def black_box(gan_type,data_set_type,target_model_type, sub_model_type, hold_out,test_nums, 
              adv_rate=1.0,epsilon=0.1,rr=10,rec_L=300,visiable=False,rand_fgsm=False):
  train_images, train_labels, test_images, test_labels, images_sub, labels_sub = prepare_data(hold_out, data_set_type)

  target_model = get_trained_cnn(data_set_type,target_model_type,True, 
                                  train_images, train_labels, test_images, test_labels)

  sub_model = train_sub_model(target_model, sub_model_type, True,
                              7, images_sub, labels_sub, test_images, test_labels)

  #---------------------------try the adverserial attack----------------------------#
  start_ind = np.random.randint(0,test_images.shape[0]-1-test_nums,size=1)[0]
  image = test_images[start_ind : start_ind+test_nums]
  label = test_labels[start_ind : start_ind+test_nums]
  image = tf.convert_to_tensor(image)
  image = tf.cast(image, tf.float32)
  adv_nums = int(image.shape[0] * adv_rate)
  #visiable_images(target_model, image, 'Test Examples for Target Model',save=True)
  #visiable_images(sub_model, image, 'Test Examples for Substitute Model', save=True)

  if gan_type == 1:
    #will make all the test images as adversarial examples

    adv_x, adv_all = fgsm_adversarial_advanced(sub_model,image,epsilon=epsilon,rand=rand_fgsm,alpha=0.02,rate=1)
    #adv_x = tf.clip_by_value(adv_x, -1.0, 1.0)
    visiable_images(target_model, target_model_type, adv_x[:16], 'Adversarial Examples for Target Model',save=visiable)

    ori_accuracy = evaluation.eval(target_model, label, image)
    print('Accuracy for {} image Original is: {:0.3f}'.format(test_nums,ori_accuracy))

    adv_accuracy = evaluation.eval(target_model, label, adv_x)
    print('Accuracy for {} image in Adversarial is: {:0.3f}'.format(test_nums,adv_accuracy))

    gen_GAN = gan.get_trained_gen(data_set_type)
    rec_image = do_rec_images(target_model, gen_GAN, adv_x, rr=rr, rec_L=rec_L,rec_batch_size=256)
    rec_accuracy, rec_loss = evaluation.eval_gan(target_model,label, rec_image, image)
    print('Accuracy for {} image Rec is: {:0.3f}, Rec Loss in average is: {:0.3f}'.format(test_nums,rec_accuracy, rec_loss))
    visiable_images(target_model, target_model_type, rec_image[:16], 'DefenseGAN Reconstructed Adversarial Examples',save=visiable)


  elif gan_type == 2:

    adv_x, adv_all = fgsm_adversarial_advanced(sub_model, image, epsilon=epsilon,rand=rand_fgsm,alpha=0.02,rate=adv_rate)
    #adv_x = tf.clip_by_value(adv_x, -1.0, 1.0)
    visiable_images(target_model, target_model_type, adv_x[:16], 'Adversarial Examples for Target Model',save=visiable)

    train_dataset = tf.data.Dataset.from_tensor_slices(adv_all).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    gan_gen = gan.get_tmp_gen(train_dataset,TMP_GAN_EPOCHS)

    ori_accuracy = evaluation.eval(target_model, label, image)
    print('Accuracy for {} images Original is: {:0.3f}'.format(test_nums,ori_accuracy))

    adv_accuracy = evaluation.eval(target_model, label, adv_all)
    print('Accuracy for {} images Mixed is: {:0.3f}'.format(test_nums,adv_accuracy))

    rec_image = do_rec_images(target_model, gan_gen, adv_all, rr=rr, rec_L=rec_L,rec_batch_size=256)
    rec_accuracy, rec_loss = evaluation.eval_gan(target_model,label, rec_image, image)
    print('Accuracy for {} image Mixed Tmp Rec is: {:0.3f}, Rec Loss in average is: {:0.3f}'.format(test_nums,rec_accuracy, rec_loss))
    visiable_images(target_model, target_model_type, rec_image[:16], 'Tmp-GAN Reconstructed Adv Examples',save=visiable)


  elif gan_type == 3:

    adv_x, adv_all = fgsm_adversarial_advanced(sub_model, image, epsilon=epsilon,rand=rand_fgsm,alpha=0.02,rate=adv_rate)
    #adv_x = tf.clip_by_value(adv_x, -1.0, 1.0)
    visiable_images(target_model, target_model_type, adv_x[:16], 'Adversarial Examples for Target Model',save=visiable)


    ori_accuracy = evaluation.eval(target_model, label, image)
    print('Accuracy for {} images Original is: {:0.3f}'.format(test_nums,ori_accuracy))

    adv_accuracy = evaluation.eval(target_model, label, adv_all)
    print('Accuracy for {} images Mixed is: {:0.3f}'.format(test_nums,adv_accuracy))

    gen_GAN = gan.get_trained_gen(data_set_type)
    rec_image = do_rec_images(target_model, gen_GAN, adv_all, rr=rr, rec_L=rec_L,rec_batch_size=256)
    rec_accuracy, rec_loss = evaluation.eval_gan(target_model,label, rec_image, image)
    print('Accuracy for {} image Mixed DefenseGAN Rec is: {:0.3f}, Rec Loss in average is: {:0.3f}'.format(test_nums,rec_accuracy, rec_loss))
    visiable_images(target_model, target_model_type, rec_image[:16], 'DefenceGAN Reconstructed Adv Examples',save=visiable)


    train_dataset = tf.data.Dataset.from_tensor_slices(adv_all).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    gan_gen = gan.get_tmp_gen(train_dataset,TMP_GAN_EPOCHS)
    rec_image = do_rec_images(target_model, gan_gen, adv_all, rr=rr, rec_L=rec_L,rec_batch_size=256)
    rec_accuracy, rec_loss = evaluation.eval_gan(target_model,label, rec_image, image)
    print('Accuracy for {} image Mixed Tmp Rec is: {:0.3f}, Rec Loss in average is: {:0.3f}'.format(test_nums,rec_accuracy, rec_loss))
    visiable_images(target_model, target_model_type, rec_image[:16], 'Tmp-GAN Reconstructed Adv Examples',save=visiable)

  else:
    print('Wrong test condition.')




  '''
  ori_accuracy = evaluation.eval(target_model, label, image)
  print('The accuracy for {} image original is: {:0.3f}'.format(test_nums,ori_accuracy))

  adv_accuracy = evaluation.eval(target_model, label, ori_adv)
  print('The accuracy for {} image ori_adv to tar is: {:0.3f}'.format(test_nums,adv_accuracy))

  adv_accuracy = evaluation.eval(target_model, label, adv_x)
  print('The accuracy for {} image adv to tar is: {:0.3f}'.format(test_nums,adv_accuracy))

  adv_accuracy = evaluation.eval(sub_model, label, adv_x)
  print('The accuracy for {} image adv to sub is: {:0.3f}'.format(test_nums,adv_accuracy))

  gen_GAN = gan.get_trained_gen(data_set_type)
  rec_image = do_rec_images(target_model, gen_GAN, adv_x, rr=rr, rec_L=rec_L,rec_batch_size=256)
  rec_accuracy, rec_loss = evaluation.eval_gan(target_model,label, rec_image,image)
  print('The accuracy for {} image Rec is: {:0.3f}, Rec Loss in average is: {:0.3f}'.format(test_nums,rec_accuracy, rec_loss))
  '''


if __name__ == '__main__':
  print('Start black box attack test.')
  '''
  The first parameters for white_box() funciton is to decide which kind of GAN will be used
  1: for DefenseGAN
  2: for Tmp-GAN
  3: for test both DefenseGAN and Tmp-GAN
  '''
  black_box(1,'FMNIST','D', 'E',hold_out=150,test_nums=400,adv_rate=1.0,
              epsilon=0.3,rr=10,rec_L=300,visiable=False,rand_fgsm=False)