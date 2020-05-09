import math
import tensorflow as tf

BATCH_SIZE = 128

def eval_gan(cnn, original_label, rec_image, ori_image):
    print('Evaluating  accuracy...')

    batch_nums = int(math.ceil(float(len(original_label)/BATCH_SIZE)))

    assert batch_nums*BATCH_SIZE >=len(original_label)
    
    accuracy = 0.0
    rec_loss = 0.0

    for num in range(batch_nums):
        ind_start = num * BATCH_SIZE
        ind_end = min(len(original_label), (num+1) * BATCH_SIZE)

        ori_image_batch = ori_image[ind_start:ind_end]
        rec_image_batch = rec_image[ind_start:ind_end]
        label_batch = original_label[ind_start:ind_end]

        prediction =  cnn.predict_classes(rec_image_batch)
        correct_prediction = tf.equal(label_batch, prediction)

        num_of_correct = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32))
        accuracy = accuracy + num_of_correct.numpy()

        #print('batch: {}'.format(num))
        #print(accuracy)

        rec_loss_batch = tf.reduce_sum(tf.reduce_mean(
                          tf.square(rec_image_batch - ori_image_batch), 
                          axis=range(1, len(rec_image.get_shape()))))
        rec_loss = rec_loss + rec_loss_batch
    
    accuracy /= len(original_label)
    rec_loss /= len(original_label)

    return accuracy, rec_loss

def eval(cnn, original_label, image):
    print('Evaluating  accuracy...')

    batch_nums = int(math.ceil(float(len(original_label)/BATCH_SIZE)))
    #print('batch num: {}'.format(batch_nums))
    assert batch_nums*BATCH_SIZE >=len(original_label)
    
    accuracy = 0.0

    for num in range(batch_nums):
        ind_start = num * BATCH_SIZE
        ind_end = min(len(original_label), (num+1) * BATCH_SIZE)

        image_batch = image[ind_start:ind_end]
        label_batch = original_label[ind_start:ind_end]

        prediction =  cnn.predict_classes(image_batch)
        correct_prediction = tf.equal(label_batch, prediction)

        num_of_correct = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32))
        accuracy = accuracy + num_of_correct.numpy()

    accuracy /= len(original_label)

    return accuracy