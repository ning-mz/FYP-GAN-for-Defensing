import tensorflow as tf
import numpy as np



def prepare_data():
  #load data of fashion mnist data
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
  train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
  train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

  test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
  test_images = (test_images - 127.5) / 127.5 # Normalize the images to [-1, 1]


  return train_images, train_labels, test_images, test_labels


# def main():
#     train_images, train_labels, test_images, test_labels = prepare_data()


# num_points = 10
# dimensions = 3
# points = np.random.uniform(0, 100, [num_points, dimensions])

train_images, train_labels, test_images, test_labels = prepare_data()
test_images = test_images[:150]
test_labels = test_labels[:150]

# test = np.array([
#                 [[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]],

#                 [[[2],[2],[2],[2]],[[2],[2],[2],[2]],[[2],[2],[2],[2]],[[2],[2],[2],[2]]],

#                 [[[3],[3],[3],[3]],[[3],[3],[3],[3]],[[3],[3],[3],[3]],[[3],[3],[3],[3]]]

#                 ])
# print(test)
# a = tf.reshape(tf.convert_to_tensor(test, dtype=tf.float32), [3,16])
# a = tf.convert_to_tensor(test, dtype=tf.float32)
# print(a)
# a = tf.reshape(a, [3,16])
test_images = tf.reshape(test_images, [150, 784])


def input_fn():
    #return tf.data.Dataset.from_tensors(tf.convert_to_tensor(points, dtype=tf.float32)).repeat(1)
    return tf.data.Dataset.from_tensors(test_images.numpy()).repeat(1)


num_clusters =20
kmeans = tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, distance_metric='cosine', use_mini_batch=True)

# train
num_iterations = 50
previous_centers = None
for _ in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    #if previous_centers is not None:
        #print('delta:', cluster_centers - previous_centers)
    previous_centers = cluster_centers
    #print('score:', kmeans.score(input_fn))
#print('cluster centers:', cluster_centers)

# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(test_images):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    if cluster_index == 0:
        print('point:', i, 'is in cluster', cluster_index,)
        print('label is',test_labels[i])

