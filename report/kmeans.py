import tensorflow as tf
import numpy as np
"""
Initialization:
X: [n x 2], randomly drawn from normal distribution
centroids: [k x 2], also randomly drawn
Reshaping stage:
X: [n x 2] => [n x k x 2] * same as the n x 2 representation, but repeat each observation k times in the k x 2 sub-matrices
centroids: [k x 2] => [n x k x 2]  * each kx2 is the centroid coordinates. repeat these n times.
Calculate distances from observations to centroids:
* reduce_sum of squared distances between reshaped X and reshaped centroids. [n x k x 2] reduced to [n x k] distance matrix
* assignments = Arg_min(distances) => [n x 1], where each observation is the ID of the centroid with lowest distance 
Update centroids:
coords = (mean(x and y coords) where cluster == k) for each k. Uses tf.unsorted_segment_sum for this.
"""
np.random.seed(244444 + 789345)
g = tf.Graph()
sess = tf.Session(graph=g)
tf.set_random_seed(10723442)
with g.as_default():
#model parameters
  k = 4
  dimensions = 15000
  n_obs=200
  num_iter = 20 #running for a set number of iterations. set up smarter stopping-criteria if doing this for real.

#simulate data, define key variables
  X_mat = np.random.normal(loc=0.65, scale=0.7, size=[n_obs,dimensions])
  X = tf.placeholder(tf.float32, [X_mat.shape[0], X_mat.shape[1]])
  cluster_membership = tf.Variable(tf.zeros([n_obs]), dtype=tf.float32)
  centroids = tf.Variable(tf.random_uniform([k,dimensions]), dtype=tf.float32)

#reshaping data to get distances to centroids
  X_temp = tf.reshape(tf.tile(X, [1,k]), [n_obs, k, dimensions])
  centroids_temp = tf.reshape(tf.tile(centroids,[n_obs,1]), [n_obs, k, dimensions])

#calculate distances, find nearest centroid for each point and assign membership
  distances_to_centroids = tf.reduce_sum(tf.square(tf.subtract(X_temp, centroids_temp)), reduction_indices=2)  #N x k x 1
  cluster_membership = tf.arg_min(distances_to_centroids, 1) #distance-minimizing column for each row 

#update centroids by moving them to the mean of their now-updated points
  new_means_numerator = tf.unsorted_segment_sum(X, cluster_membership, k)
  new_means_denominator = tf.unsorted_segment_sum(tf.ones_like(X), cluster_membership, k)
  new_means = new_means_numerator/new_means_denominator
  update_centroids = centroids.assign(new_means)

  final_distance = tf.reduce_sum(tf.reduce_min(distances_to_centroids, 1), 0)

#run the graph
  sess.run(tf.global_variables_initializer())
  for i in range(num_iter):
    centroids, membership = sess.run([update_centroids, cluster_membership], feed_dict={X:X_mat})
    #print centroids
  print("final distance ",sess.run(final_distance, feed_dict={X:X_mat}))
print("final centroids: \n" + str(centroids))
print("final memberships: \n" + str(membership))
