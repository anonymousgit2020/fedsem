import tensorflow as tf
import numpy as np

from scipy import stats

class TrimKmeanModel():
    
    def __init__(self, num_samples, dimensions, clusters, estimate="trmean"):
        self.num_iter = 1
        self.first = True
        self.graph = tf.Graph()
        self.y, self.X = self.create_model(num_samples, dimensions, clusters)
        self.sess = tf.Session(graph=self.graph)
        with self.sess.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        self.n_clus = clusters
        self.estimate=estimate
    
    def create_model(self, num_samples, dim, k):
        with self.graph.as_default():
            X = tf.placeholder(tf.float32, [num_samples, dim])
            cluster_membership = tf.Variable(tf.zeros([num_samples]), dtype=tf.float32, name='cluster_membership')
            centroids = tf.Variable(tf.random_uniform([k,dim]), dtype=tf.float32, name='centroids')
            
            X_temp = tf.reshape(tf.tile(X, [1,k]), [num_samples, k, dim])
            centroids_temp = tf.reshape(tf.tile(centroids,[num_samples,1]), [num_samples, k, dim])

            distances_to_centroids = tf.reduce_sum(tf.square(tf.subtract(X_temp, centroids_temp)), reduction_indices=2)  #N x k x 1
            cluster_membership = tf.arg_min(distances_to_centroids, 1) 

            return cluster_membership, X

    def assign_clusters(self, data):
        iteration = self.num_iter
        with self.sess.graph.as_default():
            var_list = tf.trainable_variables()
            if self.first:
                init_points = data[np.random.randint(data.shape[0], size=self.n_clus), :]
                centroids =  var_list[len(var_list) -1]
                centroids.load(init_points, self.sess)
                # seems should set iteration correspond to dataset
                iteration = 8
                self.first = False
            for i in range(iteration):
                y = self.sess.run([self.y], feed_dict={self.X:data})
                # I can calculate my own trimmed mean after this
                # and then assign them to centroids now                
                centroids =  var_list[len(var_list) -1]
                new_means = self.get_mean(data, y[0], self.n_clus)
                centroids.load(new_means, self.sess)

            return y[0]

    def get_mean(self, X, y, num_clusters):
        output = np.zeros((num_clusters, X.shape[1]))
        for k in range(num_clusters):
            sliced_x = X[np.isin(y, k)]
            if self.estimate == "gmean":
                output[k,:] = np.median(sliced_x, axis=0)
            else:
                output[k,:] = stats.trim_mean(sliced_x, 0.05)

        return output
    
