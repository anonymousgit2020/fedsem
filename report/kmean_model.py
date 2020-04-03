import numpy as np
import os
import sys
import tensorflow as tf

clusters = 5
dimensions = 15000
num_samples=200
num_iter = 1
input_fn = np.random.normal(loc=0.65, scale=0.7, size=[num_samples,dimensions])

class KmeanModel():
    
    def __init__(self):
        self.graph = tf.Graph()
        self.train_ops, self.cmembership, self.X = self.create_model(num_samples, dimensions, clusters)
    
    def create_model(self, num_samples, dim, k):
        with self.graph.as_default():
            X = tf.placeholder(tf.float32, [num_samples, dim])
            cluster_membership = tf.Variable(tf.zeros([num_samples]), dtype=tf.float32)
            centroids = tf.Variable(tf.random_uniform([k,dim]), dtype=tf.float32)
            X_temp = tf.reshape(tf.tile(X, [1,k]), [num_samples, k, dim])
            centroids_temp = tf.reshape(tf.tile(centroids,[num_samples,1]), [num_samples, k, dim])

            distances_to_centroids = tf.reduce_sum(tf.square(tf.subtract(X_temp, centroids_temp)), reduction_indices=2)  #N x k x 1
            cluster_membership = tf.arg_min(distances_to_centroids, 1) 
            #distance-minimizing column for each row 

            new_means_numerator = tf.unsorted_segment_sum(X, cluster_membership, k)
            new_means_denominator = tf.unsorted_segment_sum(tf.ones_like(X), cluster_membership, k)
            new_means = new_means_numerator/new_means_denominator
            update_centroids = centroids.assign(new_means)
            return update_centroids, cluster_membership, X

    def assign_clusters(self):
        self.sess = tf.Session(graph=self.graph)
        with self.sess.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            for i in range(num_iter):
                centroids, membership = self.sess.run([self.train_ops, self.cmembership], feed_dict={self.X:input_fn})  
            self.sess.close()
            return membership
    
if __name__ == '__main__':
  kmean = KmeanModel()
  print(kmean.assign_clusters())
