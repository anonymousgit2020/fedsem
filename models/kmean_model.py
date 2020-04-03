import numpy as np
import os
import sys
import tensorflow as tf



class KmeanModel():
    
    def __init__(self, num_samples, dimensions, clusters, seed):
        self.num_iter = 1
        self.first = True
        self.graph = tf.Graph()
        self.train_ops, self.y, self.X, self.init_ops = self.create_model(num_samples, dimensions, clusters)
        self.sess = tf.Session(graph=self.graph)
    
    def create_model(self, num_samples, dim, k):
        with self.graph.as_default():
            X = tf.placeholder(tf.float32, [num_samples, dim])
            cluster_membership = tf.Variable(tf.zeros([num_samples]), dtype=tf.float32)
            centroids = tf.Variable(tf.random_uniform([k,dim]), dtype=tf.float32)
            
            # This ops is for the first to select K random center from X
            centroids_init = centroids.assign(tf.slice(X, [0,0], [k, dim]))
            X_temp = tf.reshape(tf.tile(X, [1,k]), [num_samples, k, dim])
            centroids_temp = tf.reshape(tf.tile(centroids,[num_samples,1]), [num_samples, k, dim])

            distances_to_centroids = tf.reduce_sum(tf.square(tf.subtract(X_temp, centroids_temp)), reduction_indices=2)  #N x k x 1
            cluster_membership = tf.arg_min(distances_to_centroids, 1) 

            new_means_numerator = tf.unsorted_segment_sum(X, cluster_membership, k)
            new_means_denominator = tf.unsorted_segment_sum(tf.ones_like(X), cluster_membership, k)
            new_means = new_means_numerator/new_means_denominator
            update_centroids = centroids.assign(new_means)
            return update_centroids, cluster_membership, X, centroids_init

    def assign_clusters(self, data):
        iteration = self.num_iter
        with self.sess.graph.as_default():
            if self.first:
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(self.init_ops, feed_dict={self.X:data})
                # seems should set iteration correspond to dataset
                iteration = 5
                self.first = False
            for i in range(iteration):
                centroids, y = self.sess.run([self.train_ops, self.y], feed_dict={self.X:data})  
            return y
    
