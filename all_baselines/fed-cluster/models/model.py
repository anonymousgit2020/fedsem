"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import tensorflow as tf

from baseline_constants import ACCURACY_KEY

from utils.model_utils import batch_data
from utils.tf_utils import graph_size

from sklearn.metrics import precision_score, recall_score, f1_score

class Model(ABC):

    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        self._optimizer = optimizer

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(seed)
            self.features, self.labels, self.train_op, self.eval_metric_ops, self.loss, self.pred_ops = self.create_model()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            #metadata = tf.RunMetadata()
            #opts = tf.profiler.ProfileOptionBuilder.float_operation()
            #self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def set_params(self, model_params):
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, model_params):
                variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params
    

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 6-tuple return value for the task
        """
        return None, None, None, None, None, None

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        for _ in range(num_epochs):
            self.run_epoch(data, batch_size)

        update = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * 1
        return comp, update

    def run_epoch(self, data, batch_size):
        for batched_x, batched_y in batch_data(data, batch_size):
            
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            
            with self.graph.as_default():
                self.sess.run(self.train_op,
                    feed_dict={
                        self.features: input_data,
                        self.labels: target_data
                    })

    def prepare_test(self, data, batch_size, min_loss):
        # try to reach convergence before testing
        # return loss for diag
        loss_list = list()
        first = True
        for batched_x, batched_y in batch_data(data, batch_size):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            with self.graph.as_default():
                _, loss = self.sess.run([self.train_op, self.loss],
                    feed_dict={
                        self.features: input_data,
                        self.labels: target_data
                    })  
                
            loss_list.append(loss)
            # A conditon if new_loss is almost stable
        return loss_list

            
    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """        
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            tot_acc, loss, pred_ops = self.sess.run(
                [self.eval_metric_ops, self.loss, self.pred_ops],
                feed_dict={self.features: x_vecs, self.labels: labels}
            )
        acc = float(tot_acc) / x_vecs.shape[0]

        if  len(np.array(labels).shape) ==2:
            y_true = np.argmax(np.array(labels), axis=1)
        else:
            y_true = np.array(labels)
            
        mircof1 = f1_score(y_true, pred_ops, average='micro') 
        macrof1 = f1_score(y_true, pred_ops, average='macro') 

        return {ACCURACY_KEY: acc, 'loss': loss, "microf1":mircof1, "macrof1":macrof1}

    def close(self):
        self.sess.close()
        
    def save_file(self, path):
        with self.graph.as_default():
            tmp = tf.train.Saver()
            tmp.save(self.sess, path)

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        with self.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                var_vals[v.name] = val
        for c in clients:
            with c.model.graph.as_default():
                all_vars = tf.trainable_variables()
                for v in all_vars:
                    v.load(var_vals[v.name], c.model.sess)

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()
