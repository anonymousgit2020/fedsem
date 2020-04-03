import os
import csv
import numpy as np
import tensorflow as tf


ROUND = 0
def save_metric_csv(my_round, acc, loss, f1, ws):
    
    global ROUND
    
    f_metric = 'feddane_met.tsv'

    tot_micro_acc = 0.
    tot_micro_loss = 0.
    tot_macro_loss = 0.
    tot_micro_f1 = 0.
    tot_macro_f1 = 0.
    
    tot_ws = np.sum(ws)
    
    tot_micro_acc = np.average(acc, weights=ws/ tot_ws)
    tot_micro_loss = np.average(loss, weights=ws / tot_ws)
    tot_macro_loss = np.average(loss, weights=None)
    tot_micro_f1 = np.average(f1, weights=ws / tot_ws)
    tot_macro_f1 = np.average(f1, weights=None)
        

    with open(f_metric, 'a+') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow([ROUND, tot_micro_acc,
                     tot_micro_loss, tot_macro_loss, tot_micro_f1, tot_macro_f1])     
    ROUND +=1
    
    
def get_f1(y_true, y_pred):
    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return macro