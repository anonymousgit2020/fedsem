import numpy as np
import os
import tensorflow as tf
import csv

from tensorflow.python import pywrap_tensorflow
from utils.args import parse_args

def input_fn(data):
    return tf.train.limit_epochs(
          tf.convert_to_tensor(data, dtype=tf.float32), num_epochs=1)   

def restore_ckpt(variable, id, ckpt_path):
    ckpt_file = os.path.join( ckpt_path, "write_%s.ckpt" % id )
    reader = pywrap_tensorflow.NewCheckpointReader( ckpt_file )
    var =  reader.get_tensor( variable ) 
    line = np.array( var ).flatten()
    
    return line   

""" This is the function to generate a point dataset for 
    Tensorflow native clustering 
    
    Note: just discover Kmeans can handle multi-thousands dimensions data,
    so not using a auto encoder any longer and this should boost
    the speed for kmeans per iteration

"""
def get_points_dataset(joined_clients, clients, points, variable, dimensions, ckpt_path):
    # Points is a dict keeps client_id and client_weights pair
    ids = [client.id for client in joined_clients]
    for x, client_id in enumerate(points):
        if client_id in ids:
            points[client_id] = restore_ckpt(variable, client_id, ckpt_path)

    return points

def count_num_point_from(learned_cluster):
    val = ""
    for counter, grp in enumerate(learned_cluster):
        val = val + "cluster_%d=%d, " % (counter, grp[0])
    if len(val) > 1 :
        val = val[:-2]
    return val

def save_metric_csv(my_round, eval_to_use, stack_list, cluster):
    
    f_metric = 'metrics_'
    args = parse_args()
    if args.metric_file != '':
        f_metric = args.metric_file
    else:
        f_metric += "{}-{}-K{}.tsv".format(args.dataset, args.model, args.num_clusters)
    
    tot_micro_acc = 0.
    tot_micro_loss = 0.
    tot_micro_f1 = 0.
    tot_macro_f1 = 0.
    w_s = [c[0] for c in cluster if c[0] > 1]
    tot_ws = [c[0] for c in cluster if c[0] > 1]
    for center, w in zip(stack_list, w_s):
        device_micro_accuracy = np.average([center[k]["accuracy"]  for k in center.keys()]) 
        device_micro_loss = np.average([center[k]["loss"]  for k in center.keys()]) 
        device_microf1 = np.average([center[k]["f1"]  for k in center.keys()]) 
        device_macrof1 = np.average([center[k]["f1"]  for k in center.keys()]) / len(w_s)
        tot_micro_acc += (device_micro_accuracy * w) / np.sum(tot_ws)
        tot_micro_loss += (device_micro_loss * w) / np.sum(tot_ws)
        tot_micro_f1 += (device_microf1 * w) / np.sum(tot_ws)
        tot_macro_f1 += device_macrof1
        

    with open(f_metric, 'a+') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow([my_round, eval_to_use,
                     tot_micro_acc, tot_micro_loss, tot_micro_f1, tot_macro_f1])     