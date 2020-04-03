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

def save_metric_csv(my_round, micro_acc, stack_list):
    
    args = parse_args()
    
    tot_center = [0., 0., 0., 0.]
    for center in stack_list:
        # This one is a macro acc, we cant compute micro acc here
        # as there is no weights for num samples.
        device_accuracy = np.average([center[k]["accuracy"]  for k in center.keys()])
        device_loss = np.average([center[k]["loss"]  for k in center.keys()])
        device_microf1 = np.average([center[k]["microf1"]  for k in center.keys()])
        device_macrof1 = np.average([center[k]["macrof1"]  for k in center.keys()])
        tot_center[0] += device_accuracy
        tot_center[1] += device_loss
        tot_center[2] += device_microf1
        tot_center[3] += device_macrof1
        
    if len(stack_list) > 1:
        avg_center = [v / len(stack_list) for v in tot_center]
    else:
        avg_center = tot_center

    with open(args.metric_file, 'a+') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow([my_round, micro_acc,
                     avg_center[0], avg_center[1], avg_center[2], avg_center[3]])     