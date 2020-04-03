import copy
import os
import sys
import argparse
import importlib
import math
import numpy as np
import tensorflow as tf


from baseline_constants import ACCURACY_KEY

from mlhead_client import Client
from server import Server

from utils.args import parse_args
from utils.model_utils import read_data

from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering

soln_and_clients_dict = list()
total_num_rounds = 30



MODEL_PARAMS = {
    'sent140.bag_dnn': (2, "dense/kernel"), # num_classes
    'femnist.cnn': (0.003, 62, "dense_1/kernel") ,
    'celeba.cnn': (0.01, 2, "dense/kernel")
}


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.
    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = [ACCURACY_KEY, 'loss']
    to_ret = None
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights)))

def print_stats(
    num_round, server, clients, num_samples, args):

    eval_set = 'test' 
    test_stat_metrics = server.test_model(clients, True, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))

def create_clients(model, args):        
    train_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'test')  
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)    
    if len(groups) == 0:
        groups = [[] for _ in users]    
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients

def client_model_factory_fn(seed, args):
    
    def factory_fn(init_model_params = None):
        model_path = '%s.%s' % (args.dataset, args.model)
        model_params = MODEL_PARAMS[model_path]
        if args.lr != -1:
            model_params_list = list(model_params)
            model_params_list[0] = args.lr
            model_params = tuple(model_params_list)        
        optimizer = None
        offset = len(model_params) -1
        cm = ClientModel(seed, *model_params[:offset], optimizer)
        if not (init_model_params == None):
            cm.set_params(init_model_params)        
        return cm
    
    model_path = '%s.%s' % (args.dataset, args.model)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    return factory_fn

def get_similarity(c1, c2, X_c1, X_c2):
    a = X_c1
    b = X_c2
    z = 1 - distance.cdist(a, b, 'cosine')
    sim_m = np.array(z).max()
    z = np.sqrt((1 - sim_m) /2)    
    #print("Matrix of sim\n", z)
    return z

def fed(init_soln, clients_from, client_model_factory, args):
    '''
    Assume that we need to create a server compoent
    this component will init from a init_soln.
    
    This bits of function returns a global model and 
    along with all applied clients 
    that are used to generate this model
    
    what condition to stop here? 
    no cond, just run 50 epoch to stop
    '''
    client_model = client_model_factory(init_soln)
    server = Server(client_model)
    for k in range(total_num_rounds):
        print('--- Round %d of %d ' % (k + 1, total_num_rounds),  "> ---")
        server.select_clients(k, clients_from, num_clients=args.clients_per_round)
        server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)
        
        server.update_model_wmode()
        print_stats(k+1, server, server.selected_clients, c_num_samples, args)
        
    
    # Train one more to get all clients train and obtain the 
    # final \nlab R(theta)start for each client
    
    server.select_clients(total_num_rounds, clients_from, num_clients=len(clients_from))
    server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
    updated_w = copy.deepcopy(server.updates) 
    g_o_theta = copy.deepcopy(server.model)
    print('We just train last round for ease the similarity calculation')
    server.close_model()
    return g_o_theta, updated_w

def default_bi_cluster(g_soln, clients):
    t = math.ceil(len(clients)/2)
    c1 = list()
    c2 = list()
    for i in range(int(t)):
        c1.append(clients[i])
    for i in range(int(t), len(clients)):
        c2.append(clients[i])
    return c1, c2
    
def bi_cluster(g_soln, clients, updated_w):
    '''
    This takes a collection of clients then fit them,
    features input will be the delta of weight update
    of each client
    should divde them base on cosine sim
    
    Note, using sklearn's AgglomerativeClustering model to fit
    '''
    args = parse_args()
    model_path = '%s.%s' % (args.dataset, args.model)
    model_params = MODEL_PARAMS[model_path]
    pos = len(model_params) -1
    op_name = model_params[pos]
    def get_x(model_params, g_soln):
        for var, m_p, g_p in zip(all_vars, model_params, g_soln):
            if var.op.name == op_name:
                #print("shape of layer:", m_p.shape)
                z = m_p - g_p
                return np.array(z).flatten()
     
    
    # need to see if g_soln is correct
    c = clients[0]
    with c.model.graph.as_default():
        all_vars = tf.trainable_variables()
        '''
        for var, g_value, c_value in zip(allv, g_soln, z):
            if var.op.name == op_name:
                y = np.array(g_value).flatten()
                y_s = np.sort(y)
                z = np.array(c_value).flatten()
                z_s = np.sort(z)
                print("Global\n",y_s[:20])
                print("Client\n",z_s[:20])
         '''   
    X = [get_x(updated_w[idx][1], g_soln) for idx, client in enumerate(clients)]
    clustering = AgglomerativeClustering(
        n_clusters=2
        ,affinity='cosine'
        ,linkage='average').fit(X)
    c1 = list()
    c2 = list()
    X_c1 = list()
    X_c2 = list()
    labels = clustering.labels_
    for counter, c in enumerate(clients):
        if labels[counter] == 1:
            c1.append(c)
            X_c1.append(X[counter])
        else:
            c2.append(c)
            X_c2.append(X[counter])
    return c1, c2, X_c1, X_c2

def cluster_fed(init_soln, clients, args):
    '''
    This is the main recurrsive function that takes 1..m clients 
    and then returns the global solution
    '''
    gamma = args.gamma
    client_model_factory = client_model_factory_fn(args.seed, args)
    client_model = client_model_factory()
    if clients == None:    
        clients = create_clients(client_model, args)
        
    g_soln, updated_w = fed(init_soln, clients, client_model_factory, args)
    c1, c2, X_c1, X_c2 = bi_cluster(g_soln, clients, updated_w)
    print("Cluster1 len: ", len(c1))
    print("Cluster2 len: ", len(c2))
    cross_clus_dis = get_similarity(c1, c2, X_c1, X_c2)

    print("max cross sim: %s" % cross_clus_dis)
    if cross_clus_dis>= gamma:
        solved = False
    else:
        solved = True# cond
    if not solved:
        if len(c1) > args.min_clients_per_clus :
            print("c1 splitting..")
            cluster_fed(g_soln, c1, args)
        if len(c2) > args.min_clients_per_clus:
            print("c2 splitting..")
            cluster_fed(g_soln, c2, args)
    else:
        print("condtion stop")
        current_soln_and_clients = {
            'clients1': c1,
            'clients2': c2,
            'solution': g_soln
        }
        soln_and_clients_dict.append(current_soln_and_clients)
    return soln_and_clients_dict

def get_init_param(args):
    model_params = None
    return model_params

def main():

    # hyper max-cross simi:

    '''
    Three parts to run this algorithm:
    1. call Fed to solve the problem 
    2. clustering into two c1, c2
    3. if already solved terminate else recurrsively call itself cluster_fed with c1 and c2
    
    '''    
    args = parse_args()
    tf.reset_default_graph()

        
    init_soln = get_init_param(args)
    clients = None
    cluster_fed(init_soln, clients, args)
    
    print("Test on result")
    

if __name__ == '__main__':
    main()