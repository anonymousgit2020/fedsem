import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.optimizer.pggd import PerGodGradientDescent


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated Dane to Train')
        self.inner_opt = PerGodGradientDescent(params['learning_rate'], params['mu'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated DANE'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            # test model
            if i % self.eval_every == 0:
                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()
                tqdm.write('At round {} testing accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

            # uniform sampling
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)

            cgrads = [] # buffer for receiving client gradients
            for c in tqdm(selected_clients, desc='Grads: ', leave=False, ncols=120):
                c.set_params(self.latest_model)  # w^{t-1}
                grad, stats = c.solve_grad() # \nabla F_k(w^{t-1})
                cgrads.append(grad)
            avg_gradient = self.aggregate(cgrads)  # g_t = avg(\nabla F_k(w^{t-1}))
            # weighted aggregation to get the average gradients

            # choose another set of devices for updating
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)

            csolns = [] # buffer for receiving client solutions
            for c in tqdm(selected_clients, desc='Solve: ', leave=False, ncols=120):
                # communicate the latest model
                c.set_params(self.latest_model)  # w^{t-1}

                # setup local optimizer
                self.inner_opt.set_params(self.latest_model, avg_gradient, c)  # set g_t

                # solve minimization locally
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)
        
            # update model
            self.latest_model = self.aggregate(csolns)  # w^t

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
