"""Script to run the fedmc, check the github page 
    and our paper of Multihead Federated Learning for
    specific usage.    
."""

import os
import sys
import argparse

from utils.args import parse_args
from utils.model_utils import read_data

from mlhead_trainer import MlheadTrainer

def main():
    args = parse_args()
    
    train_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'test')   
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    
    trainer = MlheadTrainer(args, users, groups, train_data, test_data)
    trainer.train(args)
    trainer.finish(args)


if __name__ == '__main__':
    main()
