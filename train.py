import argparse
import logging
import os
from scripts import YAML
from scripts import dataset
from scripts import model
import torch.optim as optim

# Set the logging config to print info\
logging.basicConfig(
    format='%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)


def set_dir(folder_name):
    if not folder_name:
        logging.error('No ID')
        exit()

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        logging.debug('create new folder')
    else:
        logging.debug('id existed')


# Set argparse
def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    # TODO: need to set --id's required to True
    parser.add_argument('--id', type=str, required=False)

    # training config
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epoch', type=int, default=10000)
    parser.add_argument('--d_step', type=int, default=100)
    print(parser.parse_args().train)
    return parser.parse_args()


def train(args, train_loader, test_loader):

    min_loss = 5.0
    for epoch in range(0, args.num_epoch):
        logging.info(f'Epoch {epoch}:')
        epoch_loss = []
        cnt = 0
        for batch in train_loader:
            [
                obs_traj, pred_traj,
                obs_rel_traj, pred_rel_traj,
                seq_start_end
            ] = batch
            logging.debug(f'{cnt} batch.')
            cnt += 1
            pass
    pass


args = set_argparse()
# set_dir(args.id)
config = YAML.get_yaml(args, args.id)
logging.info(config)

test_dataset, test_loader = dataset.get_loader('test1', 'test')
train_dataset, train_loader = dataset.get_loader('test1', 'train')

train(args, train_loader, test_loader)
print('DONE')


