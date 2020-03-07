import argparse
import logging
import os
from scripts import YAML

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
    print(parser.parse_args().train)
    return parser.parse_args()


args = set_argparse()
set_dir(args.id)
config = YAML.get_yaml(args, args.id)
logging.info(config)
