import argparse
import logging


# Set the logging config to print info
logging.basicConfig(
    format='%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)


# Set argparse
def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int)
    print(parser.parse_args().lr)


set_argparse()