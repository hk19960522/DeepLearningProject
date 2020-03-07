import os
import yaml
import logging


def gen():
    file = open('../config.yaml', 'w')

    data = {'data_set': 'zara1',
            'lr': 0.005,
            'batch_size': 128}

    yaml.dump(data, file)
    file.close()


def read(file_path='config.yaml'):
    file = open(file_path, 'r')
    data = yaml.load(file, Loader=yaml.SafeLoader)
    logging.info('read complete.')
    return data


def get_yaml(args, folder_name=''):
    if args.train:
        return read()

    file_name = folder_name + '/config.yaml'
    logging.debug(file_name)
    if not os.path.exists(file_name):
        logging.info(f'Directory "{folder_name}" have no config.yaml')
        file_name = 'config.yaml'
    logging.debug('file name : ' + file_name)
    return read(file_name)



# gen()
