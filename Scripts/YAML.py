import yaml
import logging


def gen():
    file = open('../config.yaml', 'w')

    data = {'data_set': 'zara1',
            'lr': 0.005,
            'batch_size': 128}

    yaml.dump(data, file)
    file.close()


def read():
    file = open('../config.yaml', 'r')
    data = yaml.load(file, Loader=yaml.SafeLoader)
    logging.info('read complete.')
    return data


def get_yaml(args, folder_name=''):
    if args.train:
        return read()
    if folder_name == '':
        folder_name = '..'
    else:
        folder_name = '../' + folder_name
    file_name = folder_name + '/config.yaml'


# gen()
