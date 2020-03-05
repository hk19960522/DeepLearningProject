import yaml


def gen():
    file = open('config.yaml', 'w')

    data = {'data_set': 'zara1',
            'lr': 0.005,
            'batch_size': 128}

    yaml.dump(data, file)
    file.close()


def read():
    file = open('config.yaml', 'r')
    data = yaml.load(file, Loader=yaml.SafeLoader)
    print(data)


gen()
