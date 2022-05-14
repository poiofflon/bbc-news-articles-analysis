import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config = dict(config['DEFAULT'])


def __getattr__(name):
    if name in config or name == 'shape':
        return config[name]
    else:
        print(f'{name} is not a config setting: {list(config.keys())}')
