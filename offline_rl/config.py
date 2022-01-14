import yaml
import os


def load_cfg(cfg_file_name):
    with open(os.path.join(os.getcwd(), 'offline_rl', 'configs', cfg_file_name), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg
