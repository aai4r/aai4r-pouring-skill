import yaml
import os


def load_cfg(cfg_file_name, des_path=None):
    d = des_path if des_path else os.getcwd()
    with open(os.path.join(*d, 'configs', cfg_file_name), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg
