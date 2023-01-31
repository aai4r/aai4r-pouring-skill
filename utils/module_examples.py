import numpy as np
import torchvision.transforms
from torchvision import models, transforms
import torch
import torch.nn as nn
import os
import time
import cv2
import imp

from spirl.utility.general_utils import AttrDict, ParamDict
from spirl.components.checkpointer import get_config_path
from spirl.components.data_loader import GlobalSplitVideoDataset
from spirl.modules.subnetworks import BaseProcessingLSTM, Predictor, Encoder
from spirl.utility.pytorch_utils import map2np, ten2ar, RemoveSpatial, ResizeSpatial, map2torch, find_tensor, \
                                        TensorModule, RAdam


class StateCondImageSkillPriorNet(nn.Module):
    def __init__(self, hp, enc_params):
        super().__init__()
        self._hp = hp
        self._enc_params = enc_params

        self.resize = ResizeSpatial(self._hp.prior_input_res)
        self.enc = Encoder(self._enc_params)
        self.rm_spatial = RemoveSpatial()

        input_size = self._hp.nz_mid_prior + self._hp.state_cond_size  # * self._hp.n_input_frames
        self.fc = Predictor(self._hp, input_size=input_size,
                            output_size=self._hp.nz_vae * 2, num_layers=self._hp.num_prior_net_layers,
                            mid_size=self._hp.nz_mid_prior)

    def forward(self, inputs):
        """
        * state-conditioned Skill Prior Net
        * inputs should contain: {images, states}
        """
        out = self.resize(inputs.images)
        out = self.enc(out)
        out = self.rm_spatial(out)
        z = self.fc(torch.cat((out, inputs.states), dim=-1))
        return z


def get_config(path):
    conf = AttrDict()

    # paths
    conf.exp_dir = os.environ['EXP_DIR']
    conf.conf_path = get_config_path(path)

    # general and model configs
    print('loading from the config file {}'.format(conf.conf_path))
    conf_module = imp.load_source('conf', conf.conf_path)
    conf.general = conf_module.configuration
    conf.model = conf_module.model_config

    # data config
    try:
        data_conf = conf_module.data_config
    except AttributeError:
        data_conf_file = imp.load_source('dataset_spec', os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
        data_conf = AttrDict()
        data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
        data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
    conf.data = data_conf

    # model loading config
    conf.ckpt_path = conf.model.checkpt_path if 'checkpt_path' in conf.model else None

    return conf


def load_expert_demo_data():
    data_dir = "../data/pouring_water_img_test"
    path = '../spirl/configs/skill_prior_learning/pouring_water_img/hierarchical_cl'

    _hp = default_dict = ParamDict({
            'model': None,
            'model_test': None,
            'logger': None,
            'logger_test': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'batch_size': 128,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 300,
            'epoch_cycles_train': 1,
            'optimizer': 'adam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'lr': 1.e-4,
            'gradient_clip': None,
            'init_grad_clip': 0.001,
            'init_grad_clip_step': 100,     # clip gradients in initial N steps to avoid NaNs
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,       # beta1 param in Adam
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
            'logging_target': 'wandb',
        })

    conf = get_config(path=path)
    conf.model['batch_size'] = _hp.batch_size if not torch.cuda.is_available() \
        else int(_hp.batch_size / torch.cuda.device_count())
    conf.model.update(conf.data.dataset_spec)
    conf.model['device'] = conf.data['device'] = "cuda"

    _hp.overwrite(conf.general)  # override defaults with config file
    _hp.data_dir = data_dir

    loader = GlobalSplitVideoDataset(data_dir=_hp.data_dir, data_conf=conf.data, resolution=256, phase="train", dataset_size=-1)\
        .get_data_loader(_hp.batch_size, 1)

    model = load_weights_and_freeze()
    print("dataset load...")
    print("loader", loader)
    for batch_idx, sample_batched in enumerate(loader):
        print("batch idx: ", batch_idx)
        n_frames = 2
        inputs = sample_batched["images"]   # (batch, rollout, channel, height, width)
        batch_size = inputs.shape[0]
        h, w, c = inputs.shape[-2], inputs.shape[-1], 3     # height, width, channel
        inputs_seq = inputs[:, :n_frames].reshape(batch_size, -1, h, w)  # (128, 6, 256, 256)

        for idx in range(batch_size):
            unroll = torch.tensor([])
            for i in range(n_frames):
                start, end = i * c, (i + 1) * c
                unroll = torch.cat((unroll, inputs_seq[:, start:end]), dim=-1)

            tr = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
            start = time.time()
            unroll = tr((unroll + 1.0) / 2.0)
            print("elapsed tr: ", time.time() - start)
            print("min/max: {} / {}".format(unroll.min(), unroll.max()))

            img = unroll[idx, :, :, :w].cpu().numpy().transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for i in range(1, n_frames):
                start, end = i * w, (i + 1) * w
                temp = unroll[idx, :, :, start:end].cpu().numpy().transpose(1, 2, 0)
                temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
                img = np.concatenate((img, temp), axis=-2)

            cv2.imshow("images", img)
            print("Frame: {} / {}".format(idx, batch_size))
            k = cv2.waitKey()
            if k == 27:
                break

    print("end...")


def load_weights_and_freeze():
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1],     # exclude the last fc layer
                          nn.Flatten(),)
    for param in model.parameters():
        param.requires_grad = False
    return model


def resnet_test():
    model = models.resnet18(pretrained=True)
    res = nn.Sequential(*list(model.children())[:-1])
    inputs = torch.rand(64, 3, 128, 128)
    outs = res(inputs)
    # print(model)
    for c in model.children():
        print(c)
    print("out: ", outs.shape)


if __name__ == "__main__":
    os.environ["EXP_DIR"] = "../experiments"
    os.environ["DATA_DIR"] = "../data"

    # pre_trained()
    # load_expert_demo_data()
    resnet_test()

