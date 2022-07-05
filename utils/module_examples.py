import numpy as np
from torchvision import models
import torch
import torch.nn as nn
import os
import cv2
import imp

from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.components.checkpointer import get_config_path
from spirl.components.data_loader import GlobalSplitVideoDataset
from spirl.modules.subnetworks import BaseProcessingLSTM, Predictor, Encoder
from spirl.utils.pytorch_utils import map2np, ten2ar, RemoveSpatial, ResizeSpatial, map2torch, find_tensor, \
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
        inputs = sample_batched["images"]   # (batch, rollout, channel, height, width)
        batch_size = inputs.shape[0]
        height, width = inputs.shape[3], inputs.shape[4]
        # input = sample_batched["images"][:, 0]
        inputs_seq = inputs[:, :2].reshape(batch_size, -1, height, width)   # (128, 6, 256, 256)
        inputs_batch = inputs[:, :2].reshape(-1, 3, height, width)          # (256, 3, 256, 256)
        print("sample: ", inputs.shape)

        # input comparison
        img1 = inputs_seq[:, :3]    # tensor img
        img2 = inputs_batch[:128]

        unroll = inputs_seq.reshape(batch_size, 3, height, -1)
        print("img1, shape: {}, type: {},  min/max: {} / {}".format(img1.shape, img1.dtype, img1.min(), img1.max()))
        print("img2, shape: {}, type: {},  min/max: {} / {}".format(img2.shape, img2.dtype, img2.min(), img2.max()))
        print("img diff: ", (img1 - img2).sum())

        _img1_np = img1.numpy().transpose(0, 2, 3, 1)
        _img2_np = img2.numpy().transpose(0, 2, 3, 1)
        _unroll_np = unroll.numpy().transpose(0, 2, 3, 1)
        print("unroll shape: ", _unroll_np.shape)

        _img1 = cv2.cvtColor(_img1_np[1], cv2.COLOR_RGB2BGR)
        _img2 = cv2.cvtColor(_img2_np[1], cv2.COLOR_RGB2BGR)
        _unroll = cv2.cvtColor(_unroll_np[1], cv2.COLOR_RGB2BGR)
        print("unroll shape: ", _unroll.shape)
        print("np img diff: ", (_img1_np - _img2_np).sum())

        np_horizon_stack = np.concatenate((_img1, _img2), axis=1)
        # cv2.imshow("image", np_horizon_stack)
        cv2.imshow("unroll", _unroll)
        cv2.waitKey()

        out1 = model(img1)
        out2 = model(img2)
        print("out diff: ", (out1 - out2).sum())   # should be zero

        out_all = model(inputs_batch)
        print("comp2: ", (out1 - out_all[0]).sum())

        out_unroll = model(unroll)
        print("unroll result: ", out_unroll.shape)
        print("out min/max: {} / {}".format(out_unroll.min(), out_unroll.max()))

        # out_all_1 = out_all[:batch_size].reshape(batch_size, -1)
        # out_all_2 = out_all[batch_size:].reshape(batch_size, -1)
        # print("result 1: ", (out1 - out_all_1).sum())
        # print("result 2: ", (out2 - out_all_2).sum())
        # print("result 12: ", (out1 - out_all_2).sum())
        # print("output, shape:{}, min/max: {}/{}".format(out.shape, out.min(), out.max()))
        break
    print("end...")


def load_weights_and_freeze():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model


if __name__ == "__main__":
    os.environ["EXP_DIR"] = "../experiments"
    os.environ["DATA_DIR"] = "../data"

    # pre_trained()
    load_expert_demo_data()

