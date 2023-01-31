import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from spirl.utility.pytorch_utils import DataParallelWrapper, RAdam
from spirl.modules.variational_inference import MultivariateGaussian, Gaussian
from spirl.utility.general_utils import AttrDict
from spirl.modules.losses import KLDivLoss, NLL
from spirl.utility.pytorch_utils import _gather


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 3 * 2)
        self.fc2 = nn.Linear(input_size, input_size)
        # self.last = MultivariateGaussian(mu=input_size, log_sigma=None)

    def forward(self, input):
        # p, q, z, recon

        out = AttrDict()
        _q = self.fc1(input)
        # print("_q: ", _q.shape, _q[:, -1].shape)
        out.p = MultivariateGaussian(mu=_q[:, :])
        # print("p mu: ", out.p.mu.shape, " p sig shape ", out.p.log_sigma.shape)
        out.q = Gaussian(torch.zeros_like(out.p.mu), torch.ones_like(out.p.log_sigma))
        # out.z = MultivariateGaussian(mu=_q[:, :]).rsample()
        out.z = out.p.rsample()
        # print("z: ", out.z.shape)

        output = self.fc2(input)
        output = torch.relu(output)
        out.recon = output
        # output = self.last.rsample()
        # print("\tIn Model: input size", input.size(),
        #       "output size", output.size())

        return out

    def loss(self, model_out, label):
        # p, z, recon
        losses = AttrDict()
        losses.pq_loss = KLDivLoss()(model_out.q, model_out.p)
        # losses.recon_loss =
        losses.total = self._compute_total_loss(losses)
        return losses

    @staticmethod
    def _compute_total_loss(losses):
        total_loss = torch.stack([loss[1].value * loss[1].weight for loss in
                                  filter(lambda x: x[1].weight > 0, losses.items())]).sum()
        return AttrDict(value=total_loss)


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.rand(length, size)
        self.label = torch.where(torch.rand(len(self.data)) > 0.5, 1.0, 0.0)

    def __getitem__(self, index):
        return self.data[index], self.label

    def __len__(self):
        return self.len


def multi_gpu_proc():
    input_size = 5
    output_size = 1

    batch_size = 32
    data_size = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device: ", device)

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                             batch_size=batch_size, shuffle=True)

    model = Model(input_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), " GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...] on 2 GPUs
        # model = nn.DataParallel(model)
        model = DataParallelWrapper(model, output_device=device)

    model.to(device)

    start = time.time()
    n_epochs = 100
    for epoch in range(n_epochs):
        for sample in rand_loader:
            data, lable = sample
            data = data.to(device)
            lable = lable.to(device)

            output = model(data)
            loss = model.loss(model_out=output, label=lable)
            optimizer.zero_grad()
            loss.total.value.backward()
            optimizer.step()

    print("=======================")
    print("Elapsed learning time: ", time.time() - start)


def gather_map(outputs):
    out = outputs[0]
    if isinstance(out, torch.Tensor):
        return Gather.apply(target_device, dim, *outputs)
    if out is None:
        return None
    if isinstance(out, dict):
        if not all((len(out) == len(d) for d in outputs)):
            raise ValueError('All dicts must have the same number of keys')
        return type(out)(((k, gather_map([d[k] for d in outputs]))
                          for k in out))
    try:
        # print("outputs... ", outputs)
        mus = [d.mu for d in outputs]
        sigmas = [d.sigma for d in outputs]
        mu_out = Gather.apply(target_device, dim, *mus)
        sigma_out = Gather.apply(target_device, dim, *sigmas)

        return type(out)(mu=mu_out, sigma=sigma_out)
        # return type(out)(map(gather_map, zip(*outputs)))
    except:
        return type(out).reduce(*outputs)


if __name__ == "__main__":
    print("multi gpu example code!")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    multi_gpu_proc()
    exit()

    from torch.nn.parallel._functions import Gather

    class Model:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma

        # @property
        # def device(self):
        #     return self.sigma.device
        #
        # @property
        # def shape(self):
        #     return self.sigma.shape

    # a = torch.rand(2, 3, device="cuda:0")
    # b = torch.rand(2, 3, device="cuda:1")
    batch = 2
    a = Model(mu=torch.rand(batch, 3, device="cuda:0"), sigma=torch.rand(batch, 3, device="cuda:0"))
    b = Model(mu=torch.rand(batch, 3, device="cuda:1"), sigma=torch.rand(batch, 3, device="cuda:1"))
    # print("dev num, a: {},  shape: {}".format(a.device, a.shape))
    # print("dev num, b: {},  shape: {}".format(b.device, b.shape))
    ar = [a.mu, b.mu]
    print("zip ", list(zip(*ar)))

    target_device = 0   # just number
    dim = 0
    out = Gather.apply(target_device, dim, *ar)
    print("out: ", out)
    # for i, j in zip(*ar):
    #     print("i: ", i, j)
    #     out = Gather.apply(target_device, dim, *ar)
    #     print("out: ", out.shape)
    # m = Model(mu=mu_out, sigma=)
    # print("final ", m.mu)
    # out = Gather.apply(target_device, dim, *ar)
    # print(out, out.shape)

    # # just after scatter...
    # batch = 2
    # model1 = MultivariateGaussian(mu=torch.zeros(batch // 2, 6, device="cuda:0"))
    # print("model 1 dev num: {},  shape: {}", model1.tensor().get_device())
    # model2 = MultivariateGaussian(mu=torch.rand(batch // 2, 6, device="cuda:1"))
    # print("model 2 dev num: ", model2.tensor().get_device())
    # models = [model1, model2]
    # print("models: ", models)


    # print(
    #     (get_mu([zip(*models)]))
    # )
    # out = MultivariateGaussian.cat(*models, dim=0)
    # print("model1 mu ", model1.mu.shape)
    # print("model2 mu ", model2.mu.shape)
    # print("out mu ", out.mu.shape)
    # model = DataParallelWrapper(model)
    # members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith('_')]
    # print(members)
    # print([getattr(model, mem).shape for mem in members if isinstance(getattr(model, mem), torch.Tensor)])
    # out = gather_map(m)
    # out = m.rsample()
    # m2 = type(m)(mu=torch.zeros(64, 6))
    # print("m: ", m)
    # print("out: ", out.shape)
    # print("mu: {}, sigma: {}".format(m.mu.shape, m.log_sigma.shape))
    # print("shape: {}, val: {}".format(out.shape, out))
