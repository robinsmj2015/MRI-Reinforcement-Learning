import torch
from torch import nn
from torch import device
from CustomResNet import CustomResNet


class Net(nn.Module):
    def __init__(self, name, num_im_per_state, num_axns, resized_dims, is_using_server, multi_res, best_gpu_nums, version, to_visualize):
        super().__init__()
        flatten_nodes = 0
        if version == 18:
            flatten_nodes = 512
        elif version == 34:
            flatten_nodes = 512
        elif version == 50:
            flatten_nodes = 2048
        elif version == 101:
            flatten_nodes = 2048

        self.num_axns = num_axns
        self.resized_dims = resized_dims
        self.multi_res = multi_res
        self.name = name
        self.is_using_server = is_using_server
        self.fc_hidden0 = nn.Linear(flatten_nodes * num_im_per_state if multi_res else flatten_nodes, 1024, device=torch.device("cuda:" + str(best_gpu_nums[0])) if is_using_server else torch.device("cpu"))
        self.fc_hidden1 = nn.Linear(1024, 512, device=torch.device("cuda:" + str(best_gpu_nums[0])) if is_using_server else torch.device("cpu"), bias=True)
        self.out = nn.Linear(512, num_axns, device=torch.device("cuda:" + str(best_gpu_nums[0])) if is_using_server else torch.device("cpu"), bias=True)
        nn.init.zeros_(self.out.bias.data)
        self.leaky_relu = nn.LeakyReLU()
        self.models = []
        self.best_gpu_nums = best_gpu_nums
        if self.multi_res:
            for i in range(num_im_per_state):
                device_str = "cuda:" + str(self.best_gpu_nums[i]) if is_using_server else "cpu"
                self.models.append(CustomResNet(1, device(device_str), version, to_visualize))
        else:
            device_str = "cuda:" + str(self.best_gpu_nums[0]) if is_using_server else "cpu"
            self.models.append(CustomResNet(num_im_per_state, device(device_str), version, to_visualize))

    def forward(self, x):
        z_device = device("cuda:" + str(self.best_gpu_nums[0])) if self.is_using_server else device("cpu")
        z = None
        if self.multi_res:
            for i in range(x.shape[1]):
                model = self.models[i]
                inputs = torch.unsqueeze(x[:, i, :, :], dim=1).to(model.device)
                y = model.forward(inputs).to(z_device)
                if not isinstance(z, type(None)):
                    z = torch.cat((z, y), dim=-1)
                else:
                    z = torch.Tensor.copy_(y, True)
        else:
            model = self.models[0]
            x = x.to(model.device)
            z = model.forward(x).to(z_device)

        z = self.leaky_relu(z)
        z = self.leaky_relu(self.fc_hidden0(z))
        z = self.leaky_relu(self.fc_hidden1(z))
        z = self.out(z)
        return z

