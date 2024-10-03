from torchvision.models.resnet import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torch import nn
from torch import flatten
import matplotlib.pyplot as plt
import torch

class CustomResNet(nn.Module):
    def __init__(self, num_channels, device, version, to_visualize):
        super().__init__()
        if version == 18:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        elif version == 34:
            self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT).to(device)
        elif version == 50:
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        elif version == 101:
            self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT).to(device)
        self.to_visualize = to_visualize
        copy_conv1 = self.resnet.conv1.weight.data.clone()
        del self.resnet.fc
        del self.resnet.conv1
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, device=device)
        self.conv1.weight.data = copy_conv1[:, 0:num_channels, :, :]
        self.device = device

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = flatten(x, 1)
        return x

    def forward(self, x):
        return self._forward_impl2(x) if self.to_visualize else self._forward_impl(x)

    @staticmethod
    def plot_init(x):
        fig, ax = plt.subplots(1, x.size(dim=1))
        if x.size(dim=1) == 1:
            ax.imshow(x[0, 0], cmap="Greys")
            ax.axis("off")
        else:
            for i, a in enumerate(ax):
                a.imshow(x[0, i], cmap="Greys")
                a.axis("off")

        plt.gcf().suptitle("Initial")
        plt.tight_layout()
        plt.gcf().canvas.manager.full_screen_toggle()
        plt.show()

    @staticmethod
    def plot_inter_net(x, name, plots_per_length=3):
        for i in range(plots_per_length ** 2):
            ax = plt.subplot(plots_per_length, plots_per_length, i + 1)
            ax.axis("off")
            ax.imshow(x[0, i], cmap="Greys")
        plt.gcf().suptitle(name)
        plt.tight_layout()
        plt.gcf().canvas.manager.full_screen_toggle()
        plt.show()

    def _forward_impl2(self, x):
        self.plot_init(x)
        x = self.conv1(x)
        self.plot_inter_net(x, "After conv 1")
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        self.plot_inter_net(x, "After max pool")
        x = self.resnet.layer1(x)
        self.plot_inter_net(x, "After layer 1")
        x = self.resnet.layer2(x)
        self.plot_inter_net(x, "After layer 2")
        x = self.resnet.layer3(x)
        self.plot_inter_net(x, "After layer 3")
        x = self.resnet.layer4(x)
        self.plot_inter_net(x, "After layer 4")
        x = self.resnet.avgpool(x)
        x = flatten(x, 1)
        return x
