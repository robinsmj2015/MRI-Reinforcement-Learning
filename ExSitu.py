import torch.cuda
from CustomResNet import CustomResNet
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from torch import save
from torch import load
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
from Box import Box
import random
import numpy as np
import pandas as pd
from torchmetrics import functional
import datetime
from torch.nn import Sigmoid


class ExSitu:
    def __init__(self, params, net, lr, version):
        flatten_nodes = 0
        if version == 18:
            flatten_nodes = 512
        elif version == 34:
            flatten_nodes = 512
        elif version == 50:
            flatten_nodes = 2048
        elif version == 101:
            flatten_nodes = 2048
        self.device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
        self.optimizer = Adam(lr=lr, amsgrad=True, params=params)
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.out = nn.Linear(flatten_nodes, 4, device=self.device, bias=False)
        self.sigmoid = Sigmoid()
        self.f1s = []
        self.accs = []
        self.precs = []
        self.recs = []
        self.specs = []
        self.b_losses = []
        self.val_losses = {}

    def train_ex(self, data, val_data, val_period, epochs):
        self.net.train()
        for i in range(epochs):
            for inputs, labels in data:
                outs = self.sigmoid(self.out(self.net.forward(inputs.to(self.device))))
                loss = self.criterion(outs, labels.to(self.device))
                self.b_losses.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if i % val_period == 0:
                for val_inputs, val_labels in val_data:
                    val_outs = self.sigmoid(self.out(self.net.forward(val_inputs.to(self.device))))
                    val_loss = self.criterion(val_outs, val_labels.to(self.device))
                    self.val_losses[i] = torch.mean(val_loss.detach().cpu())

            print(f"{(i + 1) / epochs:.1%} complete")

    def validate_ex(self, data):
        self.net.eval()
        for inputs, labels in data:
            outs = self.sigmoid(self.out(self.net.forward(inputs.to(self.device))))
            rev_labels = labels.to(self.device)
            f1 = functional.classification.f1_score(outs, rev_labels, num_classes=4, num_labels=4, average="micro", task="multilabel")
            acc = functional.classification.accuracy(outs, rev_labels, num_classes=4, num_labels=4, average="micro", task="multilabel")
            rec = functional.classification.recall(outs, rev_labels, num_classes=4, num_labels=4, average="micro", task="multilabel")
            prec = functional.classification.precision(outs, rev_labels, num_classes=4, num_labels=4, average="micro", task="multilabel")
            spec = functional.classification.specificity(outs, rev_labels, num_classes=4, num_labels=4, average="micro", task="multilabel")
            self.f1s.append(f1.detach().cpu())
            self.accs.append(acc.detach().cpu())
            self.precs.append(prec.detach().cpu())
            self.recs.append(rec.detach().cpu())
            self.specs.append(spec.detach().cpu())

        names = ["f1", "accuracy", "precision", "recall", "specificity"]
        scores = [self.f1s, self.accs, self.precs, self.recs, self.specs]
        print([f"{name}: {np.mean(np.array(score)):.3f}"for name, score in zip(names, scores)])
        return scores

    def plot_losses(self, epochs):
        # for file in os.listdir("plt_ims"):
        # os.remove("plt_ims/" + file)
        to_plot = np.array([l.detach().cpu() for l in self.b_losses])
        arrays = np.array_split(to_plot, epochs)
        epochs_to_plot = [np.round(np.mean(arr), 3) for arr in arrays]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax.plot(np.arange(len(to_plot)), to_plot)
        ax2.plot(np.arange(len(epochs_to_plot)), epochs_to_plot, color="green", label="train")
        ax2.plot(list(self.val_losses.keys()), list(self.val_losses.values()), color="red", label="val")
        ax2.legend()
        ax.set(title="batch losses")
        ax2.set(title="epoch losses")
        time = str(datetime.datetime.now().strftime("%b %d, %Y %I_%M%p"))
        plt.savefig("plt_ims/losses " + time + " .png")
        # print(epochs_to_plot)

    def save_model(self, version, box_mode, metric, euclid_thresh, size, acc, path_mode):
        # for file in os.listdir("models"):
        # os.remove("models/" + file)
        time = str(datetime.datetime.now().strftime("%b %d, %I_%M%p"))
        save(self.net.state_dict(),
             f"models/resnet ex_situ acc={int(100 * acc)}, V={version}, bm={box_mode}_{size}, m={metric}_{euclid_thresh}, path={path_mode} " + time + ".pt")


def ex_situ_data(slices2d, size, epochs, dims=(224, 224)):
    length = dims[0]
    inputs = pd.DataFrame(columns=["name", "x", "y", "label"])
    half = size // 2
    for _ in range(epochs):
        for slice2d in slices2d:
            box = slices2d[slice2d].box
            crit_length = length - half - 1
            y_u = random.randint(box.center[1] + half, crit_length)
            y_d = random.randint(half, min(crit_length, box.center[1] - half - 1))
            x_l = random.randint(box.center[0] + half, crit_length)
            x_r = random.randint(half, min(crit_length, box.center[0] - half - 1))

            inputs.loc[len(inputs)] = [slices2d[slice2d].name, x_l, y_u, torch.tensor([1., 0., 1., 0.])]
            inputs.loc[len(inputs)] = [slices2d[slice2d].name, x_l, y_d, torch.tensor([0., 1., 1., 0.])]
            inputs.loc[len(inputs)] = [slices2d[slice2d].name, x_r, y_u, torch.tensor([1., 0., 0., 1.])]
            inputs.loc[len(inputs)] = [slices2d[slice2d].name, x_r, y_d, torch.tensor([0., 1., 0., 1.])]
    return inputs


class ExSituDataset(Dataset):
    def __init__(self, input_df, box_mode, zoom_fit, size):
        super().__init__()
        self.input_df = input_df  # data to combine into dataset
        self.box_mode = box_mode
        self.max_zoom_fit = zoom_fit
        self.fill_value = -1 * (0.485 + 0.456 + 0.406) / (0.229 + 0.224 + 0.225)
        self.data_path = "data/scans/"
        self.size = size

    def __len__(self):
        return len(self.input_df)

    def ex_situ_transform(self, box, master_file, resized_dims=(224, 224)):
        pad_box = v2.Compose([v2.Pad([int(box.left_most), int(box.top_most),
                                      int(resized_dims[0] - box.right_most),
                                      int(resized_dims[1] - box.bottom_most)], fill=self.fill_value)])

        slice_chan = load(self.data_path + master_file)  # 0th channel
        cropped = slice_chan[:, int(box.top_most): int(box.bottom_most), int(box.left_most): int(box.right_most)]
        pad_chan = pad_box(cropped)  # 1st channel
        fit_chan = None
        if "F" in self.box_mode:
            zoomed_size1, zoomed_size2 = cropped.size()[1] * self.max_zoom_fit, cropped.size()[2] * self.max_zoom_fit,
            hor_pad = resized_dims[1] - zoomed_size1
            vert_pad = resized_dims[0] - zoomed_size2
            zoom = v2.Compose([v2.Resize([zoomed_size1, zoomed_size2], antialias=False),
                               v2.Pad([0, 0, hor_pad, vert_pad], fill=self.fill_value)])
            fit_chan = zoom(cropped)  # 2nd channel

        if self.box_mode == "S":
            inputs = slice_chan
        elif self.box_mode == "P":
            inputs = pad_chan
        elif self.box_mode == "F":
            inputs = fit_chan
        elif self.box_mode == "P_F":
            inputs = torch.stack([torch.squeeze(pad_chan), torch.squeeze(fit_chan)])
        elif self.box_mode == "S_F":
            inputs = torch.stack([torch.squeeze(slice_chan), torch.squeeze(fit_chan)])
        elif self.box_mode == "S_P":
            inputs = torch.stack([torch.squeeze(slice_chan), torch.squeeze(pad_chan)])
        else:
            # "S_P_F":
            inputs = torch.stack([torch.squeeze(slice_chan), torch.squeeze(pad_chan), torch.squeeze(fit_chan)])
        return inputs

    def __getitem__(self, idx):
        # no labels for prediction mode
        # combine file methods combines the state files together
        y = self.input_df.iloc[idx]["label"]
        br_x = self.input_df.iloc[idx]["x"] + self.size // 2
        br_y = self.input_df.iloc[idx]["y"] + self.size // 2
        tl_x = br_x - self.size
        tl_y = br_y - self.size

        master_file = (self.input_df.iloc[idx]["name"])
        box = Box((tl_x, tl_y), (br_x, br_y))
        x = self.ex_situ_transform(box, master_file)
        return x, y


def run(train_slices, val_slices, path_mode):
    box_mode = "F"
    size = 56
    train_epochs = 50
    val_period = 5
    t_data_epochs = 25
    v_data_epochs = 1

    train_inputs = ex_situ_data(train_slices, size, epochs=t_data_epochs)
    val_inputs = ex_situ_data(val_slices, size, epochs=v_data_epochs)
    res = CustomResNet(box_mode.count("_") + 1, torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu"), 50, False)
    ex = ExSitu(res.parameters(), res, lr=5e-5, version=50)
    train_sup_dataset = ExSituDataset(train_inputs, box_mode, 4, size)
    val_sup_dataset = ExSituDataset(val_inputs, box_mode, 4, size)
    t_dl = DataLoader(train_sup_dataset, batch_size=64, shuffle=True)
    v_dl = DataLoader(val_sup_dataset, batch_size=128, shuffle=True)
    print("VALIDATION INITIAL:")
    _ = ex.validate_ex(v_dl)

    print("\nTRAINING NOW...")
    ex.train_ex(t_dl, v_dl, val_period=val_period, epochs=train_epochs)

    print("\nVALIDATION FINAL: ")
    scores = ex.validate_ex(v_dl)
    ex.save_model(50, box_mode, "euclid", 15, size, np.mean(np.array(scores[1])), path_mode)

    ex.plot_losses(epochs=train_epochs)

