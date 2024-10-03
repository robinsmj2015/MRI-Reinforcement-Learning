from torch import nn
from torch.optim import Adam
from torch.nn import Linear
import matplotlib.pyplot as plt
from torch import save
from torch import load
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
from Box import Box
#from torchrl import record
import random
import numpy as np
from CustomResNet import CustomResNet
from itertools import chain
import pandas as pd
import os
import torchmetrics
import datetime
from torchsummary import summary
import typing

class SuperviseResNet:
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

        self.net = net
        self.criterion = nn.BCEWithLogitsLoss()
        self.out = Linear(flatten_nodes, 1, device=self.device, bias=False)
        self.optimizer = Adam(lr=lr, amsgrad=True, params=chain(params, self.out.parameters()))

        #self.out.weight.data = torch.ones((1, flatten_nodes), dtype=torch.float, device=self.device) / flatten_nodes
        self.f1s = []
        self.precs = []
        self.accs = []
        self.recs = []
        self.specs = []
        self.e_losses = []
        self.b_losses = []
        self.val_losses = {}

        #self.logger = record.CSVLogger(exp_name="my_exp")


    def train(self, data, val_data, val_period, epochs):
        self.net.train()

        for i in range(epochs):
            for inputs, labels in data:

                labels = torch.unsqueeze(labels.to(torch.float32), dim=1).to(self.device)
                outs = self.out(self.net.forward(inputs.to(self.device)))
                loss = self.criterion(outs, labels)
                self.b_losses.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if i % val_period == 0:
                for val_inputs, val_labels in val_data:
                    val_outs = self.out(self.net.forward(val_inputs.to(self.device)))
                    val_loss = self.criterion(val_outs, torch.unsqueeze(val_labels, dim=1).to(self.device))
                    self.val_losses[i] = torch.mean(val_loss.detach().cpu())  # todo this is not right...

            print(f"{(i + 1) / epochs:.1%} complete")

    def validate(self, data):
        self.net.eval()
        for inputs, labels in data:
            outs = self.out(self.net.forward(inputs.to(self.device)))
            labels = labels.to(torch.float32)


            rev_labels = torch.unsqueeze(labels, dim=1).to(self.device)

            f1 = torchmetrics.functional.f1_score(outs, rev_labels, "binary")
            acc = torchmetrics.functional.accuracy(outs, rev_labels, "binary")
            prec = torchmetrics.functional.precision(outs, rev_labels, "binary")
            rec = torchmetrics.functional.recall(outs, rev_labels, "binary")
            spec = torchmetrics.functional.specificity(outs, rev_labels, "binary")
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
        #for file in os.listdir("plt_ims"):
            #os.remove("plt_ims/" + file)
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
        #print(epochs_to_plot)

    def save_model(self, version, box_mode, metric, euclid_thresh, size, acc, path_mode):
        #for file in os.listdir("models"):
           # os.remove("models/" + file)
        time = str(datetime.datetime.now().strftime("%b %d, %I_%M%p"))
        save(self.net.state_dict(), f"models/resnet acc={int(100 * acc)}, V={version}, bm={box_mode}_{size}, m={metric}_{euclid_thresh}, path={path_mode} " + time + ".pt")


class SuperviseDataset(Dataset):
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
        x = self.transform(box, master_file)
        return x, y

    def transform(self, box, master_file, resized_dims=(224, 224)):
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


def gen_sup_learn(slices2d, size, epochs, metric="euclid", euclid_thresh=None, dims=(224, 224)):
    length = dims[0]
    inputs = pd.DataFrame(columns=["name", "x", "y", "label"])
    half = size // 2
    for _ in range(epochs):
        for slice2d in slices2d:
            box = slices2d[slice2d].box
            crit_length = length - half - 1
            if metric.lower() == "center":
                x_neg = random.choice(list(set(np.arange(half, crit_length)) - set(np.arange(box.left_most, box.right_most))))
                y_neg = random.choice(list(set(np.arange(half, crit_length)) - set(np.arange(box.top_most, box.bottom_most))))

                x_pos = random.randint(max(half, box.left_most), min(box.right_most, crit_length))
                y_pos = random.randint(max(half, box.top_most), min(box.bottom_most, crit_length))

            elif metric.lower() == "root2":
                root2 = int(round(euclid_thresh / (2 ** 0.5)))
                x_neg = random.choice(list(set(np.arange(half, crit_length)) - set(np.arange(box.center[0] - root2, box.center[0] + root2))))
                y_neg = random.choice(list(set(np.arange(half, crit_length)) - set(np.arange(box.center[1] - root2, box.center[1] + root2))))

                x_pos = random.randint(max(half, box.center[0] - root2), min(crit_length, box.center[0] + root2))
                y_pos = random.randint(max(half, box.center[1] - root2), min(crit_length, box.center[1] + root2))

            else:
                # euclid
                x_neg = random.choice(list(set(np.arange(half, crit_length)) - set(np.arange(box.center[0] - euclid_thresh, box.center[0] + euclid_thresh))))
                y_neg = random.choice(list(set(np.arange(half, crit_length)) - set(np.arange(box.center[1] - euclid_thresh, box.center[1] + euclid_thresh))))

                x_pos = random.randint(max(half, box.center[0] - euclid_thresh), min(crit_length, box.center[0] + euclid_thresh))
                y_pos = random.randint(max(half, box.center[1] - euclid_thresh), min(crit_length, box.center[1] + euclid_thresh))

            inputs.loc[len(inputs)] = [slices2d[slice2d].name, x_neg, y_neg, 0.]
            inputs.loc[len(inputs)] = [slices2d[slice2d].name, x_pos, y_pos, 1.]
    return inputs


def run_supervised_learning(train_slices, val_slices, train_epochs, train_batch, val_batch, t_data_epochs, lr, box_mode, max_zoom_fit=4,
                            size=32, v_data_epochs=1, val_period=1, version=50, metric="center", euclid_thresh=10, path_abr="E", scan_type="T1"):

    train_inputs = gen_sup_learn(train_slices, size, epochs=t_data_epochs, metric=metric, euclid_thresh=euclid_thresh)
    val_inputs = gen_sup_learn(val_slices, size, epochs=v_data_epochs, metric=metric, euclid_thresh=euclid_thresh)

    res = CustomResNet(box_mode.count("_") + 1, torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu"), version, False)
    sup_resnet = SuperviseResNet(res.parameters(), res, lr, version)
    train_sup_dataset = SuperviseDataset(train_inputs, box_mode, max_zoom_fit, size)
    val_sup_dataset = SuperviseDataset(val_inputs, box_mode, max_zoom_fit, size)
    t_dl = DataLoader(train_sup_dataset, batch_size=train_batch, shuffle=True)
    v_dl = DataLoader(val_sup_dataset, batch_size=val_batch, shuffle=True)
    print("VALIDATION INITIAL:")
    _ = sup_resnet.validate(v_dl)

    print("\nTRAINING NOW...")
    sup_resnet.train(t_dl, v_dl, val_period=val_period, epochs=train_epochs)

    print("\nVALIDATION FINAL: ")
    scores = sup_resnet.validate(v_dl)
    sup_resnet.save_model(version, box_mode, metric, euclid_thresh, size, np.mean(np.array(scores[1])), path_abr + "," + scan_type)

    sup_resnet.plot_losses(epochs=train_epochs)

