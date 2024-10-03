import matplotlib.pyplot as plt
import pandas as pd
from Box import Box
import os
from PIL import Image
from torch import save
from torch import load
import torch
import numpy as np
import datetime
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2


class FileProcessing:
    def __init__(self, box_mode, is_using_server, epochs, max_zoom_fit, normalizing_thresh):
        self.epochs = epochs
        self.is_using_server = is_using_server
        self.transition_path = "transitions/"
        self.logs_path = "logs/"
        self.plt_ims_path = "plt_ims/"
        self.models_path = "models/"
        self.full_models_path = "full_models/"
        self.data_splits_path = "data_splits/"
        self.summaries_path = "summaries/"
        self.start_path = "starts/"
        self.current_log_file = None
        self.box_mode = box_mode
        self.slice_file_name = None
        self.max_zoom_fit = max_zoom_fit
        self.normalizing_thresh = normalizing_thresh
        self.fill_value = -1 * (0.485 + 0.456 + 0.406) / (0.229 + 0.224 + 0.225)
        self.data_path = "data/scans/"
        col_names = ["times", "steps", "ious", "euclids"]
        endings = ["_median", "_mean", "_std", "_total"]
        final_cols = ["training", "times", "steps", "ious", "euclids"]
        for col_name in col_names:
            for ending in endings:
                final_cols.append(col_name + ending)
        self.summary_df = pd.DataFrame(columns=final_cols)
        self.summary_df.index.name = "Results"

    # gets the bounding boxes from csv file
    def process_box_file(self, dims, resized_dims, path_abr, scan_type):
        # SAGIR: (256, 242) 23%, (256, 256) 11%, (512, 512) 31%
        # SAGT1: (768, 768) 19%, (864, 864) 49%
        if path_abr == "E":
            pathology = "['Edema']"
        elif path_abr == "F":
            pathology = "['Fracture']"
        else:
            pathology = "['Bone']"
        if path_abr != "B":
            scan_type = "SAG" + scan_type + " bounding boxes.csv"
            raw_df = pd.read_csv(scan_type)
        else:
            raw_df = pd.read_csv("SAGT1 Bone.csv")
            scan_type = "SAGT1 bounding boxes.csv"
        df = raw_df.loc[(raw_df["im_w"] == dims[0]) & (raw_df["im_h"] == dims[1]) &
                        (raw_df["box_type"] == pathology)]

        df = df[df.im.str.contains(scan_type.split(" ")[0])]
        # removing non-zero rotation images
        if "SAGT1" in scan_type:
            df = df[df.r == 0]
        df = df.drop_duplicates(subset="im")
        df["im"] = df["im"].apply(lambda m: self.get_file_path(m))
        df.set_index("im", inplace=True)
        df = df[["box_x", "box_y", "box_w", "box_h"]]
        factor = 1 if path_abr == "B" else 100
        # axis 1 is row wise!
        df["boxes"] = df.apply(lambda row: self.process_box_file_helper(row, resized_dims, factor), axis=1)
        return df["boxes"]

    # a function (for lambda) that actually makes the box instances (to be stored in the dataframe)
    # np arrays just temporarily used to simplify rounding... note box coordinates are saved as list
    # rounding is used since we must use whole numbers for pixels

    def process_box_file_helper(self, row, resized_dims, factor):
        tl_0 = np.array([(row["box_x"] / factor) * resized_dims[0], (row["box_y"] / factor) * resized_dims[1]])
        delta_y = (row["box_h"] / factor) * resized_dims[1]
        delta_x = (row["box_w"] / factor) * resized_dims[0]
        br_0 = np.array([tl_0[0] + delta_x, tl_0[1] + delta_y])
        b0, b90, b180, b270 = self.get_rotations(np.round(tl_0).astype(int), np.round(br_0).astype(int))
        return [b0, b90, b180, b270]

    # a simple helper function to get file names when processing box files
    def get_file_path(self, m):
        folder_path = os.path.join(self.data_path + m[:10])
        (_, to_append, _), _ = os.walk(folder_path)
        name = str(os.path.join(m[:10], to_append[0], m)).replace(".jpg", " RES.jpg")
        return name

    @staticmethod
    def get_rotations(tl_0, br_0):
        size = 224
        box0 = Box(tl_0, br_0)

        tl_90 = np.array([box0.tr.copy()[1], size - box0.tr.copy()[0]])
        br_90 = np.array([box0.bl.copy()[1], size - box0.bl.copy()[0]])

        tl_180 = np.array([size - box0.br.copy()[0], size - box0.br.copy()[1]])
        br_180 = np.array([size - box0.tl.copy()[0], size - box0.tl.copy()[1]])

        tl_270 = np.array([size - box0.bl.copy()[1], box0.bl.copy()[0]])
        br_270 = np.array([size - box0.tr.copy()[1], box0.tr.copy()[0]])

        return box0, Box(tl_90, br_90), Box(tl_180, br_180), Box(tl_270, br_270)

    @staticmethod
    def rotate_box(box0, turns):
        size = 224
        if turns == 0:
            return Box(box0.tl.copy(), box0.br.copy())
        if turns == 1:
            tl_90 = np.array([box0.tr.copy()[1], size - box0.tr.copy()[0]])
            br_90 = np.array([box0.bl.copy()[1], size - box0.bl.copy()[0]])
            return Box(tl_90, br_90)
        if turns == 2:
            tl_180 = np.array([size - box0.br.copy()[0], size - box0.br.copy()[1]])
            br_180 = np.array([size - box0.tl.copy()[0], size - box0.tl.copy()[1]])
            return Box(tl_180, br_180)
        if turns == 3:
            tl_270 = np.array([size - box0.bl.copy()[1], box0.bl.copy()[0]])
            br_270 = np.array([size - box0.tr.copy()[1], box0.tr.copy()[0]])
            return Box(tl_270, br_270)


    # puts transition images in folder

    # 5-7-24
    # let's save all 3 types of images... pad (pos=true), slice and fit
    # codes T - train, I - inference, H - heatmap
    # use 3d tensor save as... Slice - Pad - Fit
    # giving us ... ie I_S12.pt
    def create_transition_file(self, resized_dims, code, box, master_file, track_num):
        fit_chan = None
        file_name = code + str(track_num) + ".pt"
        pad_box = v2.Compose([v2.Pad([int(box.left_most), int(box.top_most),
                                      int(resized_dims[0] - box.right_most),
                                      int(resized_dims[1] - box.bottom_most)], fill=self.fill_value)])

        slice_chan = load(self.data_path + master_file)  # 0th channel

        cropped = slice_chan[:, int(box.top_most): int(box.bottom_most), int(box.left_most): int(box.right_most)]
        pad_chan = pad_box(cropped)  # 1st channel
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
        torch.save(inputs, self.transition_path + file_name)
        return file_name

    # deletes ALL files of the least recent (most out-dated) episode
    def delete_transition_file(self, file_name):
        os.remove(self.transition_path + file_name)

    def create_start_file(self, max_slices, scaling_factor, train_slices, infer_slices):
        if not os.path.exists(self.start_path):
            os.mkdir(self.start_path)
        df = pd.DataFrame(columns=["tl", "br"],
                          index=[key for key in
                                 (list(infer_slices.keys()) + list(train_slices.keys()))[:max_slices]])
        for key in list(infer_slices.keys())[:max_slices]:
            df.at[key, "tl"] = [infer_slices[key].proposed_start.tl[0] * scaling_factor,
                                infer_slices[key].proposed_start.tl[1] * scaling_factor]
            df.at[key, "br"] = [infer_slices[key].proposed_start.br[0] * scaling_factor,
                                infer_slices[key].proposed_start.br[1] * scaling_factor]
        for key in list(train_slices.keys())[:max_slices]:
            df.at[key, "tl"] = [train_slices[key].proposed_start.tl[0] * scaling_factor,
                                train_slices[key].proposed_start.tl[1] * scaling_factor]
            df.at[key, "br"] = [train_slices[key].proposed_start.br[0] * scaling_factor,
                                train_slices[key].proposed_start.br[1] * scaling_factor]
        df.to_csv(self.start_path + "new_start.csv")

    def read_start_file(self):
        df = pd.read_csv(self.start_path + "new_start.csv", index_col=[0])
        return df

    # clearing folders
    def clear_folders(self, to_clear_starts, to_clear_models, to_clear_transitions):
        folders = [self.plt_ims_path, self.logs_path, self.summaries_path, self.data_splits_path]
        if to_clear_starts:
            folders.append(self.start_path)
        if to_clear_models:
            folders.append(self.models_path)
        if to_clear_transitions:
            folders.append(self.transition_path)

        for folder in folders:
            if not os.path.exists(folder):
                os.mkdir(folder)
            else:
                for file in os.listdir(folder):
                    os.remove(folder + file)

    # makes a file of results for training and testing
    def make_summary_file(self, epoch_num, steps, times, ious, euclids, is_training):
        steps = np.array(steps, dtype=float)
        times = np.array(times, dtype=float)
        ious = np.array(ious, dtype=float)
        euclids = np.array(euclids, dtype=float)
        is_training = is_training
        values = [is_training, times, steps, ious, euclids] + [0] * (len(self.summary_df.columns) - 5)
        self.generate_summary_file(epoch_num, values)

    # summary file
    def generate_summary_file(self, epoch_num, values):
        for file in os.listdir(self.summaries_path):
            self.summary_df = pd.read_csv(self.summaries_path + file)
            self.summary_df.set_index(self.summary_df.columns[0], inplace=True)
            os.remove(self.summaries_path + file)
        self.summary_df.loc[epoch_num] = values
        for variable in ["times", "steps", "ious", "euclids"]:
            for func, name in zip([np.median, np.mean, np.std, np.sum], ["_median", "_mean", "_std", "_total"]):
                self.summary_df.loc[epoch_num, variable + name] = func(self.summary_df.at[epoch_num, variable])
        self.summary_df.to_csv(self.summaries_path + str(datetime.datetime.now().strftime("%b %d, %Y %I_%M%p")) +
                               " summary.csv")

    # log file creation
    def create_log_file(self, to_save, file_name, to_create_new_file, is_training):
        if to_create_new_file:
            self.current_log_file = self.logs_path + str(datetime.datetime.now().strftime("%b %d, %Y %I_%M%p")) + \
                                    " Training " + str(is_training) + " " + file_name.replace("jpg", "txt")
            file = open(self.current_log_file, "w")
        else:
            file = open(self.current_log_file, "a")
        file.write(to_save)
        file.close()

    # opens log file
    @staticmethod
    def choose_file(ending, path):
        log_file_dic = {}  # log files to choose from
        num = 0
        # selects the file & opens it
        for file_name in os.listdir(path):
            if file_name.endswith(ending):
                print("{0}. {1}".format(num, file_name))
                log_file_dic[num] = file_name
                num += 1
        if num == 1:
            selected = 0
        else:
            selected = input("Enter number to select file: ")
            if selected.isnumeric():
                selected = int(selected)
            else:
                print("Invalid entry... selecting the first file available")
                selected = 0
        file_name = log_file_dic[selected]
        file = open(path + file_name, "r")
        return file, file_name

    # saving the networks with current time
    def save_net(self, p_net, p_net_controller):
        if not os.path.exists(self.full_models_path):
            os.mkdir(self.full_models_path)
        for file in os.listdir(self.full_models_path):
            os.remove(self.full_models_path + file)
        save({"weights": p_net.state_dict(), "optim": p_net_controller.optimizer.state_dict()},
             self.full_models_path + str(datetime.datetime.now().strftime("%b %d, %Y %I_%M%p")) + ".tar")

    # loading networks
    def load_net(self, p_net, t_net, multi_res, p_net_controller):
        # note that t_net weights will be the same as p_net
        if not multi_res:
            file, file_name = self.choose_file(".pt", self.models_path)
            file.close()
            checkpoint = torch.load(os.path.join(self.models_path, file_name), map_location=p_net.models[0].device)
            if "weights" in checkpoint:
                p_net.models[0].load_state_dict((checkpoint["weights"]))
                t_net.models[0].load_state_dict((checkpoint["weights"]))
            else:
                p_net.models[0].load_state_dict(checkpoint)
                t_net.models[0].load_state_dict(checkpoint)
        else:
            for i in range(self.box_mode.count("_") + 1):
                print("Choose model to load for " + self.box_mode[i*2])
                file, file_name = self.choose_file(".pt", self.models_path)
                file.close()

                checkpoint = torch.load(os.path.join(self.models_path, file_name), map_location=p_net.models[0].device)
                if "weights" in checkpoint:
                    p_net.models[i].load_state_dict((checkpoint["weights"]))
                    t_net.models[i].load_state_dict((checkpoint["weights"]))
                else:
                    p_net.models[i].load_state_dict(checkpoint)
                    t_net.models[i].load_state_dict(checkpoint)

        return p_net, t_net, p_net_controller

    def load_q_net(self, p_net_controller, p_net, t_net):
        print("Choose Q-net:")
        file, file_name = self.choose_file(".tar", self.full_models_path)
        file.close()
        checkpoint = torch.load(os.path.join(self.full_models_path, file_name))
        p_net_controller.optimizer.load_state_dict((checkpoint["optim"]))
        p_net.load_state_dict(checkpoint["weights"])
        t_net.load_state_dict(checkpoint["weights"])
        return p_net_controller, p_net, t_net

    # to ensure random splits are seeded (same split each time) we could load from this file
    def make_pat_file(self, train_pat_nums, val_pat_nums, test_pat_nums, train_slices, val_slices, test_slices):
        file_df = pd.DataFrame(
            index=["num of pats", "% pats", "pat list", "num of slices", "% slices", "slice list"],
            columns=["Train", "Validation", "Test"])
        file_df.at["num of pats", "Train"] = len(train_pat_nums)
        file_df.at["num of pats", "Validation"] = len(val_pat_nums)
        file_df.at["num of pats", "Test"] = len(test_pat_nums)
        total_pats_len = file_df.loc["num of pats"].sum(axis=0)
        file_df.loc["% pats"] = 100 * file_df.loc["num of pats"] / total_pats_len
        file_df.at["num of slices", "Train"] = len(train_slices)
        file_df.at["num of slices", "Validation"] = len(val_slices)
        file_df.at["num of slices", "Test"] = len(test_slices)
        total_slices_len = file_df.loc["num of slices"].sum(axis=0)
        file_df.loc["% slices"] = 100 * file_df.loc["num of slices"] / total_slices_len
        file_df.loc["pat list"] = [train_pat_nums, val_pat_nums, test_pat_nums]
        file_df.loc["slice list"] = [train_slices, val_slices, test_slices]
        time = str(datetime.datetime.now().strftime("%b %d, %Y %I_%M%p"))
        name = "train_val_test pats " + time + ".csv"
        file_df.to_csv(self.data_splits_path + name)


