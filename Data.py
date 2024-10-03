import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Slice2d import Slice2d
from PIL import Image
from matplotlib import patches
import torch

# makes slices
# splits pats
# makes input data for nets
class Data:
    def __init__(self, resized_dims):
        self.resized_dims = resized_dims

    # counts number of slices for each pat
    # remember slices that healthy or with 2+ boxes have been removed
    # randomly splits slices into train and test set with ~ 75:25
    def split_pat_data(self, fp, train_frac, val_frac, dims, resized_dims, seed, gen_file, path_abr, scan_type):
        df = fp.process_box_file(dims, resized_dims, path_abr, scan_type)
        num_slices = len(df)
        # print(num_slices)
        im_names = list(df.index)
        pat_slice_count = {}

        # counts slices per pat
        for name in im_names:
            pat_num = int(name[1:5])
            if pat_num in pat_slice_count:
                pat_slice_count[pat_num] += 1
            else:
                pat_slice_count[pat_num] = 1

        # this part ensures a pat will not be both in training and testing
        # iteratively adds more pats to training until at least 75% is reached
        train_pat_nums = set()  # the pat numbers
        train_slice_count = 0  # total number of training slices

        val_pat_nums = set()
        val_slice_count = 0

        buf_percent = 5
        np.random.seed(seed)
        while train_slice_count < ((train_frac * num_slices) - buf_percent):
            pats_not_selected = set(pat_slice_count.keys()) - train_pat_nums
            pat_num = np.random.choice(list(pats_not_selected))
            train_pat_nums.add(pat_num)
            train_slice_count += pat_slice_count[pat_num]

        while val_slice_count < ((val_frac * num_slices) - buf_percent):
            pats_not_selected = set(pat_slice_count.keys()) - train_pat_nums - val_pat_nums
            pat_num = np.random.choice(list(pats_not_selected))
            val_pat_nums.add(pat_num)
            val_slice_count += pat_slice_count[pat_num]

        # train_pat_nums - pat numbers for training
        # pat numbers for testing
        test_pat_nums = set(pat_slice_count.keys()) - train_pat_nums - val_pat_nums
        #self.visualize_bounding_box(df)
        train_slices, val_slices, test_slices, center = self.make_slice_data(im_names,
                                                                             df,
                                                                             train_pat_nums,
                                                                             val_pat_nums,
                                                                             test_pat_nums)

        line1 = "Number of training pats {0}\nNumber of validation pats {1}\nNumber of testing pats {2}\n". \
            format(len(train_pat_nums), len(val_pat_nums), len(test_pat_nums))

        line2 = "\nNumber of training slices {0}\nNumber of validation slices {1}\nNumber of testing slices {2}". \
            format(len(train_slices), len(val_slices), len(test_slices))
        print(line1 + line2)
        if gen_file:
            fp.make_pat_file(list(train_pat_nums), list(val_pat_nums), list(test_pat_nums), train_slices.keys(),
                             val_slices.keys(), test_slices.keys())


        return train_pat_nums, val_pat_nums, test_pat_nums, train_slices, val_slices, test_slices, center

    # makes slice instances of all training and testing slices
    @staticmethod
    def make_slice_data(im_names, df, train_pat_nums, val_pat_nums, test_pat_nums):
        train_slices = {}
        val_slices = {}
        test_slices = {}
        x_sum = 0
        y_sum = 0
        for name in im_names:
            if int(name[1:5]) in train_pat_nums:
                for i, modifier in enumerate(["r0 NORM.pt", "r90 NORM.pt", "r180 NORM.pt", "r270 NORM.pt"]):
                    mod_name = name.replace("RES.jpg", modifier)
                    train_slices[mod_name] = (Slice2d(mod_name, df.loc[name][i]))
                    x_sum += train_slices[mod_name].box.center[0]
                    y_sum += train_slices[mod_name].box.center[1]
            elif int(name[1:5]) in val_pat_nums:
                for i, modifier in enumerate(["r0 NORM.pt", "r90 NORM.pt", "r180 NORM.pt", "r270 NORM.pt"]):
                    mod_name = name.replace("RES.jpg", modifier)
                    val_slices[mod_name] = (Slice2d(mod_name, df.loc[name][i]))
            elif int(name[1:5]) in test_pat_nums:
                for i, modifier in enumerate(["r0 NORM.pt", "r90 NORM.pt", "r180 NORM.pt", "r270 NORM.pt"]):
                    mod_name = name.replace("RES.jpg", modifier)
                    test_slices[mod_name] = (Slice2d(mod_name, df.loc[name][i]))
            else:
                assert False, "Warning slice allocation incorrect - see make_slice_data method in Data file!"

        x_cent = int(round(x_sum / len(train_slices)))
        y_cent = int(round(y_sum / len(train_slices)))
        return train_slices, val_slices, test_slices, [x_cent, y_cent]

    # this method makes the dataset to train the policy net
    @staticmethod
    def make_net_training_input_data(batch_size, transitions, dic, to_calc_targets):
        # sample a batch size of random transitions from the memory, gets each of their characteristics and then unzips
        sel_trans = random.sample(transitions, batch_size)
        info = [(sel_tran.r, dic[sel_tran.a], sel_tran.a, sel_tran.is_terminal, sel_tran.file_states, sel_tran.target)
                for sel_tran in
                sel_trans]
        rs, actions, named_actions, terminals, states, targets = zip(*info)

        # makes q dataset which is current state and q' dataset which is s'
        # the dropping ensures s' and s are correct (s' needs the newest box, while s does not... etc.)
        q_dataset = pd.DataFrame(zip(*states)).transpose()
        q_dataset.drop(q_dataset.columns[-1], axis=1, inplace=True)
        q_prime_dataset = None
        if not to_calc_targets:
            q_prime_dataset = pd.DataFrame(zip(*states)).transpose()
            q_prime_dataset.drop(q_dataset.columns[0], axis=1, inplace=True)
        return q_dataset, q_prime_dataset, actions, rs, terminals, named_actions, targets  # named_actions are ["U", "D", ...]

    @staticmethod
    def visualize_bounding_box(df):
        # just edema 512 have been made on IR and t1 fracture for 864
        for name in df.index:
            box = df[name]
            im = Image.open("data/scans/" + name)
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(name)
            ax.imshow(im, cmap="Greys")
            ground_truth = patches.Rectangle(box.tl,
                                             box.right_most - box.left_most,
                                             box.bottom_most - box.top_most,
                                             fill=False,
                                             linewidth=0.3,
                                             color="orangered")

            ax.add_patch(ground_truth)
            plt.savefig("/Users/robinson/Documents/Career stuff/DePaul University/MRI Fractures Research/ground_truths/"
                        + name.split("/")[-1])
            #plt.show(block=True)
            plt.close()
            # input("Press RETURN to continue")
        exit()
