import numpy as np
import torch
from torchvision.ops import box_iou
from torch import Tensor
from Box import Box
import random


class Environ:
    def __init__(self, trans_dist, scale_dist, iou_thresh, euclid_thresh, data_instance, max_train_states,
                 max_infer_states, delta_alpha, end_alpha, reward_metric, start_fraction, train_center, box_size,
                 pos_reward, neg_reward):
        self.data = data_instance
        # default should be 0.1 * dimension of current box
        self.trans_dist = trans_dist  # how many pixels box moves during translation,
        # default should be 0.1 * dimension of current box
        self.scale_dist = scale_dist  # how many pixels box grows or shrinks when making taller/ wider/ zooming in.
        self.iou_thresh = iou_thresh  # when the intersection over union is high enough, stop training on this slice
        self.euclid_thresh = euclid_thresh
        self.iou = 0
        self.euclid = 0
        self.final_box = None
        self.start_fraction = start_fraction
        self.max_train_states = max_train_states
        self.max_infer_states = max_infer_states
        self.train_center = train_center
        self.delta_alpha = delta_alpha
        self.end_alpha = end_alpha
        self.reward_metric = reward_metric
        self.box_size = box_size
        self.display = ""
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward

    # sees where the bounding box will hypothetically end up if we carry out the selected action

    def move_box(self, sel_axn, temp_box):
        top_left = temp_box.tl
        bot_right = temp_box.br
        h = bot_right[1] - top_left[1]
        w = bot_right[0] - top_left[0]

        # axn 0
        if sel_axn == "U":
            factor = self.move_box_helper(self.trans_dist * h, sel_axn)
            tl = [top_left[0], top_left[1] - factor]
            br = [bot_right[0], bot_right[1] - factor]

        # axn 1
        elif sel_axn == "D":
            factor = self.move_box_helper(self.trans_dist * h, sel_axn)
            tl = [top_left[0], top_left[1] + factor]
            br = [bot_right[0], bot_right[1] + factor]

        # axn 2
        elif sel_axn == "L":
            factor = self.move_box_helper(self.trans_dist * w, sel_axn)
            tl = [top_left[0] - factor, top_left[1]]
            br = [bot_right[0] - factor, bot_right[1]]

        # axn 3
        elif sel_axn == "R":
            factor = self.move_box_helper(self.trans_dist * w, sel_axn)
            tl = [top_left[0] + factor, top_left[1]]
            br = [bot_right[0] + factor, bot_right[1]]

        # axn 4
        elif sel_axn == "T":
            factor = self.move_box_helper(self.scale_dist * h, sel_axn)
            tl = [top_left[0], top_left[1] - factor / 2]
            br = [bot_right[0], bot_right[1] + factor / 2]

        # axn 5
        elif sel_axn == "W":
            factor = self.move_box_helper(self.scale_dist * w, sel_axn)
            tl = [top_left[0] - factor / 2, top_left[1]]
            br = [bot_right[0] + factor / 2, bot_right[1]]

        # axn 6
        elif sel_axn == "O":
            aspect_ratio = w / h
            h_factor = self.scale_dist * h
            w_factor = self.move_box_helper(h_factor * aspect_ratio, sel_axn)
            h_factor = self.move_box_helper(h_factor, sel_axn)
            tl = [top_left[0] - w_factor / 2, top_left[1] - h_factor / 2]
            br = [bot_right[0] + w_factor / 2, bot_right[1] + h_factor / 2]

        # axn 7
        elif sel_axn == "S":
            factor = self.move_box_helper(self.scale_dist * h, sel_axn)
            tl = [top_left[0], top_left[1] + factor / 2]
            br = [bot_right[0], bot_right[1] - factor / 2]

        # axn 8
        elif sel_axn == "N":
            factor = self.move_box_helper(self.scale_dist * w, sel_axn)
            tl = [top_left[0] + factor / 2, top_left[1]]
            br = [bot_right[0] - factor / 2, bot_right[1]]

        # axn 9 I
        else:
            aspect_ratio = w / h
            h_factor = self.scale_dist * h
            w_factor = self.move_box_helper(h_factor * aspect_ratio, sel_axn)
            h_factor = self.move_box_helper(h_factor, sel_axn)
            tl = [top_left[0] + w_factor / 2, top_left[1] + h_factor / 2]
            br = [bot_right[0] - w_factor / 2, bot_right[1] - h_factor / 2]

        return Box(list(np.round(np.array(tl))), list(np.round(np.array(br))))

    # makes sure that moves actually have an effect, minimum moving distance of 1 pixel...
    @staticmethod
    def move_box_helper(factor, axn):
        if axn in ["U", "D", "L", "R"]:
            if factor < 1:
                return 1
        else:
            # axn in ["T", "W", "N", "S", "O", "I"]:
            if factor < 2:
                return 2
        return factor

    # calculating IoU and Euclidean distance between box centers (ground truth and proposed boxes)
    def iou_and_euclid(self, true_box, current_box, to_set):
        # uses pytorch function to compute IoU and compares it to previous value
        with torch.no_grad():
            new_iou = (box_iou(Tensor([[current_box.top_most, current_box.left_most,
                                        current_box.bottom_most, current_box.right_most]]),
                               Tensor([[true_box.top_most, true_box.left_most, true_box.bottom_most,
                                        true_box.right_most]]))).item()
        quantity = (true_box.center[0] - current_box.center[0]) ** 2 + (true_box.center[1] - current_box.center[1]) ** 2
        # calculates Euclidean distance of box centers
        new_euclid = quantity ** 0.5
        if to_set:
            self.iou = new_iou
            self.euclid = new_euclid
            return
        return new_iou, new_euclid

    # is_exploring is true when previewing reward for guided exploration
    def get_reward(self, true_box, current_box, is_exploring):
        reward = None
        new_iou, new_euclid = self.iou_and_euclid(true_box, current_box, False)
        improvement_margin = 0.00001  # small error
        if self.reward_metric == "BOTH":
            if new_iou > (self.iou + improvement_margin):
                reward = self.pos_reward
            elif new_iou < (self.iou - improvement_margin):
                reward = self.neg_reward
            else:
                if new_euclid < (self.euclid - improvement_margin):
                    reward = self.pos_reward
                else:
                    reward = self.neg_reward

        elif self.reward_metric == "IOU":
            if new_iou > (self.iou + improvement_margin):
                reward = self.pos_reward
            else:
                reward = self.neg_reward

        elif self.reward_metric == "EUCLID":
            if new_euclid < (self.euclid - improvement_margin):
                reward = self.pos_reward
            else:
                reward = self.neg_reward

        # when previewing reward for guided exploration don't change the iou or euclid - but do if not the case
        if not is_exploring:
            # updating euclid and iou
            self.euclid = new_euclid
            self.iou = new_iou
        return reward

    def reward_wrapper(self, true_box, current_box, is_exploring, always_neg_reward):
        reward = self.get_reward(true_box, current_box, is_exploring)
        return self.neg_reward if always_neg_reward else reward

    # checking for terminal state in training or inference
    def check_terminal_state(self, is_training, state_num, current_box, hits, oscillate, is_pretraining,
                             pre_transition_steps):
        q_is_0 = False
        # pretraining
        if is_pretraining:
            if state_num >= pre_transition_steps:
                self.final_box = current_box
                self.display = ""
                return True, q_is_0
            else:
                return False, q_is_0
        # inference mode
        if not is_training:
            if self.max_infer_states <= state_num:
                self.display = "Terminal state ({0}) reached with iou of {1:.3f} and euclid of {2:.3f}".format([current_box.tl,
                                                                                                       current_box.br],
                                                                                                      self.iou,
                                                                                                      self.euclid)
                self.final_box = current_box
                return True, q_is_0
            else:
                return self.check_terminal_state_helper(hits, oscillate, current_box), q_is_0
        else:
            # training
            if ((self.iou > self.iou_thresh) and self.reward_metric != "EUCLID") or \
                    ((self.euclid < self.euclid_thresh) and self.reward_metric == "EUCLID") or \
                    (state_num >= self.max_train_states):
                self.final_box = current_box
                if ((self.iou > self.iou_thresh) and self.reward_metric != "EUCLID") or \
                        ((self.euclid < self.euclid_thresh) and self.reward_metric == "EUCLID"):
                    q_is_0 = True
                self.display = "Terminal state ({0}) reached with iou of {1:.3f} and euclid of {2:.3f}".format([current_box.tl,
                                                                                                       current_box.br],
                                                                                                      self.iou,
                                                                                                      self.euclid)
                self.final_box = current_box
                return True, q_is_0
            return False, q_is_0

    # when in inference mode
    # looks at box buffer (essentially positions of bounding box)
    # and when positions 0, 2, 4 are the same
    # assumes oscillation and says terminal state reached
    # this isn't exactly oscillation
    # we need to also check that positions 1 and 3 are the same
    # and that the q values are relatively lower
    # low q values means closer to the final state
    def check_terminal_state_helper(self, hits, oscillate, current_box):
        return False
        if -1 in last_x_axns:
            return False
        _, counts = np.unique(np.array(last_x_axns), return_counts=True, axis=0)
        to_return = False
        if oscillate:
            last_x_axns = np.array(last_x_axns)
            for i in range(2, int(last_x_axns.shape[0] / 2) + 1):
                remainder = last_x_axns.shape[0] % i
                to_split = np.copy(last_x_axns)[remainder:]
                splits = np.split(to_split, i)
                if np.all(splits[0] == splits):
                    to_return = True
                    break

        elif np.max(counts) >= hits:
            to_return = True

        if to_return:
            self.final_box = current_box
            print("Terminal state ({0}) reached with iou of {1:.3f} and euclid of {2:.3f}".format([current_box.tl,
                                                                                                   current_box.br],
                                                                                                  self.iou,
                                                                                                  self.euclid))
            return True
        return False

    # picks two random points and makes a box out of this
    # to randomly start the agent at
    def start_box(self, start_mode, fp, slice_name):
        tl, br = None, None
        x, y = self.data.resized_dims
        # for 'standard' start where agent sees entire slice
        if start_mode == "ORIGINAL":
            x_start = int(self.start_fraction * x)
            y_start = int(self.start_fraction * y)
            tl = [x_start, y_start]
            br = [x - x_start - 1, y - y_start - 1]

        elif start_mode == "RAND":
            pt0x, pt0y, pt1x, pt1y = 0, 0, 0, 0
            while pt0x == pt1x:
                pt0x = random.randint(0, x - 1)
                pt1x = random.randint(0, x - 1)
            while pt0y == pt1y:
                pt0y = random.randint(0, y - 1)
                pt1y = random.randint(0, y - 1)
            t = min(pt0y, pt1y)
            l = min(pt0x, pt1x)
            b = max(pt0y, pt1y)
            r = max(pt0x, pt1x)
            tl = [l, t]
            br = [r, b]

        elif start_mode == "SCALING":
            # scaling start mode
            df = fp.read_start_file()
            str_tl = (df.loc[slice_name, "tl"]).replace("[", "").replace("]", "").replace(" ", "").split(",")
            tl = [float(str_tl[0]), float(str_tl[1])]
            str_br = (df.loc[slice_name, "br"]).replace("[", "").replace("]", "").replace(" ", "").split(",")
            br = [float(str_br[0]), float(str_br[1])]

        elif start_mode.startswith("FIX"):
            length = self.box_size
            if start_mode.split("_")[1] == "MEAN":
                # training centered
                tl = [self.train_center[0] - length // 2, self.train_center[1] - length // 2]
                br = [self.train_center[0] + length // 2, self.train_center[1] + length // 2]

            elif start_mode.split("_")[1] == "RAND":
                # random start
                tl_x = random.randint(0, x - length - 1)
                tl_y = random.randint(0, y - length - 1)
                tl = [tl_x, tl_y]
                br = [tl_x + length, tl_y + length]

            elif start_mode.split("_")[1] == "CENTER":
                # centered...
                tl = [x // 2 - length // 2, y // 2 - length // 2]
                br = [x // 2 + length // 2, y // 2 + length // 2]

            else:
                # designated start position
                pos = list(start_mode.split("_")[1].split(","))
                tl = [int(pos[0]), int(pos[1])]
                br = [int(pos[2]), int(pos[3])]

        return Box(tl, br), Box(tl, br), tl, br
