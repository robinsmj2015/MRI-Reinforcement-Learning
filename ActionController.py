import random
import numpy as np
import torch


class ActionController:
    def __init__(self, epsilon, axns, guided_exploration, visualizer, to_help, neg_reward):
        self.epsilon = epsilon  # (guided) exploration in percent of actions
        self.guided_exploration = guided_exploration
        self.sel_axn = None  # the action the agent will take
        self.temp_box = None  # where agent is looking to move to (different possibly than where it actually moves to)
        self.attempted_actions = set()  # actions tried during an exploration sequence
        # actions [up, down, left, right, make taller, make wider, zoom out, make shorter, make more narrow, zoom in]
        self.axns = axns
        self.axn_num_dict = {}  # converts action string to number
        self.q_summ = 0
        self.num_q = 0
        for i, axn in enumerate(self.axns):
            self.axn_num_dict[axn] = i
        self.num_axns = len(self.axns)  # the number of possible actions
        self.agent_is_stuck = False  # for inference mode
        self.to_help = to_help  # scaling guided exploration
        self.visualizer = visualizer
        self.neg_reward = neg_reward

    # selects next action for agent
    def pick_action(self, box, fp, file_state_buff, p_net, environ, slice2d_box, state_num, track_num, resized_dims,
                    is_pretraining, max_train_states):
        qs = None
        rand_int = random.randint(1, 100)
        if is_pretraining:
            self.sel_axn = random.choice(self.axns)

        # initiating exploration sequence
        elif rand_int <= self.epsilon:
            boost = state_num * (1 - self.guided_exploration) / max_train_states if self.to_help else 0
            self.sel_axn = random.choice(self.axns)
            is_blocked = True if rand_int < (self.guided_exploration + boost) * self.epsilon else False
            i = 0
            # guided exploration only
            while is_blocked and i < self.num_axns:
                self.sel_axn = random.choice(list(set(self.axns) - self.attempted_actions))
                self.temp_box = environ.move_box(self.sel_axn, box)
                is_blocked = self.action_outside_slice2d(resized_dims) or (
                            environ.get_reward(slice2d_box, self.temp_box, is_exploring=True) == self.neg_reward)
                self.attempted_actions.add(self.sel_axn)
                i += 1

        else:
            # BEST - uses greedy choice, resorts to second/ third/ etc. best action if action moves outside slice
            # gets q values, sorts them
            qs = self.get_qs(p_net, fp, file_state_buff[1])
            q = None
            np_qs = np.array(qs[0]).copy()
            sorted_np_qs = np.sort(np_qs)
            is_blocked = True
            i = p_net.num_axns - 1
            while is_blocked:
                q = sorted_np_qs[i]
                best_indices = np.where(np_qs == q)[0]
                self.sel_axn = self.axns[np.random.choice(best_indices)]
                self.temp_box = environ.move_box(self.sel_axn, box)
                is_blocked = self.action_outside_slice2d(resized_dims)
                i -= 1
                assert i != -2, "ERROR: all actions blocked!"
            self.q_summ += q
            self.num_q += 1

        # gets location of the prospective bounding box
        self.temp_box = environ.move_box(self.sel_axn, box)
        self.attempted_actions = set()
        state_num += 1
        track_num += 1
        return qs, self.temp_box, state_num, track_num

    # returns true if temp bounding box is outside of image or the box gets smooshed
    def action_outside_slice2d(self, resized_dims):
        x, y = resized_dims
        # outside box?
        past_tl = self.temp_box.top_most < 0 or self.temp_box.left_most < 0
        past_br = self.temp_box.right_most >= x or self.temp_box.bottom_most >= y
        # smooshed?
        h_error = self.temp_box.bottom_most <= self.temp_box.top_most
        w_error = self.temp_box.right_most <= self.temp_box.left_most
        return past_tl or past_br or h_error or w_error

    # get q values from policy network to decide on action
    @staticmethod
    def get_qs(p_net, fp, file):
        inputs = torch.load(fp.transition_path + file)
        with torch.no_grad():
            p_net.eval()
            qs = p_net.forward(torch.unsqueeze(inputs, dim=0)).to(torch.device("cpu"))
        return qs

