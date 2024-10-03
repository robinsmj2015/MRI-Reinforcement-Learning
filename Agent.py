import random
from collections import deque
from torch import optim
from torch.nn import MSELoss
from torch.nn import HuberLoss
from Message import Message
from Environ import Environ
from TransitionController import TransitionController
from Box import Box
from Net import Net
from NetController import NetController
from Visualizer import Visualizer
from ActionController import ActionController
import time
import numpy as np


class Agent:
    def __init__(self, name, data_instance, file_processing, epsilon, delta_epsilon, trans_dist, scale_dist, iou_thresh,
                 euclid_thresh, max_slices, loss, lr, batch_size, t_net_update_time, soft_update_tau,
                 max_transitions, graphics_mode, to_load_model, max_train_states, max_infer_states, discount,
                 stopping_criteria, start_mode, scaling_factor, axns, end_epsilon, delta_alpha, end_alpha,
                 reward_metric, start_fraction, pre_transition_steps, pre_transition_epochs, guided_exploration,
                 train_center, box_size, to_separate_q_prime, to_run, multi_res, to_help, best_gpu_nums, pos_reward,
                 neg_reward, to_calc_targets, version, to_visualize, always_neg_reward):
        self.name = name  # agent's name
        self.qs = None  # q values
        self.start_epsilon = epsilon  # local copy
        self.message = Message()
        self.tc = TransitionController(max_transitions, axns)
        self.visualizer = Visualizer(file_processing)  # visualizing
        self.ac = ActionController(epsilon, axns, guided_exploration, self.visualizer, to_help, neg_reward)
        self.fp = file_processing  # file processing
        self.data = data_instance  # data instance
        self.delta_epsilon = delta_epsilon
        self.box_memory_buff = []  # stores the last 2 boxes
        self.box_memory_buff_size = 2
        self.file_state_buff = []  # list of files (cropped slices to bounding box) in box memory buff
        self.reward = 0  # reward agent gets - either 1 or -1
        self.is_terminal_state = False  # training episode (on a slice is complete)
        self.slice2d = None  # the image slice instance that the agent is currently exploring
        self.box = None  # box instance, where bounding box is currently
        self.num_train_slices = None
        self.num_infer_slices = None
        self.total_episodes = 0
        self.is_pretraining = None
        self.loss = loss
        self.lr = lr  # learning rate
        self.batch_size = batch_size
        self.environ = Environ(trans_dist, scale_dist, iou_thresh, euclid_thresh, data_instance, max_train_states,
                               max_infer_states, delta_alpha, end_alpha, reward_metric, start_fraction, train_center,
                               box_size, pos_reward, neg_reward)  # environ
        self.pre_transition_steps = pre_transition_steps
        self.pre_transition_epochs = pre_transition_epochs
        self.start_trans_dist = trans_dist
        self.start_scale_dist = scale_dist
        self.t_net_update_time = t_net_update_time  # how often (in time-steps of agent) to update target net
        self.soft_update_tau = soft_update_tau  # tau value for soft update
        self.ep_num = 0  # which episode ... needed to keep track of transition
        self.state_num = 0  # which state in episode... needed to track transition
        self.track_num = 0  # number of transitions completed by the agent
        self.graphics_mode = graphics_mode  # displaying of graphics: full, save or text
        self.start_mode = start_mode  # random starts, traditional (whole slice) or scaling
        self.to_load_model = to_load_model  # load past model
        self.max_infer_states = max_infer_states  # max number of inference states allowed
        self.start_pos = None
        self.p_net = None  # policy network
        self.t_net = None  # target network
        self.p_net_controller = None  # policy network controller
        self.t_net_controller = None  # target network controller
        self.is_training = None  # training mode or not (inference mode)
        self.old_track_num = 0  # for figuring out how many states in an episode
        self.steps = []  # tracking num states per episode
        self.times = []  # time per episode
        self.ious = []  # iou for each episode
        self.euclids = []  # euclidean for each episode
        self.total_epochs = 0
        self.total_train_epochs = 0
        self.end_epsilon = end_epsilon
        self.max_slices = max_slices  # which slices to use
        self.stopping_criteria = stopping_criteria  # for stopping inference mode: [hits, out of x], x refers to above
        self.scaling_factor = scaling_factor
        self.total_start_time = 0
        self.episodes_in_full_run = None
        self.avg_q = 0
        self.multi_res = multi_res
        self.best_gpu_nums = best_gpu_nums
        self.to_calc_targets = to_calc_targets
        self.always_neg_reward = always_neg_reward
        self.last_four = []
        if to_run:
            self.initialize_nets(discount, lr, to_separate_q_prime, version, to_visualize)  # loads networks if relevant

    # the main training loop of agent
    def train_or_infer(self, is_training, num_epochs, slices2d, start_mode, is_pretraining):
        # initialization
        to_iterate_over = self.start_new_train_or_infer(is_training, start_mode, is_pretraining, slices2d)
        # iterate by epoch
        for epoch_num in range(num_epochs):
            self.start_new_epoch()
            # iterate by slice (episode)
            for key in to_iterate_over[: self.max_slices]:
                start_time = self.start_new_slice2d(slices2d, key)
                # below sets iou & euclid initially
                self.environ.iou_and_euclid(self.slice2d.box, self.box, to_set=True)
                # fpr displaying visuals and logs ----------------------------------------------------------------------
                if not self.is_training:
                    display, name = self.visualizer.step_wise_visualize(box=self.box,
                                                                        slice2d=self.slice2d,
                                                                        iou=self.environ.iou,
                                                                        reward=self.reward,
                                                                        sel_axn=self.ac.sel_axn,
                                                                        qs=self.qs,
                                                                        ep_num=self.ep_num,
                                                                        state_num=self.state_num,
                                                                        euclid=self.environ.euclid,
                                                                        key=key,
                                                                        epoch=epoch_num,
                                                                        track_num=self.track_num,
                                                                        axns=self.ac.axns,
                                                                        start_mode=self.start_mode,
                                                                        box_mode=self.fp.box_mode,
                                                                        epsilon=self.ac.epsilon,
                                                                        total_epochs=self.total_epochs)

                    self.fp.create_log_file(display, name, to_create_new_file=True, is_training=self.is_training)
                # agent moving in slice until terminal state is reached
                while not self.is_terminal_state:
                    # choose action
                    self.qs, temp_box, self.state_num, self.track_num = self.ac.pick_action(self.box,
                                                                                            self.fp,
                                                                                            self.file_state_buff,
                                                                                            self.p_net,
                                                                                            self.environ,
                                                                                            self.slice2d.box,
                                                                                            self.state_num,
                                                                                            self.track_num,
                                                                                            self.data.resized_dims,
                                                                                            self.is_pretraining,
                                                                                            self.environ.max_train_states)
                    # box moves (as long as it is not trying to leave the slice or smoosh itself)
                    self.box = temp_box if not self.ac.action_outside_slice2d(self.data.resized_dims) else self.box
                    # update buffers... file and box and last x positions (after each state) - state update essentially
                    self.reset_buffs_and_actions()
                    # get rewards
                    self.reward = self.environ.reward_wrapper(self.slice2d.box,
                                                              self.box,
                                                              is_exploring=False,
                                                              always_neg_reward=self.always_neg_reward)
                    # checking for terminal state
                    self.is_terminal_state, q_is_0 = self.environ.check_terminal_state(is_training=self.is_training,
                                                                                       state_num=self.state_num,
                                                                                       current_box=self.box,
                                                                                       hits=self.stopping_criteria[0],
                                                                                       oscillate=bool(
                                                                                           self.stopping_criteria[2]),
                                                                                       is_pretraining=self.is_pretraining,
                                                                                       pre_transition_steps=self.pre_transition_steps)

                    if not self.is_training:
                        # update logs and graphics ---------------------------------------------------------------------
                        display, name = self.visualizer.step_wise_visualize(box=self.box,
                                                                            slice2d=self.slice2d,
                                                                            iou=self.environ.iou,
                                                                            reward=self.reward,
                                                                            sel_axn=self.ac.sel_axn,
                                                                            qs=self.qs,
                                                                            ep_num=self.ep_num,
                                                                            state_num=self.state_num,
                                                                            euclid=self.environ.euclid,
                                                                            key=key,
                                                                            epoch=epoch_num,
                                                                            track_num=self.track_num,
                                                                            axns=self.ac.axns,
                                                                            start_mode=self.start_mode,
                                                                            box_mode=self.fp.box_mode,
                                                                            epsilon=self.ac.epsilon,
                                                                            total_epochs=self.total_epochs)

                        self.fp.create_log_file(display, name, to_create_new_file=False, is_training=self.is_training)
                        self.fp.create_transition_file(self.data.resized_dims, "I", self.box, self.slice2d.name,
                                                       self.track_num)

                    else:

                        # updating transitions -------------------------------------------------------------------------
                        self.tc.manage_transitions(slice2d=self.slice2d,
                                                   file_state_buff=self.file_state_buff,
                                                   box_memory_buff=self.box_memory_buff,
                                                   reward=self.reward,
                                                   ep_num=self.ep_num,
                                                   state_num=self.state_num,
                                                   track_num=self.track_num,
                                                   fp=self.fp,
                                                   ac=self.ac,
                                                   resized_dims=self.data.resized_dims,
                                                   is_terminal=q_is_0,
                                                   pos_reward=self.environ.pos_reward,
                                                   discount=self.p_net_controller.discount,
                                                   to_calc_targets=self.to_calc_targets,
                                                   euclid_thresh=self.environ.euclid_thresh,
                                                   trans_dist=self.environ.trans_dist)

                        # update networks ------------------------------------------------------------------------------
                        # policy network update
                        if not self.is_pretraining:
                            self.p_net_controller.update_net(self.data,
                                                             self.tc.transitions,
                                                             self.ac.axn_num_dict,
                                                             self.t_net,
                                                             self.fp,
                                                             self.environ.pos_reward,
                                                             self.to_calc_targets)
                            # target network update
                            if self.track_num % self.t_net_update_time == 0:
                                self.t_net_controller.soft_update(self.soft_update_tau, self.p_net)

                # after each slice2d -----------------------------------------------------------------------------------
                self.visualizer.visualize_q_map(slices2d[key], self.data.resized_dims, self.fp, self.p_net, self.ac,
                                                self.total_epochs, self.environ.box_size,
                                                self.p_net_controller.chosen_transitions, self.tc.categories,
                                                self.environ.final_box, self.start_pos, self.is_training, self.ep_num,
                                                self.environ.trans_dist)

                # if not (self.test_weight_init is None):
                # print(np.all(np.array(self.p_net.fc_hidden.weight.flatten().detach()) == self.test_weight_init))
                # self.test_weight_init = np.array(self.p_net.fc_hidden.weight.flatten().detach())
                # should be True: self.p_net.model.layer4[0].conv1.weight.flatten() if np.all
                # should be False np.all self.p_net.fc_hidden.weight.flatten().detach()
                self.end_episode(start_time, epoch_num)
            # after each epoch -----------------------------------------------------------------------------------------
            self.end_epoch()

    # calls the main training/ inference loop
    def run_train_and_inference(self, mega_epochs, train_epochs, infer_epochs, train_slices, infer_slices, start_mode,
                                to_save_models, to_clear_starts, to_clear_models, to_clear_transitions):
        self.num_train_slices = len(train_slices)
        self.num_infer_slices = len(infer_slices)
        self.episodes_in_full_run = self.fp.epochs[0] * (self.fp.epochs[1] * min(self.num_train_slices, self.max_slices)
                                                         + self.fp.epochs[2] * min(self.num_infer_slices,
                                                                                   self.max_slices))
        self.fp.clear_folders(to_clear_starts, to_clear_models, to_clear_transitions)  # clears files
        # pre-transition memories
        self.train_or_infer(is_training=True, num_epochs=self.pre_transition_epochs,
                            slices2d=train_slices,
                            start_mode=start_mode,
                            is_pretraining=True)
        self.total_start_time = time.time()
        for i in range(mega_epochs):
            # inference
            self.train_or_infer(is_training=False,
                                num_epochs=infer_epochs,
                                slices2d=infer_slices,
                                start_mode=start_mode,
                                is_pretraining=False)
            # training
            self.train_or_infer(is_training=True,
                                num_epochs=train_epochs,
                                slices2d=train_slices,
                                start_mode=start_mode,
                                is_pretraining=False)
            if to_save_models:
                self.fp.save_net(self.p_net, self.p_net_controller)

        if self.scaling_factor != 1:
            self.fp.create_start_file(self.max_slices, self.scaling_factor, train_slices, infer_slices)

    ''' ============================================ INITIALIZATION ================================================ '''

    # when starting training or validation
    def start_new_train_or_infer(self, is_training, start_mode, is_pretraining, slices2d):
        self.is_pretraining = is_pretraining
        self.is_training = is_training
        self.start_mode = start_mode
        to_iterate_over = list(slices2d.keys())
        if self.is_training:
            random.shuffle(to_iterate_over)
        return to_iterate_over

    # at the start of each episode
    def start_new_slice2d(self, slices2d, key):
        start_time = time.time()
        # current slice and current slice name
        self.slice2d = slices2d[key]
        self.fp.slice_file_name = self.slice2d.name
        # to get the starting position of the box
        self.box, self.start_pos, tl, br = self.environ.start_box(self.start_mode, self.fp, self.slice2d.name)

        # resets file buffer
        self.file_state_buff = [self.fp.create_transition_file(resized_dims=self.data.resized_dims,
                                                               code="T" if self.is_training else "I",
                                                               box=self.box,
                                                               master_file=self.slice2d.name,
                                                               track_num=self.track_num)] * self.box_memory_buff_size
        # resets box buffer (no transitions across episodes)
        self.box_memory_buff = []
        for i in range(self.box_memory_buff_size):
            self.box_memory_buff.append(Box(tl, br))
        return start_time

    # Both policy and target nets, and their controllers
    def initialize_nets(self, discount, lr, to_separate_q_prime, version, to_visualize):
        num_channels = self.fp.box_mode.count("_") + 1
        self.p_net = Net("Policy",
                         num_channels,
                         self.ac.num_axns,
                         self.data.resized_dims,
                         self.fp.is_using_server,
                         self.multi_res,
                         self.best_gpu_nums,
                         version,
                         to_visualize)

        self.t_net = Net("Target",
                         num_channels,
                         self.ac.num_axns,
                         self.data.resized_dims,
                         self.fp.is_using_server,
                         self.multi_res,
                         self.best_gpu_nums,
                         version,
                         to_visualize)

        if self.to_load_model:
            self.p_net, self.t_net, self.p_net_controller = self.fp.load_net(self.p_net, self.t_net, self.multi_res, self.p_net_controller)

        if self.loss.upper() == "MSE":
            loss = MSELoss()
        else:
            loss = HuberLoss()

        self.p_net_controller = NetController(net=self.p_net,
                                              optimizer=optim.AdamW(self.p_net.parameters(), lr=lr, amsgrad=True),
                                              loss_func=loss,
                                              batch_size=self.batch_size,
                                              discount=discount,
                                              axns=self.ac.axns,
                                              to_separate_q_prime=to_separate_q_prime,
                                              best_gpu_nums=self.best_gpu_nums)
        input_str = input("Load in Q net?\n")
        if input_str.upper().startswith("Y"):
            self.p_net_controller, self.p_net, self.t_net = self.fp.load_q_net(self.p_net_controller, self.p_net, self.t_net)

        self.t_net_controller = NetController(net=self.t_net,
                                              optimizer=optim.AdamW(self.t_net.parameters(), lr=lr, amsgrad=True),
                                              loss_func=loss,
                                              batch_size=self.batch_size,
                                              discount=discount,
                                              axns=self.ac.axns,
                                              to_separate_q_prime=to_separate_q_prime,
                                              best_gpu_nums=self.best_gpu_nums)
        # print([item for item in self.p_net.state_dict()])
        # print([item for item in self.p_net_controller.optimizer.state_dict()['param_groups']])
        # print(sum(1 for _ in self.p_net.models[0].parameters()))
        # print(sum(1 for _ in self.p_net.parameters()))
        # exit()
        # state dict for two layers above resnet - params are also separate
        # print(self.p_net.state_dict().keys())
        # state dict for resnet - params are also separate
        # print(self.p_net.models[0].state_dict().keys())

        # self.p_net_controller.print_summary([3] + self.data.resized_dims)

    '''=============================================== RESETS ======================================================='''

    # resets agent after new epoch
    def start_new_epoch(self):
        # determining epsilon, translation distance and scaling distance
        if self.is_pretraining:
            self.ac.epsilon = 100
        elif self.is_training:
            proposed_epsilon = self.start_epsilon - self.delta_epsilon * self.total_train_epochs
            proposed_trans_dist = self.start_trans_dist - self.environ.delta_alpha * self.total_train_epochs
            proposed_scale_dist = self.start_scale_dist - self.environ.delta_alpha * self.total_train_epochs
            if proposed_epsilon < self.end_epsilon:
                self.ac.epsilon = self.end_epsilon
            else:
                self.ac.epsilon = proposed_epsilon
            if proposed_scale_dist < self.environ.end_alpha:
                self.environ.scale_dist = self.environ.end_alpha
            else:
                self.environ.scale_dist = proposed_scale_dist
            if proposed_trans_dist < self.environ.end_alpha:
                self.environ.trans_dist = self.environ.end_alpha
            else:
                self.environ.trans_dist = proposed_trans_dist
        else:
            self.ac.epsilon = 0

        # clearing the running results... for the coming epoch
        self.times = []
        self.steps = []
        self.ious = []
        self.euclids = []
        # resetting the episode number
        self.ep_num = 0

    # resets agent for the new episode...
    def end_episode(self, start_time, epoch_num):
        # correcting display and removing inference transitions
        if not self.is_training:
            training_string = "INFERENCE SLICE:"
            for i in range(self.old_track_num + 1, self.track_num + 1):
                self.fp.delete_transition_file("I" + str(i) + ".pt")
        elif self.is_pretraining:
            training_string = "PRE-TRAIN SLICE:"
        else:
            training_string = "TRAIN SLICE:"

        # avg q value chosen
        if self.ac.num_q == 0:
            self.avg_q = 0
        else:
            self.avg_q = self.ac.q_summ / self.ac.num_q
        self.ac.q_summ = 0
        self.ac.num_q = 0

        # memory distribution
        summ = 0  # total transitions in memory
        chosen_summ = 0  # chosen transitions total
        for item in self.tc.categories.values():
            summ += item
        for item in self.p_net_controller.chosen_transitions.values():
            chosen_summ += item

        # network weights and bias
        weight_sum = [round(self.p_net.out.weight[0].mean().item(), 3),
                      round(self.p_net.out.weight[1].mean().item(), 3),
                      round(self.p_net.out.weight[2].mean().item(), 3),
                      round(self.p_net.out.weight[3].mean().item(), 3)]
        bias = [round(self.p_net.out.bias[0].item(), 3), round(self.p_net.out.bias[1].item(), 3),
                round(self.p_net.out.bias[2].item(), 3), round(self.p_net.out.bias[3].item(), 3)]

        # episode time and percent completion
        if not self.is_pretraining:
            self.total_episodes += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_elapsed_time = end_time - self.total_start_time
        percent_done = 100 * self.total_episodes / self.episodes_in_full_run
        time_left = (100 / percent_done) * total_elapsed_time - total_elapsed_time if percent_done != 0 else 0

        # getting percentages of transitions
        total = max(sum(self.p_net_controller.chosen_transitions.values()), 1)
        chosen = {k: f"{v / total:.0%}" for k, v in self.p_net_controller.chosen_transitions.items()}
        total = max(sum(self.tc.categories.values()), 1)
        categories = {k: f"{v / total:.0%}" for k, v in self.tc.categories.items()}

        # display
        print("\nCOMPLETED {0} Total Epoch "
              "{1}, Sub-epoch {2}, Episode {3} in {4} steps ({5:.1f} SEC). "
              "EPSILON={6}, ALPHA={7} \n\t>>> OVERALL {8:.1f}% done, {9:.0f} hr: {10:.0f} min. "
              "REMAINING {11:.0f} hr: {12:.0f} min\n\t>>> distribution: {13}, total: {14}"
              "\n\t>>> chosen transitions: {15}, total: {16}"
              "\n\t>>> weight mean of U D L R: {17}"
              "\n\t>>> bias: {18}"
              "\n\t>>> avg q chosen: {19:.3f}"
              "\n\t>>> {20}".format(training_string,
                                    self.total_epochs,
                                    epoch_num,
                                    self.ep_num,
                                    self.state_num,
                                    elapsed_time,
                                    self.ac.epsilon,
                                    self.environ.trans_dist,
                                    percent_done,
                                    total_elapsed_time // 3600,
                                    (total_elapsed_time % 3600) // 60,
                                    time_left // 3600,
                                    (time_left % 3600) // 60,
                                    categories,
                                    summ,
                                    chosen,
                                    chosen_summ,
                                    weight_sum,
                                    bias,
                                    self.avg_q,
                                    self.environ.display))
        # for logging results
        if not self.is_pretraining:
            if not self.is_training:
                rotated_box = self.fp.rotate_box(self.environ.final_box, 3 - len(self.last_four))
                self.last_four.append(rotated_box)
                if len(self.last_four) == 4:
                    median_box = self.find_median()
                    iou, euclid = self.environ.iou_and_euclid(self.slice2d.box, median_box, False)
                    self.ious += [round(iou, 3)] * 4
                    self.euclids += [round(euclid, 3)] * 4
                    self.last_four = []

            else:
                self.ious.append(round(self.environ.iou, 3))
                self.euclids.append(round(self.environ.euclid, 3))

            self.steps.append(self.track_num - self.old_track_num)
            self.times.append(elapsed_time)

            # todo here

        # resetting values
        self.old_track_num = self.track_num
        self.is_terminal_state = False
        self.ep_num += 1  # increases episode number
        self.state_num = 0  # resets to state 0
        self.track_num += 1  # increases track number by 1
        self.ac.sel_axn = None
        self.reward = 0
        self.qs = None
        self.slice2d.proposed_start = Box(self.environ.final_box.tl.copy(), self.environ.final_box.br.copy())

    # after finishing an epoch
    def end_epoch(self):
        if not self.is_pretraining:
            self.fp.make_summary_file(self.total_epochs, self.steps, self.times, self.ious, self.euclids,
                                      self.is_training)
            self.total_epochs += 1
            if self.is_training:
                self.total_train_epochs += 1

    # resets file and box buff after every state
    def reset_buffs_and_actions(self):
        code = "T" if self.is_training else "I"
        del self.box_memory_buff[0]
        del self.file_state_buff[0]
        self.box_memory_buff.append(self.box)
        self.file_state_buff.append(code + str(self.track_num) + ".pt")

    def find_median(self):
        tl_xs = []
        tl_ys = []
        br_xs = []
        br_ys = []

        for item in self.last_four:
            tl_xs.append(item.tl[0])
            tl_ys.append(item.tl[1])
            br_xs.append(item.br[0])
            br_ys.append(item.br[1])
        tl_x = np.median(np.array(tl_xs))
        tl_y = np.median(np.array(tl_ys))
        br_x = np.median(np.array(br_xs))
        br_y = np.median(np.array(br_ys))
        return Box([tl_x, tl_y], [br_x, br_y])

