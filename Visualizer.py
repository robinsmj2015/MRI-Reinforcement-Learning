import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from matplotlib import patches
import time
from Box import Box
from PIL import Image, ImageDraw
import datetime
import torch

class Visualizer:
    def __init__(self, fp):
        self.fp = fp
        self.rect = None
        self.boxes = []  # boxes for animation
        self.displays = []  # text for log files... in animation
        # for animations
        self.anim = None
        self.manual_control = None  # manually display log files
        self.wait_time = None  # half second for automatic animation display
        self.skip = True  # just to skip for animation

        # lists used in making animations
        self.selected_qs = []
        self.selected_ious = []
        self.selected_euclids = []
        self.actions = []
        self.action_nums = []
        self.rewards = []

        # figure
        self.fig = None
        # axes of the figure
        self.ax_reward = None
        self.ax_action = None
        self.ax_iou = None
        self.ax_euclid = None
        self.ax_im = None
        self.ax_q = None
        # for tracking rewards and actions
        self.reward_count = {-1: 0, 1: 0}
        self.full_action_names = {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT", "T": "TALLER", "W": "WIDER",
                                  "O": "ZOOM OUT", "S": "SHORTER", "N": "NARROWER", "I": "ZOOM IN", "None": "NONE"}
        self.axn_num_dict = {}
        self.action_count = {}
        for i, axn in enumerate(self.full_action_names.keys()):
            self.axn_num_dict[axn] = i
            self.action_count[i] = 0
        self.num_axns = len(self.full_action_names)

    # used in log file call back
    def visualize_animations_helper(self, i):
        if self.skip:
            self.skip = False
            return
        if len(self.boxes) == i:
            time.sleep(10)
            exit()
        if not (self.rect is None):
            self.rect.remove()

        box = self.boxes[i]
        display = self.displays[i]
        selected_iou = self.selected_ious[i]
        selected_euclid = self.selected_euclids[i]
        selected_q = self.selected_qs[i]
        if selected_q is np.nan:
            selected_q = "Guided Exploration"
        reward = self.rewards[i]
        action = self.actions[i]
        action_num = self.action_nums[i]
        self.action_count[action_num] += 1
        if int(reward) != 0:
            self.reward_count[int(reward)] += 1
        color = self.determine_color(selected_iou)
        self.rect = patches.Rectangle(box.tl,
                                      box.right_most - box.left_most,
                                      box.bottom_most - box.top_most,
                                      fill=False,
                                      linewidth=0.2,
                                      color=color)
        xs = np.arange(i + 1)
        self.ax_im.add_patch(self.rect)
        # print(self.displays[i])

        if i == 0:
            im_title = (display.split(":\n")[0]).split("=\n")[1]
        else:
            im_title = display.split(":\n")[0]

        self.ax_action.cla()
        self.ax_reward.cla()
        self.ax_reward.set(xticks=[-1, 1], ylim=[0, 100], ylabel="%")
        self.ax_action.set(xticks=np.arange(self.num_axns), ylim=[0, 100], ylabel="%",
                           xticklabels=self.full_action_names.keys())

        self.ax_im.set_title(im_title, fontsize="medium")
        self.ax_q.set_title("Selected Q-value: {0}".format(selected_q), fontsize="medium")
        self.ax_q.scatter(xs, self.selected_qs[:i + 1], color="purple", marker=".")
        self.ax_iou.set_title("IoU: {0}".format(selected_iou), fontsize="medium")
        self.ax_iou.plot(xs, self.selected_ious[:i + 1], color="purple")  # marker=".")
        self.ax_euclid.set_title("Euclidean Distance: {0}".format(selected_euclid), fontsize="medium")
        self.ax_euclid.plot(xs, self.selected_euclids[:i + 1], color="purple")  # marker=".")
        self.ax_reward.set_title("Reward of {0}".format(reward), fontsize="medium")
        self.ax_reward.bar(x=[-1, 1], height=100 * np.array(list(self.reward_count.values())) / (i + 1), width=0.5,
                           color="purple")
        counts = np.array(list(self.action_count.values())) * 100
        self.ax_action.bar(x=np.arange(self.num_axns), height=counts / (i + 1), color="purple")
        self.ax_action.set_title("Action: {0} ({1})".format(self.full_action_names[action], self.action_nums[i]),
                                 fontsize="medium")

        if self.manual_control:
            input("Press ENTER to continue ")

    # shows real time visualizations, displays text, as agent moves
    @staticmethod
    def step_wise_visualize(box, slice2d, iou, reward, sel_axn, qs, ep_num, state_num, euclid, key, epoch, track_num,
                            axns, start_mode, box_mode, epsilon, total_epochs):
        file_name = str(track_num) + "_(" + str(ep_num) + "_" + str(state_num) + ").jpg"
        # starting out an epoch
        additional_display = ""
        if state_num == 0:
            epoch_start_display = ("==== RUNNING TOTAL EPOCH {0}, SUB-EPOCH {1}, "
                                   "EPISODE {2} for {3} with Start mode: {4}, "
                                   "Box mode: {5}, and Epsilon: {6:.1f}% ====").format(total_epochs, epoch, ep_num,
                                                                                       "P" + key.split("/P")[1],
                                                                                       start_mode,
                                                                                       box_mode, epsilon)
            # print(epoch_start_display)
            additional_display = epoch_start_display

        # gathering q values for each action for display
        the_q = "N/A"
        q_display = {}
        if not (qs is None):
            for a, q in zip(axns, qs[0]):
                q_display[a] = round(q.item(), 3)
            the_q = q_display[sel_axn]

        to_display = ("\nFor EPISODE {0}, STATE {1}:\nThe calculated Q-VALUES are: {2}"
                      "\nFor the ACTION TAKEN: {3}\nWith the Q-VALUE of: {4}"
                      "\nTo reach CURRENT BOX STATE: {5}, {6}"
                      "\nWhen the GROUND TRUTH BOX is at: {7}, {8}"
                      "\nThe agent was REWARDED: {9}\nBecause IoU is {10:.4f} and Euclid {11:.3f} sq. pixels"
                      "\n---------------------------------").format(ep_num, state_num, q_display, sel_axn, the_q,
                                                                    box.tl, box.br, slice2d.box.tl, slice2d.box.br,
                                                                    reward, iou, euclid)
        return additional_display + to_display, file_name

    # displays a log file
    def run_log_file(self, manual_control, wait_time=None):
        self.manual_control = manual_control
        self.wait_time = 100 if manual_control else wait_time
        self.rect = None
        ground_truth_box_displayed = False  # ground truth box displayed
        file, _ = self.fp.choose_file(".txt", self.fp.logs_path)
        to_display = ""
        im = None

        # searching through each line of the log file
        for line in file:
            to_display += line
            if line.endswith("==\n"):
                name = (line.split("for ")[1]).split(" with")[0].replace("=", "").replace("\n", "")
                fig_title = (line.split("for ")[1]).replace("=", "").replace("\n", "")
                full_name = self.fp.data_path + self.fp.get_file_path(name)
                # im = plt.imread(full_name)
                im = torch.load(full_name)[0]

            elif (not ground_truth_box_displayed) and "GROUND TRUTH BOX" in line:
                to_parse = line.split("is at: ")[1].replace("[", "").replace("]", "").replace(" ", "").replace("\n", "")
                box = self.run_log_file_parser(to_parse)
                ground_truth = patches.Rectangle(box.tl,
                                                 box.right_most - box.left_most,
                                                 box.bottom_most - box.top_most,
                                                 fill=False,
                                                 linewidth=0.2,
                                                 color="deepskyblue")
                ground_truth_box_displayed = True

            elif "CURRENT BOX STATE" in line:
                to_parse = line.split("CURRENT BOX STATE: ")[1].replace("[", "").replace("]", "").replace(" ",
                                                                                                          "").replace(
                    "\n", "")
                self.boxes.append(self.run_log_file_parser(to_parse))

            elif "Q-VALUE " in line:
                parsed = line.split("of: ")[1]
                if parsed == "N/A\n":
                    value = np.nan
                else:
                    value = float(parsed)
                self.selected_qs.append(value)

            elif "ACTION " in line:
                parsed = line.split("TAKEN: ")[1].replace("\n", "")
                self.actions.append(parsed)

            elif "REWARDED:" in line:
                parsed = line.split("REWARDED: ")[1].replace("\n", "")
                self.rewards.append(parsed)

            elif "IoU " in line:
                parsed_iou = (line.split(" and ")[0]).split("IoU is ")[1]
                parsed_euclid = (line.split(" Euclid ")[1]).split(" sq.")[0]
                iou_value = float(parsed_iou)
                euclid_value = float(parsed_euclid)
                self.selected_ious.append(iou_value)
                self.selected_euclids.append(euclid_value)

            elif line.endswith("--\n") or line.endswith("--"):
                self.displays.append(to_display.lower())
                to_display = ""
        file.close()

        for a in self.actions:
            action = self.axn_num_dict[a]
            self.action_nums.append(action)

        num_frames = len(self.boxes)
        self.fig, (
            (self.ax_im, self.ax_iou), (self.ax_q, self.ax_euclid), (self.ax_reward, self.ax_action)) = plt.subplots(3,
                                                                                                                     2)
        self.fig.suptitle(fig_title.split("=")[0], fontsize="medium")
        self.fig.canvas.manager.full_screen_toggle()
        self.ax_im.imshow(im, cmap="Greys")
        self.ax_im.add_patch(ground_truth)
        self.ax_q.set(xlim=[0, num_frames - 1], ylim=[-10, 10], ylabel="Q-value")
        self.ax_iou.set(xlim=[0, num_frames - 1], ylim=[0, 1], ylabel="Intersection over Union")
        self.ax_euclid.set(xlim=[0, num_frames - 1], ylim=[0, 100], ylabel="Euclidean Distance")

        self.fig.tight_layout()
        self.anim = animation.FuncAnimation(self.fig,
                                            self.visualize_animations_helper,
                                            save_count=num_frames,
                                            interval=self.wait_time,
                                            repeat=False)
        plt.show()

    @staticmethod
    def run_log_file_parser(to_parse):
        coordinates = list(to_parse.split(","))
        return Box([float(coordinates[0]), float(coordinates[1])], [float(coordinates[2]), float(coordinates[3])])

    @staticmethod
    def determine_color(iou):
        if iou < 0.33:
            color = "orangered"
        elif 0.33 < iou < 0.66:
            color = "yellow"
        else:
            color = "chartreuse"
        return color

    def visualize_summary(self):
        _, file_name = self.fp.choose_file(".csv", self.fp.summaries_path)

        df = pd.read_csv(self.fp.summaries_path + file_name)
        df.index.name = df.columns[0]
        length = len(df.index)
        # ["times", "steps", "ious", "euclids", "training"]
        # ["_median", "_mean", "_std", "_total"]

        # training
        tr_avg_times = np.zeros(length)
        tr_med_times = np.zeros(length)
        tr_total_times = np.zeros(length)
        tr_std_times = np.zeros(length)
        tr_avg_steps = np.zeros(length)
        tr_med_steps = np.zeros(length)
        tr_total_steps = np.zeros(length)
        tr_std_steps = np.zeros(length)
        tr_avg_euclids = np.zeros(length)
        tr_med_euclids = np.zeros(length)
        tr_total_euclids = np.zeros(length)
        tr_std_euclids = np.zeros(length)
        tr_avg_ious = np.zeros(length)
        tr_med_ious = np.zeros(length)
        tr_total_ious = np.zeros(length)
        tr_std_ious = np.zeros(length)

        # inference
        in_avg_times = np.zeros(length)
        in_med_times = np.zeros(length)
        in_total_times = np.zeros(length)
        in_std_times = np.zeros(length)
        in_avg_steps = np.zeros(length)
        in_med_steps = np.zeros(length)
        in_total_steps = np.zeros(length)
        in_std_steps = np.zeros(length)
        in_avg_euclids = np.zeros(length)
        in_med_euclids = np.zeros(length)
        in_total_euclids = np.zeros(length)
        in_std_euclids = np.zeros(length)
        in_avg_ious = np.zeros(length)
        in_med_ious = np.zeros(length)
        in_total_ious = np.zeros(length)
        in_std_ious = np.zeros(length)

        tr_vars = [tr_avg_times, tr_med_times, tr_total_times, tr_std_times, tr_avg_steps, tr_med_steps, tr_total_steps,
                   tr_std_steps, tr_avg_euclids, tr_med_euclids, tr_total_euclids, tr_std_euclids, tr_avg_ious,
                   tr_med_ious, tr_total_ious, tr_std_ious]

        in_vars = [in_avg_times, in_med_times, in_total_times, in_std_times, in_avg_steps, in_med_steps, in_total_steps,
                   in_std_steps, in_avg_euclids, in_med_euclids, in_total_euclids, in_std_euclids, in_avg_ious,
                   in_med_ious, in_total_ious, in_std_ious]

        names = ["times_mean", "times_median", "times_total", "times_std", "steps_mean", "steps_median", "steps_total",
                 "steps_std", "euclids_mean", "euclids_median", "euclids_total", "euclids_std", "ious_mean",
                 "ious_median", "ious_total", "ious_std"]

        tr_xs = []
        in_xs = []
        for index in df.index:
            i = int(index)
            for tr_var, in_var, name in zip(tr_vars, in_vars, names):
                val = df.loc[i, "training"]
                if bool(val):
                    tr_var[i] = df.loc[index, name]
                    tr_xs.append(i)
                else:
                    in_var[i] = df.loc[index, name]
                    in_xs.append(i)

        fig, ((ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23)) = \
            plt.subplots(nrows=3, ncols=4, sharex="col")
        axes = [ax00, ax01, ax10, ax11, ax12, ax13, ax20, ax21, ax22, ax23]

        ax00.set(title="TIME", ylabel="TOTAL time (s)")
        ax01.set(title="STEPS", ylabel="TOTAL steps")
        ax02.set(title="INTERSECTION OVER UNION", ylabel="IoU")
        ax03.set(title="EUCLIDEAN DISTANCE", ylabel="sq. pixels")

        ax00.plot(tr_xs, tr_total_times[tr_xs], label="Train", color="purple")
        ax01.plot(tr_xs, tr_total_steps[tr_xs], label="Train", color="purple")
        # ax02.plot(in_xs, in_total_ious[in_xs], label="Inference", color="purple")
        # ax03.plot(in_xs, in_total_euclids[in_xs], label="Inference", color="purple")

        ax10.set(ylabel="MEAN time (s)")
        ax11.set(ylabel="MEAN steps")
        ax12.set(ylabel="MEAN IoU")
        ax13.set(ylabel="MEAN euclidean distance")

        ax10.plot(tr_xs, tr_avg_times[tr_xs], label="Train", color="purple")
        ax11.plot(tr_xs, tr_avg_steps[tr_xs], label="Train", color="purple")
        ax12.plot(in_xs, in_avg_ious[in_xs], label="Inference", color="purple")
        ax13.plot(in_xs, in_avg_euclids[in_xs], label="Inference", color="purple")

        ax20.set(ylabel="MEDIAN time (s)")
        ax21.set(ylabel="MEDIAN steps")
        ax22.set(ylabel="MEDIAN IoU")
        ax23.set(ylabel="MEDIAN euclidean distance")

        ax20.plot(tr_xs, tr_med_times[tr_xs], label="Train", color="purple")
        ax21.plot(tr_xs, tr_med_steps[tr_xs], label="Train", color="purple")
        ax22.plot(in_xs, in_med_ious[in_xs], label="Inference", color="purple")
        ax23.plot(in_xs, in_med_euclids[in_xs], label="Inference", color="purple")

        for axis in axes:
            axis.set(xlabel="epochs")
            axis.legend()
        fig.suptitle(file_name + ":\n" + df.index.name, fontsize="small")
        fig.canvas.manager.full_screen_toggle()
        plt.tight_layout()
        plt.show()

    # makes the q map showing predictions
    def visualize_q_map(self, slice2d, resized_dims, fp, p_net, ac, epoch, box_size, sample_dist, memory_dist,
                        final_box, start_pos, is_training, ep_num, alpha):
        if is_training or ep_num % 200 != 0:
            return

        if ac.num_q == 0:
            avg_q = 0
        else:
            avg_q = ac.q_summ / ac.num_q

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8, 8))
        truth = slice2d.box
        im = torch.load(self.fp.data_path + slice2d.name)[0]

        ax2.imshow(im, cmap="Greys")

        # drawing the patches for true box and agent's final and start boxes
        length = int(box_size)
        true_w = truth.right_most - truth.left_most
        true_h = truth.bottom_most - truth.top_most
        true_rect = patches.Rectangle(truth.tl, true_w, true_h, fill=False, linewidth=0.5, color="deepskyblue")

        final_box_w = final_box.right_most - final_box.left_most
        final_box_h = final_box.bottom_most - final_box.top_most
        final_rect = patches.Rectangle(final_box.tl, final_box_w, final_box_h, fill=False, linewidth=0.4, color="red")

        start_pos_w = start_pos.right_most - start_pos.left_most
        start_pos_h = start_pos.bottom_most - start_pos.top_most
        start_rect = patches.Rectangle(start_pos.tl, start_pos_w, start_pos_h, fill=False, linewidth=0.5, color="white")

        # iterating through every box to make the heatmap
        for tl_x in range(0, resized_dims[0] - int(alpha * length) + 1, int(length * alpha)):
            for tl_y in range(0, resized_dims[1] - int(alpha * length) + 1, int(length * alpha)):
                box = Box([tl_x, tl_y], [tl_x + int(alpha * length) - 1, tl_y + int(alpha * length) - 1])
                proposed_w = box.right_most - box.left_most
                proposed_h = box.bottom_most - box.top_most
                name = fp.create_transition_file(resized_dims=resized_dims,
                                                 code="H",
                                                 box=box,
                                                 master_file=slice2d.name,
                                                 track_num="")

                qs = ac.get_qs(p_net, fp, name)
                color, rads = self.color_q_map(qs)
                predicted_box = patches.Rectangle(box.tl, proposed_w, proposed_h, fill=True, linewidth=0.2, color=color,
                                                  alpha=0.3)
                predicted = patches.RegularPolygon(box.center, 3, int(length * alpha / 2), orientation=rads, color=color, alpha=0.3,
                                                   linewidth=0.3, fill=False)
                ax2.add_patch(predicted)
                ax2.add_patch(predicted_box)
        ax2.add_patch(true_rect)
        ax2.add_patch(final_rect)
        ax2.add_patch(start_rect)
        ax2.set_title("green-U, blue-D\npurple-L, yellow-R\navg q chosen: {0:.2f}".format(avg_q))
        name = self.fp.plt_ims_path + "Map epoch " + str(epoch) + " slice " + slice2d.name.split("/P")[1].replace(
            ".jpg", ".png")
        fig.suptitle("Epoch " + str(epoch) + ".\n" + slice2d.name.split("/P")[1].replace(".jpg", ""), size="medium")
        samples = np.array(list(sample_dist.values()))
        memory = np.array(list(memory_dist.values()))
        total_samples = max(samples.sum().item(), 1)
        total_memory = memory.sum().item()
        xs = np.arange(len(sample_dist))
        c = {0: "green", 1: "green", 2: "blue", 3: "blue", 4: "purple", 5: "purple", 6: "gold", 7: "gold"}
        for i in range(0, 8):
            ax0.bar(x=xs[i], height=samples[i] / total_samples, color=c[i])
            ax1.bar(x=xs[i], height=memory[i] / total_memory, color=c[i])
            if i % 2 == 1:
                ax0.text(x=xs[i - 1], y=min(0.5, max(samples[i] / total_samples, samples[i - 1] / total_samples)),
                         s=f"{(samples[i] + samples[i - 1]) / total_samples:.2f}")
                ax1.text(x=xs[i - 1], y=min(0.5, max(memory[i] / total_memory, memory[i - 1] / total_memory)),
                         s=f"{(memory[i] + memory[i - 1]) / total_memory:.2f}")
        ax0.set(ylabel="fraction", xticks=xs, xticklabels=sample_dist.keys(), ylim=[0, .5],
                title=f"Selected:\n{total_samples}")
        ax1.set(ylabel="fraction", xticks=xs, xticklabels=sample_dist.keys(), ylim=[0, .5],
                title=f"Current:\n{total_memory}")

        weight_sum = [round(p_net.out.weight[0].mean().item(), 3), round(p_net.out.weight[1].mean().item(), 3),
                      round(p_net.out.weight[2].mean().item(), 3), round(p_net.out.weight[3].mean().item(), 3)]
        bias = [round(p_net.out.bias[0].item(), 3), round(p_net.out.bias[1].item(), 3),
                round(p_net.out.bias[2].item(), 3),
                round(p_net.out.bias[3].item(), 3)]
        ax3.set(xticks=[0.33, 1.33, 2.33, 3.33], xticklabels=["U", "D", "L", "R"], title="Mean weight, bias")
        for i in range(4):
            ax3.bar(x=[i], height=weight_sum[i], color=c[int(i * 2)], label=ac.axns[i] + " mean weight", width=0.4)
            ax3.bar(x=[i + .5], height=bias[i], color=c[int(i * 2)], label=ac.axns[i] + " bias", width=0.4)
        # ax3.legend()
        plt.tight_layout()
        plt.savefig(name.replace("Map",
                                 str(datetime.datetime.now().strftime("%b %d, %Y %I_%M%p") + " Map")).replace(
                                 "NORM.pt", ".png"), format="png")
        plt.close(fig)

    @staticmethod
    def color_q_map(qs):
        if np.argmax(qs) == 0:
            color = "lime"
            rads = np.pi
        elif np.argmax(qs) == 1:
            color = "cyan"
            rads = 0
        elif np.argmax(qs) == 2:
            color = "fuchsia"
            rads = np.pi / 2
        else:
            color = "yellow"
            rads = 3 * np.pi / 2
        return color, rads

    @staticmethod
    def visualize_inputs(inputs, std_fac=None, mean_fac=None):
        np_inputs = np.array(inputs)
        if not (std_fac is None):
            np_inputs = np_inputs * std_fac + mean_fac
        fig, axes = plt.subplots(nrows=1, ncols=np_inputs.shape[0])
        print(np_inputs.shape[0])
        for i in range(np_inputs.shape[0]):
            axes[i].imshow(np_inputs[i], cmap="Greys")
        plt.tight_layout()
        fig.canvas.manager.full_screen_toggle()
        plt.show()
        time.sleep(2)
        plt.close(fig)

    def visualize_bounding_box(self):
        df = pd.read_csv("SAGT1 Bone.csv", index_col="im")
        for name in df.index:
            new_name = self.fp.get_file_path(name)
            with Image.open(self.fp.data_path + new_name.replace(" RES", "")) as im:

                plt.imshow(im, cmap="Greys")
                #draw = ImageDraw.Draw(im)

                final_rect = patches.Rectangle((df.loc[name, "box_x"] * 512, df.loc[name, "box_y"] * 512), df.loc[name, "box_w"] * 512, df.loc[name, "box_h"] * 512, fill=False, linewidth=0.4,
                                               color="red")

                plt.gca().add_patch(final_rect)
                plt.show()


                # the four corners of the image

