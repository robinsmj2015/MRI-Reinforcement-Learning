from Transition import Transition
import math

class TransitionController:
    def __init__(self, max_transitions, axns):
        self.transitions = []
        self.max_transitions = max_transitions
        self.categories = {}
        for axn in axns:
            self.categories[axn+"+"] = 0
            self.categories[axn+"-"] = 0

    # storing in transition memory...
    # 4 boxes prior (with file name), 1 future state, action now and future action, reward
    def manage_transitions(self, slice2d, file_state_buff, box_memory_buff, reward, ep_num, state_num,
                           track_num, fp, ac, resized_dims, is_terminal, pos_reward, discount, to_calc_targets,
                           euclid_thresh, trans_dist):
        target = None
        if to_calc_targets:
            target = self.calc_target(box_memory_buff.copy()[-1], slice2d.box, pos_reward, discount, reward, euclid_thresh, trans_dist)

        if len(self.transitions) >= self.max_transitions:
            self.remove_transitions(fp, pos_reward)
        transition = Transition(slice2d.name,
                                file_state_buff.copy(),
                                box_memory_buff.copy(),
                                ac.sel_axn,
                                reward,
                                ep_num,
                                state_num,
                                track_num,
                                is_terminal,
                                target)
        self.transitions.append(transition)
        self.categorize_transition(transition, pos_reward, True)
        fp.create_transition_file(resized_dims=resized_dims, code="T", box=transition.box_states[-1], master_file=transition.master_file, track_num=transition.track_num)

    # ensures transition memory does not get too large - deleting a transition instance from the memory
    def remove_transitions(self, fp, pos_reward):
        # todo fix -- need to remove files of terminal states!!!
        file_names = set()
        for i in range(len(self.transitions[0].file_states)):
            file_names.add(self.transitions[0].file_states[i])
        for file_name in file_names:
            fp.delete_transition_file(file_name)
            while file_name in self.transitions[0].file_states:
                self.categorize_transition(self.transitions[0], pos_reward, False)
                del self.transitions[0]

    # looks at the transitions in memory
    # keeps track of each action and if the reward was + or -
    def categorize_transition(self, transition, pos_reward, to_add):
        if to_add:
            if transition.r == pos_reward:
                self.categories[transition.a + "+"] += 1
            else:
                self.categories[transition.a + "-"] += 1
        else:
            if transition.r == pos_reward:
                self.categories[transition.a + "+"] -= 1
            else:
                self.categories[transition.a + "-"] -= 1

    @staticmethod
    def calc_target(s_prime_box, true_box, pos_reward, discount, reward, euclid_thresh, trans_dist):
        target = reward
        quantity = (true_box.center[0] - s_prime_box.center[0]) ** 2 + (true_box.center[1] - s_prime_box.center[1]) ** 2
        euclid = quantity ** 0.5
        steps = (euclid - euclid_thresh) / ((s_prime_box.right_most - s_prime_box.left_most) * trans_dist)

        if steps > 0.5:
            steps = math.ceil(steps)
            target = reward + pos_reward * discount * (1 - discount ** steps) / (1 - discount)
        return target








