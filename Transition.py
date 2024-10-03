# Defines the transition that will be saved in memory and randomly sampled in experience replay
class Transition:
    def __init__(self, master_file, file_states, box_states, a, r, ep_num, state_num, track_num, is_terminal, target):
        self.master_file = master_file  # P108 SAGIR_019.jpg
        # instance of box class, s_box.UL (upper left) and s_box.BR (bottom right) are attributes
        self.a = a  # {U, D, L, R, T, W, S, N, I, O}
        self.r = r  # {neg_reward, pos_reward}
        self.ep_num = ep_num
        self.state_num = state_num
        self.track_num = track_num
        self.file_states = file_states
        self.box_states = box_states
        self.is_terminal = is_terminal
        self.target = target

