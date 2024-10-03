import numpy as np
import torch.nn.utils
from torchsummary import summary
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader


class NetController:
    def __init__(self, net, optimizer, loss_func, batch_size, discount, axns, to_separate_q_prime, best_gpu_nums):
        self.net = net
        for param in net.fc_hidden0.parameters():
            param.requires_grad = True
        for param in net.out.parameters():
            param.requires_grad = True
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.discount = discount  # discount factor
        self.to_separate_q_prime = to_separate_q_prime
        self.device = torch.device("cuda:" + str(best_gpu_nums[0])) if torch.cuda.is_available() else torch.device("cpu")
        # memory distribution initialization
        self.chosen_transitions = {}
        for axn in axns:
            self.chosen_transitions[axn + "+"] = 0
            self.chosen_transitions[axn + "-"] = 0

    # training loop
    def train_net(self, train_loader, actions, epochs=1):
        for epoch in range(epochs):
            for data in train_loader:
                # 2d (batch, chosen action num)
                inputs, labels = data
                self.net.train()
                initial_output = self.net.forward(inputs.to(self.device))
                labels = torch.unsqueeze(labels, dim=-1)
                actions = torch.unsqueeze(torch.tensor(actions, requires_grad=False), dim=-1)
                outputs = torch.gather(initial_output, index=actions.to(self.device), dim=1)
                self.optimizer.zero_grad()
                loss = self.loss_func(outputs, labels.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.net.parameters(), 100)
                self.optimizer.step()

    # organizes data for training the network
    def update_net(self, data, transitions, axn_num_dict, t_net, fp, pos_reward, to_calc_targets):
        q_dataset, q_prime_dataset, actions, rs, terminals, named_actions, targets = \
            data.make_net_training_input_data(self.batch_size, transitions, axn_num_dict, to_calc_targets)
        # keeps track of memory distribution
        for r, named_action in zip(rs, named_actions):
            if r == pos_reward:
                self.chosen_transitions[named_action + "+"] += 1
            else:
                self.chosen_transitions[named_action + "-"] += 1

        if not to_calc_targets:
            # this prepares the state for the target network to perform predictions on the Q' value
            prediction_set = CustomDataset(q_prime_dataset, fp, data.resized_dims, is_for_prediction=True)
            prediction_loader = DataLoader(prediction_set, batch_size=self.batch_size, shuffle=False)
            labels = []
            a_primes = []
            with torch.no_grad():
                for inputs, _ in prediction_loader:
                    t_net.eval()
                    labels.append(t_net.forward(inputs.to(self.device)))
                    # for separating q and a'
                    if self.to_separate_q_prime:
                        self.net.eval()
                        a_primes.append(self.net.forward(inputs.to(self.device)))
                labels = labels[0].to("cpu")
                if self.to_separate_q_prime:
                    a_primes = a_primes[0].to("cpu")

            # we only update the q value for the action taken using the Q' value of the target net for a'!
            # td(0) update
            # some papers suggest it might be better to let the a' be chosen by the policy net and Q' by target
            if self.to_separate_q_prime:
                targets = [self.discount * l[np.argmax(a_p)] + r if not ts else torch.tensor(r, requires_grad=False, device="cpu") for
                           (l, a, a_p, r, ts) in zip(labels, actions, a_primes, rs, terminals)]

            else:
                targets = [self.discount * max(l) + r if not ts else torch.tensor(r, requires_grad=False, device="cpu") for
                           (l, a, r, ts) in zip(labels, actions, rs, terminals)]
        else:
            # uses calculated target values
            targets = [torch.tensor(target, requires_grad=False, device="cpu") for target in targets]


        # the training dataset is now being prepared
        q_dataset = q_dataset.assign(s_prime=targets)
        train_set = CustomDataset(q_dataset, fp, data.resized_dims)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
        self.train_net(train_loader, actions)

    def print_summary(self, size):
        summary(self.net, size)
        print("More details...")
        print(self.net)

    def soft_update(self, tau, p_net):
        p_net_dict = p_net.state_dict()
        if tau == 1:
            self.net.load_state_dict(p_net_dict.copy())
        else:
            t_net_dict = self.net.state_dict()
            for key in p_net_dict:
                t_net_dict[key] = t_net_dict[key] * (1 - tau) + p_net_dict[key] * tau
            self.net.load_state_dict(t_net_dict)
