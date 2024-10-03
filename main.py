from Agent import Agent
from Data import Data
from UnitTester import UnitTester
from FileProcessing import FileProcessing
from ManageGPUs import ManageGPUs
from SuperviseResNet import *
from ExSitu import run


# SAGIR: (256, 242) 23%, (256, 256) 11%, (512, 512) 31%
# SAGT1: (768, 768) 19% CROPPED, (864, 864) 49%

torch.cuda.empty_cache()
is_using_server = torch.cuda.is_available()
best_gpu_nums = None
if is_using_server:
    gpu_manager = ManageGPUs()
    best_gpu_nums = gpu_manager.get_gpu_utilization()

# -------------------------------- HYPERPARAMETERS -------------------------------
# GENERAL
run_mode = "VIEW"  # MAIN or VIEW or EXSITU, SUPERVISE
path_abr, scan_type = "E", "T1"  # "F", # "IR"
mega_epochs, train_epochs, infer_epochs = 10, 4, 1
axns = ["U", "D", "L", "R"]  # "I", "T", "W", # "O", "S", "N"]
box_mode, multi_res = "F", False  # S, P, F, S_P_F, S_P, S_F, P_F
scaling_factor, max_zoom_fit, normalizing_thresh = 1, 4, 0.01  # to scale dimensions up by when ending, for fit mode max zoom in amount, thresh frac
dims = [864, 864]  # Mark dimensions [256, 242]: 23%, [256, 256]: 11%, [512, 512]: 31%
resized_dims = [224, 224]  # To use dimensions: [242, 256], [256, 256], [128, 128], [64, 64]
start_mode, start_fraction, box_size = "FIX_RAND", 0.25, 56  # ORIGINAL, RAND, SCALING, FIX_RAND, FIX_MEAN, FIX_CENT, FIX_tlx,tly,brx,bry
# Data
max_slices = 500  # can make big num to use all slices...
train_frac, val_frac, test_frac = 0.5, 0.25, 0.25
seed = 7
# Updating
t_net_update_time = 1000  # was 1
soft_update_tau = 1  # copy update,  was .005
max_transitions, pre_transition_steps, pre_transition_epochs = 10000, 3, 3  # pre transitions per slice
# Training
version, to_visualize = 101, False  # resnet version
start_epsilon, delta_epsilon, end_epsilon, guided_exploration, to_help = 80, 1, 40, .5, True  # epsilon and fraction guided exploration and scaling guided exploration
max_train_states = 40
iou_thresh, euclid_thresh = 0.70, 16  # euclid should NOT be less than box size * alpha / 2
loss, to_separate_q_prime = "Huber", True
batch_size, lr = 32, 5e-5
discount = 0.9  # discount factor (gamma)
start_alpha, delta_alpha, end_alpha = 0.5, .001, 0.25
reward_metric, to_calc_targets = "EUCLID", False  # reward metric BOTH, IOU, EUCLID
to_save_models, to_clear_starts, to_clear_models, to_clear_transitions = True, True, False, True
# Inference
to_load_model, val_with_train = True, False
max_infer_states = 30
stopping_criteria = [15, 15, 0]  # hits in last x boxes with no oscillation (0) or with oscillation (1)
pos_reward, neg_reward = 1, -1
always_neg_reward = True

# inference stopping is still disabled...
# trigger action


ut = UnitTester()
data = Data(resized_dims)
fp = FileProcessing(box_mode=box_mode,
                    is_using_server=is_using_server,
                    epochs=[mega_epochs, train_epochs, infer_epochs],
                    max_zoom_fit=max_zoom_fit,
                    normalizing_thresh=normalizing_thresh)


train_pat_nums, val_pat_nums, test_pat_nums, train_slices, val_slices, test_slices, center = \
    data.split_pat_data(fp,
                        train_frac,
                        val_frac,
                        dims,
                        resized_dims,
                        seed,
                        gen_file=False,
                        path_abr=path_abr,
                        scan_type=scan_type)

agent = Agent(name="agent",
              data_instance=data,
              file_processing=fp,
              epsilon=start_epsilon,
              delta_epsilon=delta_epsilon,
              trans_dist=start_alpha,
              scale_dist=start_alpha,
              iou_thresh=iou_thresh,
              euclid_thresh=euclid_thresh,
              max_slices=max_slices,
              loss=loss,
              lr=lr,
              batch_size=batch_size,
              t_net_update_time=t_net_update_time,
              soft_update_tau=soft_update_tau,
              max_transitions=max_transitions,
              graphics_mode="NONE",
              to_load_model=to_load_model,
              max_train_states=max_train_states,
              max_infer_states=max_infer_states,
              discount=discount,
              stopping_criteria=stopping_criteria,
              start_mode=start_mode,
              scaling_factor=scaling_factor,
              axns=axns,
              end_epsilon=end_epsilon,
              delta_alpha=delta_alpha,
              end_alpha=end_alpha,
              reward_metric=reward_metric,
              start_fraction=start_fraction,
              pre_transition_steps=pre_transition_steps,
              pre_transition_epochs=pre_transition_epochs,
              guided_exploration=guided_exploration,
              train_center=center,
              box_size=box_size,
              to_separate_q_prime=to_separate_q_prime,
              multi_res=multi_res,
              to_run=True if run_mode == "MAIN" else False,
              to_help=to_help,
              best_gpu_nums=best_gpu_nums,
              pos_reward=pos_reward,
              neg_reward=neg_reward,
              to_calc_targets=to_calc_targets,
              version=version,
              to_visualize=to_visualize,
              always_neg_reward=always_neg_reward)
agent.visualizer.visualize_bounding_box()

exit()
# For training and testing...
if run_mode == "MAIN":
    agent.run_train_and_inference(mega_epochs=mega_epochs,
                                  train_epochs=train_epochs,
                                  infer_epochs=infer_epochs,
                                  train_slices=train_slices,
                                  infer_slices=(train_slices if val_with_train else val_slices),
                                  start_mode=start_mode,
                                  to_save_models=to_save_models,
                                  to_clear_starts=to_clear_starts,
                                  to_clear_models=to_clear_models,
                                  to_clear_transitions=to_clear_transitions)

# For visualizing results...
elif run_mode == "VIEW":
    agent.visualizer.visualize_summary()
    #agent.visualizer.run_log_file(manual_control=False, wait_time=100)
    pass

elif run_mode == "SUPERVISE":
        for v in [101]:
            print(f"====================Version={v}, euclid={36}========================")
            run_supervised_learning(train_slices=train_slices,
                                    val_slices=val_slices,
                                    train_epochs=30,
                                    train_batch=32,
                                    val_batch=64,
                                    t_data_epochs=100,
                                    lr=5e-5,
                                    box_mode="F",
                                    max_zoom_fit=4,
                                    size=56,
                                    v_data_epochs=1,
                                    val_period=2,
                                    version=v,
                                    metric="euclid",
                                    euclid_thresh=36,
                                    path_abr=path_abr,
                                    scan_type=scan_type)

else:
    # ex situ: PIN
    run(train_slices, val_slices, path_abr + "," + scan_type)