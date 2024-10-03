from torch.utils.data import Dataset
from torch import load


# custom dataset of tensor files
class CustomDataset(Dataset):
    def __init__(self, inputs, file_processing, resized_dims, is_for_prediction=False):
        self.inputs = inputs  # data to combine into dataset
        self.fp = file_processing  # instance of file processor
        self.resized_dims = resized_dims  # dimensions of slices
        self.is_for_prediction = is_for_prediction  # whether dataset is for prediction
        self.box_memory_buff = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # no labels for prediction mode
        # combine file methods combines the state files together
        if self.is_for_prediction:
            inputs = load(self.fp.transition_path + self.inputs.iloc[idx, 0])
            label = 0  # labels of 0 when using for prediction... we will try to find labels
        else:
            # when using to train the nets
            # -1 because we don't include label in inputs (only in label)
            inputs = load(self.fp.transition_path + self.inputs.iloc[idx, 0])
            label = self.inputs.iloc[idx, -1]
        return inputs, label


