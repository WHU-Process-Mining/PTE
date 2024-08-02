from torch.utils.data import Dataset
import random

class APDataset(Dataset):
    """
    Dataset responsible for pairs of Activity Prediction model inputs/outputs.
    """
    def __init__(self, data_list, shuffle=False):
        super().__init__()
        self.data_list = data_list
        if shuffle:
            random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        """
        Retrieves the dataset examples corresponding to the input index
        :param idx: input index
        :return: history sequence,next activity
        """
        return self.data_list[idx][0], self.data_list[idx][1]

