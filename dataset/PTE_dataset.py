from dataset.ap_dataset import APDataset
from utils.util import get_w_list, get_time_feature
import numpy as np

class PTEDataset(APDataset):
    """
    Dataset responsible for consuming scenarios and producing pairs of model inputs/outputs.
    """

    def __init__(self, data_list, max_len, time_feature_dict, shuffle=False):
        super(PTEDataset, self).__init__(data_list, shuffle)

        self.max_len = max_len
        self.data_list = [i for i in self.data_list if len(i[0]) <= (self.max_len+1)]
        self.time_feature_dict = time_feature_dict
        

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        """
        Retrieves the dataset examples corresponding to the input index
        :param idx: input index
        :return: (history sequence(activity + time), next activity)
        """
        activity_seq, time_seq = self.data_list[idx]
        time_case, time_event, time_day, time_week = get_time_feature(time_seq, self.time_feature_dict)
        history_acyicity_seq = np.array(get_w_list(activity_seq[:-1], self.max_len))
        
        history_time_case_seq = np.array(time_case[:-1])
        history_time_case_seq = np.array(get_w_list(time_case[:-1], self.max_len))
        history_time_event_seq = np.array(get_w_list(time_event[:-1], self.max_len))
        history_time_day_seq = np.array(get_w_list(time_day[:-1], self.max_len))
        history_time_week_seq = np.array(get_w_list(time_week[:-1], self.max_len))
        history_seq = np.array([history_acyicity_seq, history_time_case_seq, history_time_event_seq, history_time_day_seq, history_time_week_seq], dtype=np.float32)
        return history_seq,  np.array(activity_seq[-1])