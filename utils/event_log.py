import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

class EventLogData():
    def __init__(self, path, activity2id = None):
        df = pd.read_csv(path, usecols=['case:concept:name', 'concept:name', 'time:timestamp'], dtype={
            'case:concept:name': str, 'concept:name': str, 'time:timestamp':str})
        
        self.event_log = df
        self.all_activities = np.unique(df['concept:name'])
        self.all_cases = np.unique(df['case:concept:name'])
        
        self.total_activities_num = len(np.unique(df['concept:name']))
        if activity2id:
            self.activity2id = activity2id
        else:
            # Map activity name to int category  1~activity_num                                                           
            self.activity2id = dict(zip(self.all_activities, range(1, self.total_activities_num + 1)))
        
        self.total_data_list = self._generate_data_for_input()

        
    def _generate_data_for_input(self):
        total_data_list = [] # [[activity_seq][time_seq]] 
        for case in self.all_cases:
            activity_seq = self.event_log[self.event_log['case:concept:name'] == case]['concept:name'].to_list()
            time_seq = self.event_log[self.event_log['case:concept:name'] == case]['time:timestamp'].to_list()
            
            activity_seq = list(map(lambda x: self.activity2id[x], activity_seq))
            time_seq = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), time_seq))
            if len(activity_seq) <2 :
                raise ValueError("Invalid sequence length < 2")
            for i in range(2, len(activity_seq)+1):
                total_data_list.append([activity_seq[:i], time_seq[:i]])
        return total_data_list
    
    def split_valid_data(self, valid_ratio):
        valid_n = int(valid_ratio * len(self.total_data_list))
        
        train_data_list , valid_data_list = train_test_split(
            self.total_data_list, test_size=valid_n, shuffle=True
        )
        return train_data_list, valid_data_list