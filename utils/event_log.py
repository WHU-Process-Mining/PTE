import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

def get_time_feature(time_seq):
    case_interval = [[i[j] - i[0] for j in range(1, len(i))]  for i in time_seq]
    event_interval = [[i[j] - i[j-1] for j in range(1, len(i))]  for i in time_seq]

    mean_case_interval = np.mean([86400 * item.days + item.seconds for sublist in case_interval for item in sublist])
    mean_event_interval = np.mean([86400 * item.days + item.seconds for sublist in event_interval for item in sublist])

    return {'mean_case_interval': mean_case_interval,
            'mean_event_interval': mean_event_interval}

def split_valid_df(df, valid_ratio):
    case_start_df = df.pivot_table(values='time:timestamp', index='case:concept:name', aggfunc='min').reset_index().sort_values(by='time:timestamp', ascending=True).reset_index(drop=True)
    ordered_id_list = list(case_start_df['case:concept:name'])

    # Get first validation case index
    first_val_case_id = int(len(ordered_id_list)*(1-valid_ratio))

    # Get lists of case ids to be assigned to val and train set
    val_case_ids = ordered_id_list[first_val_case_id:]
    train_case_ids = ordered_id_list[:first_val_case_id]

    # Final train-val split 
    train_set = df[df['case:concept:name'].isin(train_case_ids)].copy().reset_index(drop=True)
    val_set = df[df['case:concept:name'].isin(val_case_ids)].copy().reset_index(drop=True)

    return train_set, val_set

class EventLogData():
    def __init__(self, df):
        self.all_activities = np.unique(df['concept:name'])
        self.all_activities.sort()  # sort the columns to ensure the order
        self.total_activities_num = len(np.unique(df['concept:name']))
        self.activity2id = dict(zip(self.all_activities, range(1, self.total_activities_num + 1)))
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)
        time_list = df.groupby('case:concept:name')['time:timestamp'].apply(list).tolist()
        self.time_feature = get_time_feature(time_list)
        self.max_len = max([len(i) for i in time_list])-1 # max len of the prefix

        
    def generate_data_for_input(self, df):
        all_cases = np.unique(df['case:concept:name'])
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)
        total_data_list = [] # [[activity_seq][time_seq]] 
        for case_id in all_cases:
            case_row = df[df['case:concept:name'] == case_id]
            case_row = case_row.sort_values(by=['time:timestamp'])
            if len(case_row)<=self.max_len+1:
                activity_seq = case_row['concept:name'].to_list()
                activity_seq = [self.activity2id.get(activity, len(self.activity2id) + 1) for activity in activity_seq]
                time_seq = case_row['time:timestamp'].to_list()
            
                if len(activity_seq) <2 :
                    raise ValueError("Invalid sequence length < 2")
                is_valids = case_row['predictable'].to_list()
                for i in range(1, len(activity_seq)):
                    if (is_valids[i] == 1) and (activity_seq[i] <= self.total_activities_num):
                        total_data_list.append([activity_seq[:i+1], time_seq[:i+1]])
        return total_data_list
