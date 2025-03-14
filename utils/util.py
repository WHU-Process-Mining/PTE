import numpy as np
import matplotlib.pyplot as plt

def generate_curve(file_path, train_loss_list, train_acc_list, val_acc_list):
    assert len(train_loss_list) == len(train_acc_list) == len(val_acc_list), "data list length not consistent, please check out d data"
    
    epochs_list = [i for i in range(1, len(train_loss_list)+1)]
    fig = plt.figure(figsize=(12,6)) 
    ax1 = fig.add_subplot(121)  

    ax1.plot(epochs_list, train_loss_list, linewidth=2, label='train_loss', color='Blue')
    ax1.legend(loc='upper right')

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")


    ax2 = fig.add_subplot(122)  

    ax2.plot(epochs_list, train_acc_list, linewidth=2, label='train_acc', color='Blue')
    ax2.plot(epochs_list, val_acc_list, linewidth=2, label='val_acc', color='Red')

    ax2.legend(loc='upper right')

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")

    plt.savefig(file_path)
    plt.close()

# Gets the case list of the most recent window size ws
def get_w_list(current_list, ws):
    current_len = len(current_list)
    if ws > current_len:
        w_list = np.pad(current_list, (ws - current_len, 0), 'constant')
    else:
        w_list = current_list[-ws:]

    return list(w_list)

def get_time_feature(time_series:list, time_feature:dict):
    """
    Get time feature corresponding to the time series
    :time_series: datatime formate
    :time_feature: time feature dict
    :return: (time interval since the start of case, 
                time interval since the last event,
                time interval since the midnight,
                day in a week)
    """
    mean_case_interval = time_feature['mean_case_interval']
    mean_event_interval = time_feature['mean_event_interval']

    # timesincecasestart
    timesincecasestart = [i-time_series[0] for i in time_series]
    raw_time_1 = [86400 * i.days + i.seconds for i in timesincecasestart]
    time_1 = raw_time_1/mean_case_interval
    
    # timesincelastevent
    time_2 = [0] + [raw_time_1[i] - raw_time_1[i-1] for i in range(1, len(raw_time_1))]
    time_2 = time_2/mean_event_interval
    
    # timesincemidnight
    timesincemidnight = [i-i.replace(hour=0, minute=0, second=0, microsecond=0) for i in time_series]
    time_3 = [i.seconds/86400 for i in timesincemidnight]
    
    # Monday:0 Sunday:6
    time_4 = [(i.weekday()+1)/7 for i in time_series]
    # time_4 = [(np.sin(2*np.pi*i.weekday()+0.5/7)+1)/2 for i in time_series]
    return time_1, time_2, time_3, time_4
