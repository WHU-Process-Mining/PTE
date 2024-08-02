import torch
import pickle
from utils.event_log import EventLogData
from configs.config import load_config_data
from train_pte import test_model, get_time_setting
from utils.metric import EvaluationMetric
import os

if __name__ == "__main__":
    
    cfg_model = load_config_data("configs/PTE_Model.yaml")
    dataset_cfg = cfg_model['data_parameters']
    model_cfg = cfg_model['model_parameters']
    
    data_path = '{}/{}/process'.format(dataset_cfg['data_path'], dataset_cfg['dataset'])
    save_folder = 'results_kfold_{}/{}/{}'.format(dataset_cfg['k_fold_num'], model_cfg['model_name'], dataset_cfg['dataset'])
    
    os.makedirs(f'{save_folder}/best_model', exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    for idx in range(dataset_cfg['k_fold_num']):
        train_file_name = data_path + '/kfoldcv_' + str(idx) + '_train.csv'   
        test_file_name = data_path + '/kfoldcv_' + str(idx) + '_test.csv'
        
        train_log = EventLogData(train_file_name)
        test_log = EventLogData(test_file_name, train_log.activity2id)
        
        # Load the best model.
        with open(f'{save_folder}/model/best_model_kfd{idx}.pickle', 'rb') as fin:
            best_model = pickle.load(fin).to(device)
        
        max_len, max_case_interval, min_case_interval, max_event_interval, min_event_interval = get_time_setting(train_log.total_data_list)
        
        true_list, predictions_list, length_list = test_model(test_log.total_data_list, best_model, 
                                                              max_len, max_case_interval, min_case_interval, max_event_interval, min_event_interval, 
                                                              model_cfg['batch_size'], device)
        evaluator = EvaluationMetric(save_folder+"/result/k_fold_"+str(idx)+"_next_activity.csv", max_len)
        evaluator.prefix_metric_calculate(true_list, predictions_list, length_list)