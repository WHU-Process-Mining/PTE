import torch
import pickle
from utils.event_log import EventLogData
from configs.config import load_config_data
from train_pte import test_model
from utils.metric import EvaluationMetric
import os
import pandas as pd
from dataset.PTE_dataset import PTEDataset
from model.PTE import TransitionPlaceEmbeddingModel

if __name__ == "__main__":
    
    cfg_model = load_config_data("configs/PTE_Model.yaml")
    dataset_cfg = cfg_model['data_parameters']
    model_cfg = cfg_model['model_parameters']
    
    data_path = '{}/{}/time-process/'.format(dataset_cfg['data_path'], dataset_cfg['dataset'])
    save_folder = 'results/{}/{}'.format(model_cfg['model_name'], dataset_cfg['dataset'])
    
    os.makedirs(f'{save_folder}/best_model', exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    
    train_file_name = data_path + 'train.csv'   
    test_file_name = data_path + 'test.csv'
    
    train_df = pd.read_csv(train_file_name)
    test_df = pd.read_csv(test_file_name)

    event_log = EventLogData(train_df)
    test_data_list = event_log.generate_data_for_input(test_df)
    
    max_len = event_log.max_len 
    time_feature_dict = event_log.time_feature
    test_dataset = PTEDataset(test_data_list, max_len, time_feature_dict, shuffle=False)
    model_cfg['activity_num'] = len(event_log.activity2id)

    model = TransitionPlaceEmbeddingModel(
            transition_num=model_cfg['activity_num'],
            dimension=model_cfg['dimension'],
            dropout=model_cfg['dropout'],
            beta=model_cfg['beta']).to(device)
    
    # Load the best model.
    with open(f'{save_folder}/model/best_model.pth', 'rb') as fin:
       best_model_dict = torch.load(fin)
    
    model.load_state_dict(best_model_dict)
    
    true_list, predictions_list, length_list = test_model(test_dataset, model, model_cfg['batch_size'], device)
    evaluator = EvaluationMetric(save_folder+"/result/next_activity.csv", max_len)
    evaluator.prefix_metric_calculate(true_list, predictions_list, length_list)