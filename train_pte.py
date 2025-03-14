import torch
import random
import pickle
import optuna
import numpy as np
import pandas as pd
from configs.config import load_config_data
import os
from torch import optim, nn
from torch.utils.data import DataLoader
from dataset.PTE_dataset import PTEDataset
from model.PTE import TransitionPlaceEmbeddingModel

from utils.event_log import EventLogData, split_valid_df
from utils.util import generate_curve
from utils.metric import metric_calculate
from copy import deepcopy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False # 
    torch.backends.cudnn.deterministic = True
    
# Test the test data(val data)
def test_model(test_dataset, model, batch_size, device):
    predictions_list = [] 
    true_list = []
    length_list = []
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    with torch.no_grad():
        model.eval()
        for seq, targets in test_dataloader:
            batch_data = seq.to(device)
            logits = model(batch_data)
            predictions = torch.argmax(logits, dim=1)
            true_list.extend(targets.cpu().numpy().tolist())
            predictions_list.extend((predictions.cpu().numpy()+1).tolist())
            lengths = torch.sum((seq[:, 0]) != 0, dim=1)
            length_list.extend(lengths.tolist())
    
    return true_list, predictions_list, length_list

def train_model(train_dataset, val_dataset, model_parameters, device, trial=None):
    
    print("************* Training Model ***************")
    
    train_dataloader = DataLoader(train_dataset, batch_size=model_parameters['batch_size'])
    
    model = TransitionPlaceEmbeddingModel(
            transition_num=model_parameters['activity_num'],
            dimension=model_parameters['dimension'],
            dropout=model_parameters['dropout'],
            beta=model_parameters['beta']).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=model_parameters['learning_rate'], weight_decay=1e-2)
    crossentropy = nn.CrossEntropyLoss()
    
    train_loss_plt = []
    train_accuracy_plt = []
    val_accuracy_plt = []
    
    best_val_fscore = 0
    best_val_accurace = 0
    patience_count = 0
    max_patience_num = model_parameters['max_patience_num']
    
    # Train Model
    for epoch in range(model_parameters['num_epochs']):
        model.train()
        predictions_list = [] 
        true_list = []
        training_loss = 0
        num_train = 0
        
        for seq, targets in train_dataloader:
            batch_data = seq.to(device)
            logits = model(batch_data)
            loss = crossentropy(logits, targets.to(device)-1) # nn.crossentropy involves softmax

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            true_list.extend(targets.tolist())
            predictions_list.extend((torch.argmax(logits, dim=1).cpu().numpy()+1).tolist())
            num_train += 1
            training_loss += loss.item()
        
        train_loss_plt.append(training_loss/num_train)
        train_accurace,  _, _, train_fscore= metric_calculate(true_list, predictions_list)
        train_accuracy_plt.append(train_accurace)

        # test the accurace in val dataset
        val_truth_list, val_prediction_list, _ = test_model(val_dataset, model, model_parameters['batch_size'], device)
        val_accurace,  _, _, val_fscore= metric_calculate(val_truth_list, val_prediction_list)
        val_accuracy_plt.append(val_accurace)
        print(f"epoch: {epoch}, train_loss:{training_loss/num_train}, train_accurace:{train_accurace}, val_accurace:{val_accurace}, train_fscore:{train_fscore}, val_fscore:{val_fscore}")
        
        # Early Stop
        if epoch == 0 or val_accurace >= best_val_accurace:
           best_val_accurace =  val_accurace
           patience_count = 0
           best_model_dict = model.state_dict()
        else:
            patience_count += 1
        
        if patience_count == max_patience_num:
            break
            
        if trial:
            # Report intermediate objective value.
            trial.report(best_val_fscore, epoch)
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
        
    print(f"best val_accurace:{best_val_accurace} ")
    return best_model_dict, best_val_accurace, train_loss_plt, train_accuracy_plt, val_accuracy_plt

if __name__ == "__main__":
    
    # load the model config
    cfg_model_train = load_config_data("configs/PTE_Model.yaml")
    
    setup_seed(cfg_model_train['seed'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset_cfg = cfg_model_train['data_parameters']
    model_cfg = cfg_model_train['model_parameters']
    
    data_path = '{}/{}/time-process/'.format(dataset_cfg['data_path'], dataset_cfg['dataset'])
    save_folder = 'results/{}/{}/'.format(model_cfg['model_name'], dataset_cfg['dataset'])
    
    os.makedirs(save_folder + f'/result', exist_ok=True)
    os.makedirs(save_folder + f'/model', exist_ok=True)
    os.makedirs(save_folder + f'/curves', exist_ok=True)
    
    print("************* Training in different k-fold dataset ***************")

    train_file_name = data_path + 'train.csv'   
    
    train_df = pd.read_csv(train_file_name)

    event_log = EventLogData(train_df)
    train_df, val_df = split_valid_df(train_df, dataset_cfg['valid_ratio'])
    train_data_list = event_log.generate_data_for_input(train_df)
    val_data_list = event_log.generate_data_for_input(val_df)
    
    max_len = event_log.max_len 
    time_feature_dict = event_log.time_feature
    train_dataset = PTEDataset(train_data_list, max_len, time_feature_dict, shuffle=True)
    val_dataset = PTEDataset(val_data_list, max_len, time_feature_dict, shuffle=False)

    model_cfg['activity_num'] = len(event_log.activity2id)
    best_model_dict, best_val_accurace, train_loss_plt, train_accuracy_plt, val_accuracy_plt = train_model(train_dataset, val_dataset, model_cfg, device)

    # print the loss and accurace curve
    generate_curve(save_folder + f'/curves/curve.jpg', train_loss_plt, train_accuracy_plt, val_accuracy_plt)
    
    with open( f'{save_folder}/model/best_model.pth', 'wb') as fout:
        torch.save(best_model_dict, fout)
    
