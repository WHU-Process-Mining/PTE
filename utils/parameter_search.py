from train_pte import train_model
import torch
import time
import pickle

def PTE_parameters(trial, cfg):
    # define the parameter search space
    model_parameters = {}
    
    # model_parameters['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    model_parameters['dimension'] = trial.suggest_categorical('dimension', [32, 64, 128, 256, 512])
    # model_parameters['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    model_parameters['dropout'] = trial.suggest_float('dropout', 0, 1)
    model_parameters['beta'] = trial.suggest_float('beta', 1e-3, 1e3, log=True)
    
    model_parameters['learning_rate'] = cfg['model_parameters']['learning_rate']
    model_parameters['valid_ratio'] = cfg['data_parameters']['valid_ratio']
    model_parameters['num_epochs'] = cfg['model_parameters']['num_epochs']
    model_parameters['batch_size'] = cfg['model_parameters']['batch_size']
    model_parameters['activity_num'] = cfg['activity_num']
    model_parameters['max_patience_num'] = cfg['model_parameters']['max_patience_num']
    return model_parameters


def objective(trial, cfg_parameters, train_dataset, val_dataset, save_folder): 
    model_name = cfg_parameters['model_parameters']['model_name']
    if model_name=='PTE':
        model_parameters = PTE_parameters(trial, cfg_parameters)
    else:
        raise Exception("This Model Don't exit")
    
    device = 'cuda:'+ cfg_parameters['device_id'] if torch.cuda.is_available() else 'cpu'
     
    start_time = time.time()
    
    best_model_dict, best_val_accurace, _, _, _ = train_model(train_dataset, val_dataset, model_parameters, device, trial)
     
    current_best = trial.study.best_value if trial.number > 0 else 0
    if best_val_accurace > current_best:
        with open( f'{save_folder}/model/best_model.pth', 'wb') as fout:
            torch.save(best_model_dict, fout)

    duartime = time.time() - start_time
   
    record_file = open(f'{save_folder}/optimize/opt_history.txt', 'a')
    record_file.write(f"\n{trial.number},{best_val_accurace},{model_parameters['dimension']},{model_parameters['dropout']},{model_parameters['beta']},{duartime}")
    record_file.close()
    
    return best_val_accurace