import optuna
from optuna.visualization import plot_optimization_history
from optuna.samplers import TPESampler
import gc
from utils.event_log import EventLogData
import os
from train_pte import setup_seed
from configs.config import load_config_data
from utils.parameter_search import objective


if __name__ == "__main__":
    
    model_name = 'PTE'
    # load the model config
    if model_name=='PTE':
        cfg_model_train = load_config_data("configs/PTE_Model.yaml")
    else:
        raise Exception("This Model Don't exit")

    # Fixed random number seed
    setup_seed(cfg_model_train['seed']) 
    
    dataset_cfg = cfg_model_train['data_parameters']
    model_cfg = cfg_model_train['model_parameters']
    cfg_model_train['device_id'] = '0'
    
    data_path = '{}/{}/process/'.format(dataset_cfg['data_path'], dataset_cfg['dataset'])
    save_folder = 'results_kfold_{}/{}/{}/'.format(dataset_cfg['k_fold_num'], model_cfg['model_name'], dataset_cfg['dataset'])
    
    os.makedirs(f'{save_folder}/optimize', exist_ok=True)
    os.makedirs(f'{save_folder}/model', exist_ok=True)
    os.makedirs(f'{save_folder}/result', exist_ok=True)
    
    for idx in range(dataset_cfg['k_fold_num']):
        # record optimization
        record_file = open(f'{save_folder}/optimize/opt_history_{idx}.txt', 'w')
        record_file.write("tid,score,dimension,dropout,beta,duartime")
        record_file.close()
        
        train_file_name = data_path + '/kfoldcv_' + str(idx) + '_train.csv' 
        train_log = EventLogData(train_file_name)
        train_data_list, val_data_list = train_log.split_valid_data(dataset_cfg['valid_ratio'])
        
        cfg_model_train['activity_num'] = len(train_log.activity2id)
        print(f"seed: {cfg_model_train['seed']}")
        print(f"fold: {idx+1}, dataset: {dataset_cfg['dataset']}, train size: {len(train_data_list)}, valid size:{len(val_data_list)}")
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=cfg_model_train['seed'])) # fixed parameter
        study.optimize(lambda trial: objective(trial, idx, cfg_model_train, train_data_list, val_data_list, save_folder), n_trials=20, gc_after_trial=True, callbacks=[lambda study, trial:gc.collect()])
        
        # record optimization history
        history = optuna.visualization.plot_optimization_history(study)
        plot_optimization_history(study).write_image(f"{save_folder}/optimize/opt_history_{idx}.png")
        
        outfile = open(f'{save_folder}/model/best_model_kfd{idx}.txt', 'w')
        best_params = study.best_params
        best_accurace = study.best_value

        print("Best hyperparameters:", best_params)
        print("Best accurace:", best_accurace)
        
        outfile.write('Best trail:' + str(study.best_trial.number))
        outfile.write('\nBest hyperparameters:' + str(best_params))
        outfile.write('\nBest accurace:' + str(best_accurace))
        outfile.close()
        