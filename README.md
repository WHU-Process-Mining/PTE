# A Consumption-Generation Driven Place-Transition Embedding Framework for Predictive Process Monitoring

## Installation

1. Create a python environment

    ```bash
    conda create -n PTE python=3.8.0
    conda activate PTE 
    ```

2. Install pytorch

    Following the official website's guidance (<https://pytorch.org/get-started/locally/>), install the corresponding PyTorch version based on your CUDA version. For our experiments, we use torch 1.12.1+cu116. The installation command is as follows:

    ```bash
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
    ```

3. Install other related dependencies

    ```bash
    pip install -r requirements.txt
    ```

## Train
First, you need to specify data_path and dataset in configs/PTE_Model.yaml. 

Here, Two training methods are provided here:

1. Specify Hyperparameters:
    Specify model_parameters in   configs/PTE_Model.yaml.
    
2. Use Optuna for Hyperparameter Optimization:
    ```bash
    python run_pte.py
    ```
## Test
    python test_pte.py

