import pandas as pd
import torch
from pathlib import Path
import os

def save_results_csv(file_name, path, results_dict):
    """save results in csv file"""

    results_df = pd.DataFrame(results_dict, index = [0])

    if not path.exists():
        os.makedirs(path.parent, exist_ok = True)
        results_df.to_csv(path, index = False)
    else:
        df = pd.read_csv(path)
        df = pd.concat([df, results_df])
        df.to_csv(path, index = False)


def save_parameters(path, file_name, model):
    """save model parameters"""

    os.makedirs(path, exist_ok = True)

    torch.save(model.state_dict(), path / file_name)



def save_grads(path, file_appendix, model_dict, training_dataset_dict, test_dataset_dict):
    """save model parameter gradients"""

    os.makedirs(path, exist_ok = True)

    file_name = f'model_fold_{file_appendix}'
    torch.save(model_dict, path / file_name)

    if training_dataset_dict is not None:

        file_name = f'training_dataset_{file_appendix}'
        torch.save(training_dataset_dict, path / file_name)
    
    if test_dataset_dict is not None:
        file_name = f'test_dataset_{file_appendix}'
        torch.save(test_dataset_dict, path / file_name)

def save_elbos(path, file_appendix, elbos):
    """save elbos in csv file"""

    os.makedirs(path, exist_ok = True)

    torch.save(elbos, path / f'elbo_{file_appendix}')