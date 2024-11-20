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






def save_model(path, experiment_name, seed, data_idx, model):
    """save model parameter gradients"""

    os.makedirs(path, exist_ok = True)

    file_name = f'{experiment_name}_rs={seed}_data_idx={data_idx}.pt'
    torch.save(model, path / file_name)

    

def save_elbos(path, file_appendix, elbos):
    """save elbos in csv file"""

    os.makedirs(path, exist_ok = True)

    torch.save(elbos, path / f'elbo_{file_appendix}')