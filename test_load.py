from src.save_utils import save_results_csv
from pathlib import Path

import torch

experiment_name = "test"

work_dir = Path(".")


dataset = 0
seed = 0
ts = 0.1



# path_dict = {"Dalby": }

model_loaded = torch.load(f"results/Dalby/mf_static_signals/models/Dalby_test_size={ts}_num_unobs=1_rs={seed}_data_idx={dataset}.pt")
for name, param in model_loaded.named_parameters():
    print(name, param)
