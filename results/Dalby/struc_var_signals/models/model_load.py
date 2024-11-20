from pathlib import Path

import torch

experiment_name = "test"

work_dir = Path(".")


dataset = 0
seed = 0
ts = 0.1



# path_dict = {"Dalby": }

model_loaded = torch.load("results/Dalby/mf_static_signals/models/Dalby_test_size=0.1_num_unobs=1_rs=0_data_idx=5.pt")
# for name, param in model_loaded.named_parameters():
#     print(name, param)
print(model_loaded)
