from pathlib import Path

import torch

experiment_name = "test"

work_dir = Path(".")


dataset = 0
seed = 0
ts = 0.1



# path_dict = {"Dalby": }

model_loaded = torch.load("results/Dalby/struc_var_signals/models/Dalby_test_size=0.5_c_unobs=1_L=4_A=2_P=8_rs=6_data_idx=2.pt")
# for name, param in model_loaded.named_parameters():
#     print(name, param)
print(model_loaded)
