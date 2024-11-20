from src.save_utils import save_results_csv
from pathlib import Path

import torch

experiment_name = "test"

work_dir = Path(".")

# class Model(torch.nn.Parameter):
#     def __init__(self):
#         super().__init__()
#         self.param = torch.nn.Parameter(torch.ones(5))

#     def forward():
#         ...

# model = Model()

# print(model.param)
# model.param = torch.nn.Parameter(torch.zeros(5))
# compiled_model = torch.compile(model)

# torch.save(model, "./test_model.pt")

model_loaded = torch.load("results/Dalby/mf_static_signals/models/Dalby_test_size=0.1_num_unobs=1_rs=0_data_idx=5.pt")
for name, param in model_loaded.named_parameters():
    print(name, param)
# results_dict = {"hello":"world"}
# save_results_csv(file_name= f'{experiment_name}.csv', 
#                     path= work_dir / f'{experiment_name}.csv', 
#                     results_dict=results_dict)