from src.save_utils import save_results_csv
from pathlib import Path

experiment_name = "test"

work_dir = Path(".")

results_dict = {"hello":"world"}
save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / f'{experiment_name}.csv', 
                    results_dict=results_dict)