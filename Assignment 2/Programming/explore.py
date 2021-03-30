#!/usr/bin/python3
## file: explore.py
import sys
import subprocess
import os
import shutil
import warnings
import numpy as np

from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class Arguments:
  # Data
  data_folder: str = './data'
  batch_size: int = 16

  # Model
  model: str = 'lstm'  # [lstm, gpt1]
  embeddings: str = './data/embeddings.npz'
  layers: int = 1

  # Optimization
  optimizer: str = 'adamw'  # [sgd, momentum, adam, adamw]
  epochs: int = 10
  lr: float = 1e-3
  momentum: float = 0.9
  weight_decay: float = 5e-4

  # Experiment
  exp_id: str = 'debug'
  log: bool = True
  log_dir: str = './logs'
  seed: int = 42

  # Miscellaneous
  num_workers: int = 2
  device: str = 'cuda'
  progress_bar: bool = False
  print_every: int = 10
    



# Note: if there is any discrepency with the configurations in run_exp.py, the
# version from run_exp.py should be the ones to use in Problem 3.
configs = {
  1: Arguments(model='lstm', layers=1, batch_size=16, log=True, epochs=10, optimizer='adam'),
  2: Arguments(model='lstm', layers=1, batch_size=16, log=True, epochs=10, optimizer='adamw'),
  3: Arguments(model='lstm', layers=1, batch_size=16, log=True, epochs=10, optimizer='sgd'),
  4: Arguments(model='lstm', layers=1, batch_size=16, log=True, epochs=10, optimizer='momentum'),

  5: Arguments(model='gpt1', layers=1, batch_size=16, log=True, epochs=10, optimizer='adam'),
  6: Arguments(model='gpt1', layers=1, batch_size=16, log=True, epochs=10, optimizer='adamw'),
  7: Arguments(model='gpt1', layers=1, batch_size=16, log=True, epochs=10, optimizer='sgd'),
  8: Arguments(model='gpt1', layers=1, batch_size=16, log=True, epochs=10, optimizer='momentum'),

  9: Arguments(model='lstm', layers=2, batch_size=16, log=True, epochs=10, optimizer='adamw'),
  10: Arguments(model='lstm', layers=4, batch_size=16, log=True, epochs=10, optimizer='adamw'),
  11: Arguments(model='gpt1', layers=2, batch_size=16, log=True, epochs=10, optimizer='adamw'),
  12: Arguments(model='gpt1', layers=4, batch_size=16, log=True, epochs=10, optimizer='adamw'),
}



for key in configs.keys():
    args = configs[key]
    
    if(key < 10):
        args.exp_id = f"0{key}"
    else:
        args.exp_id = f"{key}"
        
    submit_command = (f"sbatch --export=MODEL={args.model},LAYERS={args.layers},BATCH_SIZE={args.batch_size},EPOCHS={args.epochs},OPTIMIZER={args.optimizer},EXP_ID={args.exp_id} run.sh")
    print(f"EXP_ID={args.exp_id}")
    exit_status = subprocess.call(submit_command, shell=True)
    if exit_status is 1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(submit_command))

print("Done submitting jobs!")





