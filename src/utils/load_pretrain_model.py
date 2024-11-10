import torch
import torch.nn as nn

from collections import OrderedDict

import sys

from accelerate import Accelerator

def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator, force_load=True):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict)
        accelerator.print(f"Successfully loaded the training model！")
        return model
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Failed to load the training model！")
        if not force_load:
            user_input = input(f"If you want to continue with untrained model? (yes/no)\n").strip()
            
            if not user_input.lower() in['y', 'yes']:
                sys.exit()
        
        return model
    
def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith("http"):
        state_dict = torch.hub.load_state_dict_from_url(
            download_path,
            model_dir=save_path,
            check_hash=check_hash,
            map_location=torch.device("cpu"),
        )
    else:
        state_dict = torch.load(download_path, map_location=torch.device("cpu"))
    return state_dict