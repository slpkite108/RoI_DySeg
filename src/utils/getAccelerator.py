import os
from datetime import datetime
from accelerate import Accelerator

from typing import Any

def getAccelerator(work_dir, checkpoint, cpu:bool=False, log_with:list=['tensorboard'], device_placement:bool=True):
    logging_dir = os.path.join(f'{work_dir}',f'{checkpoint}','{mode}','logs',f'{str(datetime.now())}')
    
    accelerator = Accelerator(
        cpu=False,
        log_with=['tensorboard'],
        project_dir=logging_dir.format(mode='train'),
        device_placement=True,
    )
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    
    return accelerator
