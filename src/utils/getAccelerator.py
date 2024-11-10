import os
from datetime import datetime
from accelerate import Accelerator

from typing import Any

def getAccelerator(work_dir, checkpoint, mode='train',cpu:bool=False, log_with:list=['tensorboard'], device_placement=True):
    logging_dir = os.path.join(f'{work_dir}',f'{checkpoint}','{mode}','logs',f'{str(datetime.now())}')
    
    accelerator = Accelerator(
        cpu=False,
        log_with=['tensorboard'],
        project_dir=logging_dir.format(mode=mode),
        device_placement='cuda:1',
    )
    #accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.init_trackers('tensorboard')
    
    return accelerator
