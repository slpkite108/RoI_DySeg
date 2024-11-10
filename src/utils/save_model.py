import os

from easydict import EasyDict
from accelerate import Accelerator

def save_model(configs, modelParameter:EasyDict, accelerator:Accelerator, epoch):
    if modelParameter.mean_acc > modelParameter.best_acc:
        accelerator.save_state(
            output_dir=os.path.join(f'{configs.run.work_dir}',f'{configs.run.checkpoint}','train','model_store','best'),
            safe_serialization = False
        )
        modelParameter.best_acc = modelParameter.mean_acc
        modelParameter.best_epoch = epoch
        
    if (epoch+1) % max(configs.train.save_cycle,1)==0 or (epoch+1)==configs.train.scheduler.max_epochs:
        accelerator.save_state(
            output_dir=os.path.join(f'{configs.run.work_dir}',f'{configs.run.checkpoint}','train','model_store',f'epoch_{epoch+1:05d}'),
            safe_serialization = False
        )