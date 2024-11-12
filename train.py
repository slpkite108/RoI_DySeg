import os
import torch

from easydict import EasyDict
from timm.optim import optim_factory
from objprint import objstr
from torch.cuda.amp import GradScaler
from monai import losses, metrics, transforms
from src import utils, loader
from src.one_epochs import train_one_epoch, val_one_epoch
from src.model import getModel
from src.optimizer import LinearWarmupCosineAnnealingLR




def train(configs):
    
#region Prepare train
    current_pid = os.getpid()
    accelerator = utils.getAccelerator(configs.run.work_dir,configs.run.checkpoint, mode='train')
    logger = utils.getLogger(accelerator, mode='train')
    utils.same_seeds(50)
    torch.cuda.set_per_process_memory_fraction(0.1)
#endregion

    with utils.Profiler(configs.run.device_num, current_pid, interval=0.1) as profiler:
    #region Prepare Vars
        max_epoch = configs.train.scheduler.max_epochs
        modelParameter = EasyDict({
            "train_step":0,
            "best_epoch":0,
            "val_step":0,
            "starting_epoch":0,
            "best_acc":0,
            "batch_acc":0,
        })
    #endregion
        
    #region setup model
        model = getModel(configs.run.model.name, **configs.run.model.args)
    #endregion
        
    #region Prepare Parameters
        optimizer = optim_factory.create_optimizer_v2(
                    model,
                    **configs.train.optimizer
                )
        scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    **configs.train.scheduler
                )
        
        scaler = GradScaler()
        
        loss_list = {
            "DiceLoss": losses.DiceLoss(**configs.train.loss.DiceLoss),
            "FocalLoss": losses.FocalLoss(**configs.train.loss.FocalLoss),
        }
        
        metric_list = {
            'DiceMetric': metrics.DiceMetric(**configs.train.metrics.DiceMetric),
            "MeanIoU": metrics.MeanIoU(**configs.train.metrics.MeanIoU),
            #'HausdorffDistanceMetric': metrics.HausdorffDistanceMetric(**configs.train.metrics.hd95_metric),
        }
        
        post_transform = transforms.Compose(
            [
                transforms.Activations(sigmoid=True),
                transforms.AsDiscrete(threshold=0.5),
            ]
        )
        
        train_loader = loader.get_loader(configs, mode='train')
        val_loader = loader.get_loader(configs, mode='validation')
        
        model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader
        )
    #endregion
        
        logger.info(objstr(configs))
        logger.info(f"train datas : {len(train_loader.dataset)}")
        logger.info(f"validation datas : {len(val_loader.dataset)}")
#region train epochs     
        for epoch in range(max_epoch):
            modelParameter.train_step = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_list, metric_list, post_transform, epoch, modelParameter.train_step, accelerator, logger)
            torch.cuda.empty_cache()
            
            modelParameter.val_step, modelParameter.mean_acc, modelParameter.batch_acc = val_one_epoch(model, val_loader, loss_list, metric_list, post_transform, epoch, modelParameter.val_step, accelerator, logger)
            torch.cuda.empty_cache()
            
            logger.info(
                f"Epoch [{epoch + 1}/{max_epoch}] segmentation lr = {scheduler.get_last_lr()} best acc: {modelParameter.best_acc}, mean acc: {modelParameter.mean_acc}, mean class: {modelParameter.batch_acc}, best epoch: {modelParameter.best_epoch+1}\n"
            )
            
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
        train_loader.dataset.shutdown()
        val_loader.dataset.shutdown()
    train_memory_dict = profiler.get_statistics()
    logger.info(f'GPU_Memory : {objstr(train_memory_dict)}')
    logger.info(f'train_runtime : {profiler.get_runtime()}')
        
#endregion