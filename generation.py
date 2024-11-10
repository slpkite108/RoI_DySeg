import os
import torch

from easydict import EasyDict
from timm.optim import optim_factory
from objprint import objstr
from monai import transforms

from src import utils, losses, metrics, loader
from src.one_epochs import test_one_epoch
from src.model import getModel
from src.optimizer import LinearWarmupCosineAnnealingLR




def generation(configs):

#region Prepare train
    current_pid = os.getpid()
    accelerator = utils.getAccelerator(configs.run.work_dir,configs.run.checkpoint, mode='generation')
    logger = utils.getLogger(accelerator, mode='generation')
    configs.run.loader.batch_size=1
    utils.same_seeds(50)
#endregion

    with utils.Profiler(configs.run.device_num, current_pid, interval=0.1) as profiler:
        
    #region setup model
        model = getModel(configs.run.model.name, **configs.run.model.args)
        model = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.checkpoint, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model.bin'), model, accelerator)
    #endregion
        
    #region Prepare Parameters
        loss_list = {
            "dice_loss": losses.DiceLoss(**configs.train.loss.dice_loss),
            "focal_loss": losses.FocalLoss(**configs.train.loss.focal_loss),
        }
        
        metric_list = {
            'dice_metric': metrics.DiceMetric(**configs.train.metrics.dice_metric),
            "iou_metric": metrics.MeanIoU(**configs.train.metrics.iou_metric),
            'hd95_metric': metrics.HausdorffDistanceMetric(**configs.train.metrics.hd95_metric),
        }
        
        post_transform = transforms.Compose(
            [
                transforms.Activations(sigmoid=True),
                transforms.AsDiscrete(threshold=0.5),
            ]
        )
        
        # train_loader = loader.get_loader(configs, mode='train')
        # val_loader = loader.get_loader(configs, mode='validation')
        gen_loader = loader.get_loader(configs, mode='generation')
        
        model, gen_loader = accelerator.prepare(
            model, gen_loader
        )
        
    #endregion

        logger.info(objstr(configs))
        logger.info(f"inference datas : {len(test_loader.dataset)}")
    
#region inference

        metrics, mean_model_time = test_one_epoch(model, test_loader, loss_list, metric_list, post_transform, accelerator, logger)
        
    logger.info(f'Results : {metrics}')
    logger.info(f"mean Model time: {mean_model_time}")
    logger.info(f'GPU_Memory : {objstr(profiler.get_statistics())}')
    logger.info(f'train_runtime : {profiler.get_runtime()}')
        
#endregion