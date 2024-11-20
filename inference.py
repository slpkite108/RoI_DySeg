import os
import torch

from easydict import EasyDict
from timm.optim import optim_factory
from objprint import objstr
from monai import losses, metrics, transforms
from src.losses.DETR_Criterion import DETR_Criterion
from src import utils, loader
from src.one_epochs import test_one_epoch
from src.det_one_epochs import det_test_one_epoch
from src.model import getModel
from src.optimizer import LinearWarmupCosineAnnealingLR




def inference(configs):

#region Prepare train
    current_pid = os.getpid()
    accelerator = utils.getAccelerator(configs.run.work_dir,configs.run.checkpoint, mode='inference')
    logger = utils.getLogger(accelerator, mode='inference')
    configs.run.loader.batch_size=1
    utils.same_seeds(50)
#endregion

    with utils.Profiler(configs.run.device_num, current_pid, interval=0.1) as profiler:
        
    #region setup model
        model = getModel(configs.run.model.name, **configs.run.model.args)
        model = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.checkpoint, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model.bin'), model, accelerator)
    #endregion
        
    #region Prepare Parameters
        if configs.run.model.type in ['seg']:
            loss_list = {
                "DiceLoss": losses.DiceLoss(**configs.train.loss.DiceLoss),
                "FocalLoss": losses.FocalLoss(**configs.train.loss.FocalLoss),
            }
        else:
            loss_list = {
                "DETR_Criterion": DETR_Criterion(**configs.train.loss.DETR_Criterion)
            }
        
        metric_list = {
            'DiceMetric': metrics.DiceMetric(**configs.train.metrics.DiceMetric),
            "MeanIoU": metrics.MeanIoU(**configs.train.metrics.MeanIoU),
            'HausdorffDistanceMetric': metrics.HausdorffDistanceMetric(**configs.train.metrics.HausdorffDistanceMetric),
        }
        
        post_transform = transforms.Compose(
            [
                transforms.Activations(sigmoid=True),
                transforms.AsDiscrete(threshold=0.5),
            ]
        )
        
        # train_loader = loader.get_loader(configs, mode='train')
        # val_loader = loader.get_loader(configs, mode='validation')
        test_loader = loader.get_loader(configs, mode='inference')
        
        model, test_loader = accelerator.prepare(
            model, test_loader
        )
        
    #endregion

        logger.info(objstr(configs))
        logger.info(f"inference datas : {len(test_loader.dataset)}")
    
#region inference
        if configs.run.model.type == 'seg':
            metric, mean_model_time = test_one_epoch(model, test_loader, loss_list, metric_list, post_transform, accelerator, logger)
        else:
            metric, mean_model_time = det_test_one_epoch(model, test_loader, loss_list, metric_list, post_transform, accelerator, logger)
        
    logger.info(f'Results : {metric}')
    logger.info(f"mean Model time: {mean_model_time}")
    logger.info(f'GPU_Memory : {objstr(profiler.get_statistics())}')
    logger.info(f'train_runtime : {profiler.get_runtime()}')
        
#endregion