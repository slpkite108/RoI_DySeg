import os
import torch

from easydict import EasyDict
from timm.optim import optim_factory
from objprint import objstr
from monai import losses, metrics, transforms
from src.losses import LossCaller
from src import utils, loader
from src.one_epochs import test_one_epoch
# from src.det_one_epochs import det_test_one_epoch
from src.det_one_merged import det_test_one_epoch
from src.comb_one_epochs import comb_test_one_epoch
from src.model import getModel
from src.optimizer import LinearWarmupCosineAnnealingLR

from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis




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
        if configs.run.model.type in ['comb']:
            detecter = getModel(configs.run.model.detecter.name, configs.run.model.detecter.args)
            detecter = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.det_check, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model.bin'), detecter, accelerator).to('cuda')
            flops = FlopCountAnalysis(detecter, torch.randn((1,3,128,128)).to('cuda'))
            print(flops.total())
            # macs, params = get_model_complexity_info(detecter, (3,128,128),as_strings=False,verbose=False)
            # print(macs)
            # print(params)
            
            segmenter = getModel(configs.run.model.segmenter.name, configs.run.model.segmenter.args)
            segmenter = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.seg_check, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model.bin'), segmenter, accelerator).to('cuda')
            flops = FlopCountAnalysis(segmenter, torch.randn((1,1,64,64,64)).to('cuda'))
            print(flops.total())
            # macs, params = get_model_complexity_info(segmenter, (1,32,32,32),as_strings=False,verbose=False)
            # print(macs)
            # print(params)
            exit(0)
            
        else:
            model = getModel(configs.run.model.name, configs.run.model.args)
            model = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.checkpoint, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model.bin'), model, accelerator)
    #endregion
        
    #region Prepare Parameters
    
        if configs.run.model.type in ['comb']:
            det_loss = {}
            for key, val in configs.train.loss.detecter.items():
                det_loss.update({
                    key:LossCaller(key, val)
                })
                
            seg_loss = {}
            for key, val in configs.train.loss.segmenter.items():
                seg_loss.update({
                    key:LossCaller(key, val)
                })
                
            loss_list = (det_loss, seg_loss)
        else:
            loss_list = {}
            for key, val in configs.train.loss.items():
                loss_list.update({
                    key:LossCaller(key, val)
                })
        
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
        if configs.run.model.type in ['comb']:
            det_loader = loader.get_loader(configs, type='det', shuffle=False, mode='inference')
            
            
            seg_loader = loader.get_loader(configs, type='seg', shuffle=False, mode='inference')
            
            detecter, segmenter, det_loader, seg_loader = accelerator.prepare(
                detecter, segmenter, det_loader, seg_loader
            )
            
            # segmenter, seg_loader = accelerator.prepare(
            #     detecter, segmenter, det_loader, seg_loader
            # )
            
            model = (detecter, segmenter)
            test_loader = (det_loader, seg_loader)
            logger.info(objstr(configs))
            logger.info(f"inference datas : {len(test_loader[0].dataset)}")
            
        else:
            test_loader = loader.get_loader(configs, type=configs.run.model.type, mode='inference')
        
            model, test_loader = accelerator.prepare(
                model, test_loader
            )
            logger.info(objstr(configs))
            logger.info(f"inference datas : {len(test_loader.dataset)}")
    #endregion

       
        
#region inference
        if configs.run.model.type in ['comb']:
            det_metric, det_mean_model_time, metric, mean_model_time = comb_test_one_epoch(model, test_loader, loss_list, metric_list, post_transform, accelerator, logger)
            logger.info(f'Det Results : {det_metric}')
            logger.info(f"Det mean Model time: {det_mean_model_time}")
        
        else:
            if configs.run.model.type == 'seg':
                metric, mean_model_time = test_one_epoch(model, test_loader, loss_list, metric_list, post_transform, accelerator, logger)
            else:
                metric, mean_model_time = det_test_one_epoch(model, test_loader, loss_list, metric_list, post_transform, accelerator, logger)
        
        logger.info(f'Results : {metric}')
        logger.info(f"mean Model time: {mean_model_time}")
    logger.info(f'GPU_Memory : {objstr(profiler.get_statistics())}')
    logger.info(f'train_runtime : {profiler.get_runtime()}')
        
#endregion