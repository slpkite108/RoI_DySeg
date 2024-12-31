import os
import torch

from easydict import EasyDict
from timm.optim import optim_factory
from objprint import objstr
from monai import transforms

from src import utils, loader
from src.one_epochs import gen_one_epoch
# from src.det_one_epochs import det_gen_one_epoch
from src.det_one_merged import det_gen_one_epoch
from src.comb_one_epochs import comb_gen_one_epoch
from src.model import getModel




def generation(configs):

#region Prepare train
    current_pid = os.getpid()
    accelerator = utils.getAccelerator(configs.run.work_dir,configs.run.checkpoint, mode='generation')
    logger = utils.getLogger(accelerator, mode='generation')
    configs.run.loader.batch_size=1
    utils.same_seeds(50)
    
    torch.cuda.set_per_process_memory_fraction(0.01) ## 중요
#endregion

    with utils.Profiler(configs.run.device_num, current_pid, interval=0.1) as profiler:
        
    #region setup model
        if configs.run.model.type in ['comb']:
            detecter = getModel(configs.run.model.detecter.name, configs.run.model.detecter.args)
            detecter = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.det_check, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model.bin'), detecter, accelerator).to('cuda')
            
            segmenter = getModel(configs.run.model.segmenter.name, configs.run.model.segmenter.args)
            if os.path.exists(os.path.join(configs.run.work_dir, configs.run.checkpoint, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model_1.bin')):
                segmenter = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.checkpoint, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model_1.bin'), segmenter, accelerator).to('cuda')
            else:
                segmenter = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.seg_check, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model.bin'), segmenter, accelerator).to('cuda')
        else:
            model = getModel(configs.run.model.name, configs.run.model.args)
            model = utils.load_pretrain_model(os.path.join(configs.run.work_dir, configs.run.checkpoint, configs.inference.weight_path, 'model_store', configs.inference.epoch, 'pytorch_model.bin'), model, accelerator)
    #endregion
    #region Prepare Parameters
        post_transform = transforms.Compose(
            [
                transforms.Activations(sigmoid=True),
                transforms.AsDiscrete(threshold=0.5),
            ]
        )
        if configs.run.model.type in ['comb']:
            det_loader = loader.get_loader(configs, type='det', shuffle=False, mode='generation')
            seg_loader = loader.get_loader(configs, type='seg', shuffle=False, mode='generation')
            
            detecter, segmenter, det_loader, seg_loader = accelerator.prepare(
                detecter, segmenter, det_loader, seg_loader
            )
            
            model = (detecter, segmenter)
            gen_loader = (det_loader, seg_loader)
            
            logger.info(f"generation datas : {len(gen_loader[1].dataset)}")
        else:
            gen_loader = loader.get_loader(configs,type=configs.run.model.type, mode='generation')
            
            model, gen_loader = accelerator.prepare(
                model, gen_loader
            )
            logger.info(f"generation datas : {len(gen_loader.dataset)}")
    #endregion

        logger.info(objstr(configs))
        logger.info(f'memory: {torch.cuda.memory_allocated() / (1024 ** 2)}Mib')
        
    
#region inference
        path = os.path.join(configs.run.work_dir, configs.run.checkpoint, 'generation')
        if configs.run.model.type == 'seg':
            ext_list = ['.nii.gz', '.nii']
            gen_one_epoch(model, path, gen_loader, post_transform, ext_list[0] ,accelerator, logger)
        elif configs.run.model.type in ['comb']:
            ext_list = ['.nii.gz', '.nii']
            comb_gen_one_epoch(model, path, gen_loader, post_transform, ext_list[0] ,accelerator, logger)
        else:
            ext_list = ['.png']
            det_gen_one_epoch(model, path, gen_loader, post_transform, ext_list[0] ,accelerator, logger)
        
    logger.info(f'GPU_Memory : {objstr(profiler.get_statistics())}')
    logger.info(f'train_runtime : {profiler.get_runtime()}')
        
#endregion