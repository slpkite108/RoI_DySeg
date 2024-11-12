import torch
import numpy as np

from monai.transforms import SaveImage
from objprint import objstr
import os
from tqdm import tqdm
from torch.cuda.amp import autocast
from datetime import datetime, timedelta

def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_list, metric_list, post_transform, epoch, step, accelerator, logger):
    model.train()
    
    progress_bar = tqdm(range(len(train_loader)), leave=False)
    for i, image_batch in enumerate(train_loader):
        progress_bar.set_description(f'Train Epoch [{epoch+1}] ')
        optimizer.zero_grad()
        image = image_batch['image']
        label = image_batch['label']
        #with autocast():
        seg_logits = model(image)
        seg_loss = torch.stack([loss_list[loss_name](seg_logits, label) for loss_name in loss_list]).sum()

        accelerator.backward(seg_loss)

        optimizer.step()
        accelerator.log(
            {
                "Train/Segmentor Loss": float(seg_loss),
            },
            step=step,
        )
        
        seg_logits = [post_transform(i) for i in seg_logits]
        for metric_name in metric_list:
            metric_list[metric_name](y_pred=seg_logits, y=label)
            
        progress_bar.set_postfix(TrainSegLoss = seg_loss, Current_LR=scheduler.get_last_lr())
        step += 1
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
    
    scheduler.step(epoch)
    
    metric = {}
    for metric_name in metric_list:
        batch_acc = metric_list[metric_name].aggregate()
        metric_list[metric_name].reset()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    logger.info(f"Train Epoch [{epoch + 1}] metric {metric}\n")
    for lab in range(len(batch_acc)):
        metric.update({
            f"Train/label{lab+1} {metric_name}": float(batch_acc[lab])
        })

    train_loader.dataset.update_cache()
    
    return step

@torch.no_grad()
def val_one_epoch(model, val_loader, loss_list, metric_list, post_transform, epoch, step, accelerator, logger): #use model.val_one_epoch
    model.eval()

    progress_bar = tqdm(range(len(val_loader)), leave=False)
    for i, image_batch in enumerate(val_loader):
        progress_bar.set_description(f'Validation Epoch [{epoch+1}] ')
        image = image_batch['image']
        label = image_batch['label']
        
        seg_logits = model(image)
        
        #with autocast():
        seg_loss = torch.stack([loss_list[loss_name](seg_logits, label) for loss_name in loss_list]).sum()
        
        accelerator.log(
            {
                "Val/Segmentor Loss": float(seg_loss),
            },
            step=step,
        )
        
        seg_logits = [post_transform(i) for i in seg_logits]
        for metric_name in metric_list:
            metric_list[metric_name](y_pred=seg_logits, y=label)

        progress_bar.set_postfix(ValSegLoss = seg_loss.item())
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
            
    metric = {}
    for metric_name in metric_list:
        batch_acc = metric_list[metric_name].aggregate()
        metric_list[metric_name].reset()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Val/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    logger.info(f"Validation Epoch [{epoch + 1}] metric {metric}\n\n")

    mean_acc = metric['Val/mean DiceMetric']
    
    return step, mean_acc, batch_acc

@torch.no_grad()
def test_one_epoch(model, test_loader, loss_list, metric_list,post_transform, accelerator, logger):
    model.eval()

    mean_model_time = timedelta()
    progress_bar = tqdm(range(len(test_loader)), leave=False)
    for i, image_batch in enumerate(test_loader):
        progress_bar.set_description(f'Inference ')
        image = image_batch['image']
        label = image_batch['label']
        
        model_start = datetime.now()
        seg_logits = model(image)
        model_end = datetime.now()
        mean_model_time += (model_end - model_start)
        
        seg_loss = torch.stack([loss_list[loss_name](seg_logits, label) for loss_name in loss_list]).sum()
        
        seg_logits = [post_transform(i) for i in seg_logits]
        
        for metric_name in metric_list:
            metric_list[metric_name](y_pred=seg_logits, y=label)
        progress_bar.set_postfix(ValSegLoss = seg_loss.item())
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
    
    mean_model_time /= len(test_loader)
    
    metric = {}
    for metric_name in metric_list:
        batch_acc = metric_list[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Test/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    return metric, mean_model_time

@torch.no_grad()
def gen_one_epoch(model, path, gen_loader, post_transform , ext , accelerator, logger):
    model.eval()
    assert ext in ['.nii.gz', '.nii']
    path = os.path.join(path, f'{ext}_files')
    
    img_tr = SaveImage(
        output_dir=path,  # 파일을 저장할 디렉토리
        output_ext=ext,      # 파일 확장자
        resample=False,            # 필요한 경우 재샘플링 여부
        separate_folder=False,      # 각 이미지마다 개별 폴더에 저장할지 여부
        print_log=False,
        output_postfix='img',
    )
    gen_tr = SaveImage(
        output_dir=path,  # 파일을 저장할 디렉토리
        output_ext=ext,      # 파일 확장자
        resample=False,            # 필요한 경우 재샘플링 여부
        separate_folder=False,      # 각 이미지마다 개별 폴더에 저장할지 여부
        print_log=False,
        output_postfix='gen',
    )
    gt_tr = SaveImage(
        output_dir=path,  # 파일을 저장할 디렉토리
        output_ext=ext,      # 파일 확장자
        resample=False,            # 필요한 경우 재샘플링 여부
        separate_folder=False,      # 각 이미지마다 개별 폴더에 저장할지 여부
        print_log=False,
        output_postfix='gt',
    )
    
    progress_bar = tqdm(range(len(gen_loader)), leave=False)
    for i, image_batch in enumerate(gen_loader):
        progress_bar.set_description(f'Generation ')
        image = image_batch['image'] # b, c, w, h, d
        image_meta = image_batch['image_meta_dict']
        label = image_batch['label']
        label_meta = image_batch['label_meta_dict']

        seg_logits = model(image)

        seg_logits = post_transform(seg_logits)
        
        # img = image.unsqueeze(0).unsqueeze(0).cpu().numpy()
        # gen = seg_logits.unsqueeze(0).unsqueeze(0).cpu().numpy()
        # gt = label.unsqueeze(0).unsqueeze(0).cpu().numpy()
        #batch == 1
        img_tr(
            image.squeeze(0),
            meta_data=image_meta,
        )
        gen_tr(
            seg_logits.squeeze(0),
            meta_data=label_meta,
        )
        gt_tr(
            label.squeeze(0),
            meta_data=label_meta,
        )

        progress_bar.set_postfix(GPUMemory = torch.cuda.memory_allocated() / (1024 ** 2) )
        progress_bar.update(1)

    progress_bar.clear()
    logger.info(progress_bar)
    logger.info(f'finished')
    return

def custom_name_formatter(meta_dict):
    # meta_dict에서 필요한 정보를 추출
    original_filename = meta_dict.get("filename_or_obj", "output")
    # 인덱스를 meta_dict에서 가져오거나 기본값 설정
    index = meta_dict.get("index", 0)
    # 새로운 파일명 생성
    new_filename = f"{original_filename}_idx{index}"
    return new_filename