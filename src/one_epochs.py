import torch
import torchio as tio

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
        # scaler.scale(seg_loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.gradient_clip)
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
            
        progress_bar.set_postfix(TrainSegLoss = seg_loss)
        step += 1
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
    
    scheduler.step(epoch)
    
    metric = {}
    for metric_name in metric_list:
        batch_acc = metric_list[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    logger.info(f"Epoch [{epoch + 1}] metric {metric}\n")
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
        
        #with autocast():
        seg_logits = model(image)
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
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Val/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    logger.info(f"Epoch [{epoch + 1}] metric {metric}\n\n")

    mean_acc = metric['Val/mean dice_metric']
    
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
def gen_one_epoch(model, gen_loader, path , ext , logger):
    model.eval()

    progress_bar = tqdm(range(len(gen_loader)), leave=False)
    for i, image_batch in enumerate(gen_loader):
        progress_bar.set_description(f'Generation ')
        image = image_batch['image']
        
        seg_logits = model(image)

        seg_logits = (seg_logits>0.5).float()


        progress_bar.set_postfix()
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
    logger.info(f'finished')
    return