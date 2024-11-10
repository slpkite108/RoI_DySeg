import torch
import torchio as tio

from tqdm import tqdm

def train_one_epoch(model, obj,epoch, accelerator):
    progress_bar = tqdm(range(len(obj.train_loader)), leave=False)
    for i, image_batch in enumerate(obj.train_loader):
        image = image_batch['image'][tio.DATA]
        label = image_batch['label'][tio.DATA]
        
        seg_logits = model(image)
        
        seg_loss = obj.loss_functions(seg_logits, label, mode='Train',step=step)
        mean_seg_loss += seg_loss
        accelerator.backward(seg_loss)
        
        obj.optimizer.step()
        obj.optimizer.zero_grad()
        accelerator.log(
            {
                "Train/Segmentor Loss": float(seg_loss),
            },
            step=step,
        )
        
        for metric_name in obj.metrics:
            obj.metrics[metric_name](y_pred=seg_logits, y=image_batch['label'], )
        
        progress_bar.set_postfix(TrainSegLoss = (mean_seg_loss/(i+1)).item())
        step += 1
        progress_bar.update(1)
        
    metric = {}
    for metric_name in obj.metrics:
        batch_acc = obj.metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    accelerator.log(metric, step=epoch)
    pass

@torch.no_grad()
def val_one_epoch(model, obj,epoch, accelerator): #use model.val_one_epoch
    progress_bar = tqdm(range(len(obj.val_loader)), leave=False)
    for i, image_batch in enumerate(obj.val_loader):
        image = image_batch['image'][tio.DATA]
        label = image_batch['label'][tio.DATA]
        seg_logits = model(image)
        
        seg_loss = obj.loss_functions(seg_logits, label, mode='Val',step=step)
        mean_seg_loss += seg_loss
        
        obj.optimizer.step()
        obj.optimizer.zero_grad()
        accelerator.log(
            {
                "Val/Segmentor Loss": float(seg_loss),
            },
            step=step,
        )
        for metric_name in obj.metrics:
            # if hasattr(obj.metrics[metric_name], "mode") and obj.metrics[metric_name].mode == 'collision':
            #     obj.metrics[metric_name](y_pred=seg_logits, y=metric_label, det_bbox = list(det_bbox.chunk(config.trainer.batch_size)), gt_bbox=list(bbox.chunk(config.trainer.batch_size)))
            # else:
            obj.metrics[metric_name](y_pred=seg_logits, y=image_batch['label'])
        
        accelerator.log({"Train/Segmentor Memory":torch.cuda.memory_reserved()/1024/1024}, step=step)
        progress_bar.set_postfix(TrainSegLoss = (mean_seg_loss/(i+1)).item())
        step += 1
        progress_bar.update(1)
    metric = {}
    for metric_name in obj.metrics:
        batch_acc = obj.metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
            }
        )
    accelerator.log(metric, step=epoch)
    pass