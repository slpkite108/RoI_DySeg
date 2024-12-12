import torch
import numpy as np
from src.utils import box_ops
from monai.data import MetaTensor
from monai.transforms import SaveImage, Resize, Pad
from objprint import objstr
import torch.nn.functional as F
import os
from collections import defaultdict
from tqdm import tqdm
from torch.cuda.amp import autocast
from datetime import datetime, timedelta

# det -> seg



def train_one_epoch(model, train_loader, optimizer, scheduler, loss_list, metric_list, post_transform, epoch, step, accelerator, logger):
    model.train()
    
    progress_bar = tqdm(range(len(train_loader)), leave=False)
    for i, image_batch in enumerate(train_loader):
        progress_bar.set_description(f'Train Epoch [{epoch+1}] ')
        optimizer.zero_grad()
        image = image_batch['image']
        label = image_batch['label']
        #bbox = image_batch['bbox']
        #roi = image_batch['roi']

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
        step += 1
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
def comb_test_one_epoch(model, test_loader, loss_list, metric_list, post_transform, accelerator, logger):
    bbox_list = []
    
    detector, segmenter = model
    det_loader, seg_loader = test_loader
    det_loss_list, seg_loss_list = loss_list
    seg_metric_list = metric_list
    
    detector.eval()
    
    for det_loss_name in det_loss_list:
        det_loss_list[det_loss_name].eval()
    
    grouped_dict = defaultdict(list)
    det_mean_model_time = timedelta()
    progress_bar = tqdm(range(len(det_loader)), leave=False)
    pp = PostProcess()
    for i, image_batch in enumerate(det_loader):
        progress_bar.set_description(f'Inference ')
        x,y,z = image_batch['x'],image_batch['y'],image_batch['z']
        bbox = image_batch['bbox']
        
        bbox_x, bbox_y, bbox_z = box_ops.box_xyxy_to_cxcywh(bbox[:,:,(1,2,4,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,2,3,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,1,3,4)]/128)
        origin_shape_x, origin_shape_y, origin_shape_z = image_batch['x_meta_dict']['spatial_shape'],image_batch['y_meta_dict']['spatial_shape'],image_batch['z_meta_dict']['spatial_shape']
        volume_box = []
        for image, box, origin_shape in zip([x,y,z],[bbox_x,bbox_y,bbox_z], [origin_shape_x, origin_shape_y, origin_shape_z]):
            model_start = datetime.now()
            seg_logits = detector(image)
            model_end = datetime.now()
            det_mean_model_time += (model_end - model_start)
            
            label = [{'labels': torch.tensor([0], device='cuda:0'), 'boxes': b} for b in box]
            seg_loss, det_metric = zip(*[det_loss_list[loss_name](seg_logits, label) for loss_name in det_loss_list])
            seg_loss, det_metric = seg_loss[0].sum(), det_metric[0]
            
            seg_logits = pp(seg_logits, torch.tensor([image.shape[2], image.shape[3]],device='cuda').unsqueeze(0))[0]
            _, max_indices = torch.max(seg_logits['scores'], dim=0)
            seg_logits = seg_logits['boxes'][max_indices]
            
            grouped_dict[f'Test/det/mean iou'].append(box_ops.box_iou(seg_logits.unsqueeze(0), box_ops.box_cxcywh_to_xyxy(label[0]['boxes'])*128)[0])
            volume_box.append(seg_logits.to(device='cpu'))

            
            for key, value in det_metric.items():
                grouped_dict[f'Test/det/mean {key}'].append(value)
            progress_bar.set_postfix(ValSegLoss = seg_loss.item())
            progress_bar.update(1)

        bbox_x = volume_box[0]
        bbox_y = volume_box[1]
        bbox_z = volume_box[2]

        # 각 list의 요소들을 하나씩 조합하여 bbox 복구
        bbox_list.append(
            torch.stack([
                (bbox_y[0] + bbox_z[0]) / 2,  # x1 (중복된 값 평균)
                (bbox_x[0] + bbox_z[1]) / 2,  # y1 (중복된 값 평균)
                (bbox_x[1] + bbox_y[1]) / 2,  # z1 (중복된 값 평균)
                (bbox_y[2] + bbox_z[2]) / 2,  # x2 (중복된 값 평균)
                (bbox_x[2] + bbox_z[2]) / 2,  # y2 (중복된 값 평균)
                (bbox_x[3] + bbox_y[3]) / 2,  # z2 (중복된 값 평균)
            ], dim=-1).unsqueeze(0)
        )
    progress_bar.clear()
    logger.info(progress_bar)
    
    det_mean_model_time /= len(det_loader)
    
    det_metric = {key: sum(values) / len(values) for key, values in grouped_dict.items()}
    logger.info(f"Inference det metric {det_metric}\n\n")
    
    for data, box in zip(seg_loader.dataset.data, bbox_list):
        data.update({'bbox':box})
    
    segmenter.eval()

    seg_mean_model_time = timedelta()
    progress_bar = tqdm(range(len(seg_loader)), leave=False)
    for i, image_batch in enumerate(seg_loader):
        progress_bar.set_description(f'Inference ')
        image = image_batch['image']
        label = image_batch['label']
        bbox = image_batch['bbox']
        origin_shape = image_batch['image_meta_dict']['spatial_shape']
        
        model_start = datetime.now()
        seg_logits = segmenter(image)
        model_end = datetime.now()
        seg_mean_model_time += (model_end - model_start)
        
        seg_loss = torch.stack([seg_loss_list[loss_name](seg_logits, label) for loss_name in seg_loss_list]).sum()
        
        seg_logits = post_transform(seg_logits)
        
        seg_logits = restore_original_image(seg_logits, bbox, origin_shape, meta=label.meta).unsqueeze(0)
        
        
        for metric_name in seg_metric_list:
            seg_metric_list[metric_name](y_pred=seg_logits, y=image_batch['gt_label'])
        progress_bar.set_postfix(ValSegLoss = seg_loss.item())
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
    
    seg_mean_model_time /= len(seg_loader)
    
    seg_metric = {}
    for metric_name in seg_metric_list:
        batch_acc = seg_metric_list[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        seg_metric.update(
            {
                f"Test/mean {metric_name}": float(batch_acc.mean()),
            }
        )

    return det_metric, det_mean_model_time, seg_metric, seg_mean_model_time

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
        bbox = image_batch['bbox']
        label_meta = image_batch['label_meta_dict']
        origin_shape = image_meta['spatial_shape']
        
        seg_logits = model(image)
        
        seg_logits = post_transform(seg_logits)
        
        seg_logits = restore_original_image(seg_logits, bbox, origin_shape, meta=label.meta)
        
        
        img_tr(
            image_batch['gt_image'].squeeze(0),
            meta_data=image_meta,
        )
        gen_tr(
            seg_logits,
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

def restore_original_image(cropped_img, bbox, original_shape, meta=None):
    # BBox 좌표를 받아와 원본 이미지에서 해당 위치에 crop된 이미지를 복사합니다.
    # bbox는 shape (1, 1, 6)이며 (x_min, y_min, z_min, x_max, y_max, z_max) 형식을 가집니다.
    x_min, y_min, z_min, x_max, y_max, z_max = bbox[0, 0].to(torch.int)
    original_shape = original_shape # [c, 3]

    # BBox 영역 크기로 먼저 resize
    target_shape = (x_max - x_min, y_max - y_min, z_max - z_min)
    resize_transform = Resize(target_shape, mode='nearest')
    resized_cropped_img = resize_transform(cropped_img.squeeze(0))

    c = cropped_img.shape[1]
    
    restored_img = torch.zeros((c, original_shape[0,0], original_shape[0,1], original_shape[0,2]),dtype=resized_cropped_img.dtype, device=resized_cropped_img.device)
    restored_img[:, x_min:x_max, y_min:y_max, z_min:z_max] = resized_cropped_img
    restored_img = MetaTensor(restored_img, meta=cropped_img.meta if meta == None else meta)
    #pad_transform = Pad([(0,0),(x_min, original_shape[0,0] - x_max),(y_min, original_shape[0,1] - y_max),(z_min, original_shape[0,2] - z_max)], mode='constant')
    #pad_transform = Pad([(0,0),(original_shape[0,0] - x_max, x_min),(original_shape[0,1] - y_max, y_min),(original_shape[0,2] - z_max, z_min)], mode='constant')
    
    #restored_img = pad_transform(resized_cropped_img.to('cpu'))

    return restored_img



class PostProcess(torch.nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        
        scores, labels = prob[..., :-1].max(-1)
        
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results