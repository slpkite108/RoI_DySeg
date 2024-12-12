import torch
import numpy as np
from src.utils import box_ops
import torch.nn.functional as F
from torchvision import transforms as T
from collections import defaultdict
from monai.transforms import SaveImage
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from PIL import ImageDraw, Image

def det_train_one_epoch(model, train_loader, optimizer, scheduler, loss_list, metric_list, post_transform, epoch, step, accelerator, logger):
    model.train()
    for loss_name in loss_list:
        loss_list[loss_name].train()
    grouped_dict = defaultdict(list)
    progress_bar = tqdm(range(len(train_loader)), leave=False)
    pp = PostProcess()
    for i, image_batch in enumerate(train_loader):
        progress_bar.set_description(f'Train Epoch [{epoch+1}] ')
        optimizer.zero_grad()
        x,y,z = image_batch['x'],image_batch['y'],image_batch['z']
        bbox = image_batch['bbox']
        bbox_x, bbox_y, bbox_z = box_ops.box_xyxy_to_cxcywh(bbox[:,:,(1,2,4,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,2,3,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,1,3,4)]/128)
        
        image = torch.cat((x, y, z), dim=1)
        label = torch.cat((bbox_x,bbox_y,bbox_z), dim=1)
        
        seg_logits = model(image)
        #print(seg_logits)
        try:
            label = [{'labels': torch.tensor([0,1,2], device='cuda:0'), 'boxes': b} for b in label]
            seg_loss, metric = zip(*[loss_list[loss_name](seg_logits, label) for loss_name in loss_list])
            seg_loss, metric = seg_loss[0].sum(), metric[0]
        except Exception as e:
            print(f'label: {label}')
            raise e
        accelerator.backward(seg_loss)

        optimizer.step()
        accelerator.log(
            {
                "Train/Detecter Loss": float(seg_loss),
            },
            step=step,
        )
        
        for key, value in metric.items():
            grouped_dict[f'Train/mean {key}'].append(value)
            
        progress_bar.set_postfix(TrainDetLoss = seg_loss, Current_LR=scheduler.get_last_lr())
        step += 1
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
    
    scheduler.step(epoch)
    
    metric = {key: sum(values) / len(values) for key, values in grouped_dict.items()}
    
    logger.info(f"Train Epoch [{epoch + 1}] metric {metric}\n")
    
    train_loader.dataset.update_cache()
    
    return step

@torch.no_grad()
def det_val_one_epoch(model, val_loader, loss_list, metric_list, post_transform, epoch, step, accelerator, logger): #use model.val_one_epoch
    model.eval()
    for loss_name in loss_list:
        loss_list[loss_name].eval()
    grouped_dict = defaultdict(list)
    progress_bar = tqdm(range(len(val_loader)), leave=False)
    #pp = PostProcess()
    for i, image_batch in enumerate(val_loader):
        progress_bar.set_description(f'Validation Epoch [{epoch+1}] ')
        x,y,z = image_batch['x'],image_batch['y'],image_batch['z']
        bbox = image_batch['bbox']
        bbox_x, bbox_y, bbox_z = box_ops.box_xyxy_to_cxcywh(bbox[:,:,(1,2,4,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,2,3,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,1,3,4)]/128)
        image = torch.cat((x, y, z), dim=1)
        label = torch.cat((bbox_x,bbox_y,bbox_z), dim=1)
        
        seg_logits = model(image)

        label = [{'labels': torch.tensor([0,1,2], device='cuda:0'), 'boxes': b} for b in label]
        
        seg_loss, metric = zip(*[loss_list[loss_name](seg_logits, label) for loss_name in loss_list])

        seg_loss, metric = seg_loss[0].sum(), metric[0]
        
        accelerator.log(
            {
                "Val/Detecter Loss": float(seg_loss),
            },
            step=step,
        )
        
        for key, value in metric.items():
            grouped_dict[f'Val/mean {key}'].append(value)

        progress_bar.set_postfix(ValDetLoss = seg_loss.item())
        step += 1
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
            
    metric = {key: sum(values) / len(values) for key, values in grouped_dict.items()}
    logger.info(f"Validation Epoch [{epoch + 1}] metric {metric}\n\n")

    mean_acc = seg_loss #metric['Val/mean loss_giou']
    batch_acc = 0
    
    return step, mean_acc, batch_acc

@torch.no_grad()
def det_test_one_epoch(model, test_loader, loss_list, metric_list,post_transform, accelerator, logger):
    model.eval()
    for loss_name in loss_list:
        loss_list[loss_name].eval()
    grouped_dict = defaultdict(list)
    pp = PostProcess()
    mean_model_time = timedelta()
    progress_bar = tqdm(range(len(test_loader)), leave=False)
    for i, image_batch in enumerate(test_loader):
        progress_bar.set_description(f'Inference ')
        x,y,z = image_batch['x'],image_batch['y'],image_batch['z']
        bbox = image_batch['bbox']
        bbox_x, bbox_y, bbox_z = box_ops.box_xyxy_to_cxcywh(bbox[:,:,(1,2,4,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,2,3,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,1,3,4)]/128)
        image = torch.cat((x, y, z), dim=1)
        label = torch.cat((bbox_x,bbox_y,bbox_z), dim=1)
        
        model_start = datetime.now()
        seg_logits = model(image)
        #print(seg_logits)
        model_end = datetime.now()
        mean_model_time += (model_end - model_start)
        
        label = [{'labels': torch.tensor([0,1,2], device='cuda:0'), 'boxes': b} for b in label]
        seg_loss, metric = zip(*[loss_list[loss_name](seg_logits, label) for loss_name in loss_list])
        
        seg_loss, metric = seg_loss[0].sum(), metric[0]
        
        seg_logits = pp(seg_logits, torch.tensor([image.shape[2], image.shape[3]],device='cuda').unsqueeze(0))[0]
        # print(seg_logits)
        # exit(0)
        _, max_indices = torch.max(seg_logits['scores'], dim=0)
        seg_logits = seg_logits['boxes'][max_indices]

        grouped_dict[f'Test/mean iou'].append(box_ops.box_iou(seg_logits.unsqueeze(0), box_ops.box_cxcywh_to_xyxy(label[0]['boxes'])*128)[0])
        
        for key, value in metric.items():
            grouped_dict[f'Test/mean {key}'].append(value)
        progress_bar.set_postfix(ValSegLoss = seg_loss.item())
        progress_bar.update(1)
        
    progress_bar.clear()
    logger.info(progress_bar)
    
    mean_model_time /= len(test_loader)
    
    metric = {key: sum(values) / len(values) for key, values in grouped_dict.items()}
    logger.info(f"Inference metric {metric}\n\n")
       
    return metric, mean_model_time

@torch.no_grad()
def det_gen_one_epoch(model, path, gen_loader, post_transform , ext , accelerator, logger):
    
    model.eval()
    assert ext in ['.png']
    path = os.path.join(path, f'{ext}_files')
    pp = PostProcess()
    
    gen_tr = SaveImage(
        output_dir=path,  # 파일을 저장할 디렉토리
        output_ext=ext,      # 파일 확장자
        resample=False,            # 필요한 경우 재샘플링 여부
        separate_folder=False,      # 각 이미지마다 개별 폴더에 저장할지 여부
        print_log=False,
        output_postfix='gen',
        writer = 'PILWriter',
    )
    gt_tr = SaveImage(
        output_dir=path,  # 파일을 저장할 디렉토리
        output_ext=ext,      # 파일 확장자
        resample=False,            # 필요한 경우 재샘플링 여부
        separate_folder=False,      # 각 이미지마다 개별 폴더에 저장할지 여부
        print_log=False,
        output_postfix='gt',
        writer = 'PILWriter',
    )
    
    progress_bar = tqdm(range(len(gen_loader)), leave=False)
    for i, image_batch in enumerate(gen_loader):
        progress_bar.set_description(f'Generation ')
        x,y,z = image_batch['x'],image_batch['y'],image_batch['z']
        image = torch.cat((x, y, z), dim=1) # b, c, w, h, d
        min_val, max_val = image.min().item(), image.max().item()

        #image_meta = image_batch['image_meta_dict']
        label = [image_batch['bbox'][:,:,(1,2,4,5)],image_batch['bbox'][:,:,(0,2,3,5)],image_batch['bbox'][:,:,(0,1,3,4)]]
        #bbox_x, bbox_y, bbox_z = box_ops.box_xyxy_to_cxcywh(bbox[:,:,(1,2,4,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,2,3,5)]/128), box_ops.box_xyxy_to_cxcywh(bbox[:,:,(0,1,3,4)]/128)
        seg_logits = model(image)
        #print(seg_logits)
        sp = torch.tensor([image.shape[2], image.shape[3]],device='cuda').unsqueeze(0)
        seg_logits = pp(seg_logits, sp)[0]
        #print(seg_logits)
        _, max_indices = torch.max(seg_logits['scores'], dim=0)
        seg_logits = seg_logits['boxes'][max_indices]
        #print(seg_logits)
        #logger.info(seg_logits)
        
        
        
        # image = torch.where(image != 0, image - min_val, image)
        # norm_image = (image/(max_val-min_val)*255).clamp(0,255).byte()
        for i in range(3):
            print(seg_logits.shape)
            bbox = seg_logits[i].squeeze().tolist()
            gt_bbox = label[i].squeeze().tolist()
            print(bbox)
            print(gt_bbox)
            
            if i == 0:
                image_meta = image_batch['x_meta_dict']
            elif i == 1:
                image_meta = image_batch['y_meta_dict']
            else:
                image_meta = image_batch['z_meta_dict']
                
            image_np = T.ToPILImage()(image[0][i].squeeze())
            image_np = Image.merge("RGB",(image_np,image_np,image_np))
            #x1, y1, x2, y2 = torch.clamp(bbox_tensor, min=0, max=torch.tensor([width - 1, height - 1, width - 1, height - 1]))
            
            #gen_processed  label과 image합성
            gen_processed = image_np.copy()
            draw_gen = ImageDraw.Draw(gen_processed)
            draw_gen.rectangle(bbox, outline='red',width=3)
            gen_processed = np.transpose(np.array(gen_processed),(2,0,1))
            #gt_processed   seg_logits와 image 합성
            
            gt_processed = image_np.copy()
            draw_gt = ImageDraw.Draw(gt_processed)
            draw_gt.rectangle(gt_bbox, outline='blue',width=3)
            gt_processed = np.transpose(np.array(gt_processed), (2,0,1))
            
            gen_tr(#빨간색 bbox
                gen_processed,
                meta_data=image_meta,
            )
            gt_tr(#파란색 bbox
                gt_processed,
                meta_data=image_meta,
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