import os
import json
import torch

from src.utils import box_ops

def load_dataset_image(root, p_image_scale, p_label_scale, p_bbox_scale, p_target_labels, mode='train'):
    
    image_scale = str(p_image_scale)
    label_scale = str(p_label_scale if p_label_scale else image_scale)
    # slice_scale = str(configs.run.loader.dataset.slice_scale if configs.run.loader.dataset.slice_scale else label_scale)
    bbox_scale = str(p_bbox_scale if p_bbox_scale else label_scale)
    
    
    image_root = os.path.join(root, image_scale)
    label_root = os.path.join(root, label_scale)
    bbox_root = os.path.join(root, bbox_scale)
    # slice_root = os.path.join(configs.run.loader.dataset.root, slice_scale)
    
    filter_labels = [1,3,4,6,7,12,15]
    target_labels = p_target_labels
    
    assert os.path.exists(image_root), f"Root directory does not exist: {image_root}"
    
    images_dir = os.path.join(image_root, 'imagesTr' if mode == 'train' else 'imagesVa')
    labels_dir = os.path.join(label_root, 'labelsTr' if mode == 'train' else 'labelsVa')
    # slices_dir = os.path.join(slice_root, 'slicesTr' if mode == 'train' else 'slicesVa')
    
    image_files = sorted(os.listdir(images_dir))
    label_files = sorted(os.listdir(labels_dir))
    # slice_files = sorted(os.listdir(slices_dir)) #amos_0000_x.png
    
    image_list = [os.path.join(images_dir, file) for file in image_files]
    label_list = [os.path.join(labels_dir, file) for file in label_files]
    
    #processing slices
    # slice_dict = {}
    # for file in slice_files:
        # file_path = os.path.join(slices_dir, file)
        # direction = file.split('_')[-1].split('.')[0] # x, y, z
        # file_id = '_'.join(file.split('_')[:2])
        
    #     if file_id not in slice_dict:
    #         slice_dict[file_id] = {'x': None, 'y': None, 'z': None}
        
    #     slice_dict[file_id][direction] = file_path
    # slice_list = [slice_dict[file_id] for file_id in sorted(slice_dict.keys())]
    
    if mode in ['validation', 'inference', 'generation']:
        if mode == 'validation':
            image_list = image_list[0::2]
            label_list = label_list[0::2]
            # slice_list = slice_list[0::2]
        elif mode in ['inference', 'generation']:
            image_list = image_list[1::2]
            label_list = label_list[1::2]
            # slice_list = slice_list[1::2]
    
    bbox_json_path = os.path.join(bbox_root, "bboxTr.json" if mode == 'train' else 'bboxVa.json')
    assert os.path.exists(bbox_json_path), f"bboxTr.json not found at: {bbox_json_path}"
    
    with open(bbox_json_path, 'r') as f:
        bbox_dict = json.load(f)
    
    subjects = []
    
    for idx, label_path in enumerate(label_list):
        image_path = image_list[idx]
        # slice_paths = slice_list[idx]
        label_filename = os.path.basename(label_path)
        
        if label_filename not in bbox_dict:
            print(f"Warning: {label_filename} not found in bboxTr.json. Skipping.")
            continue
        
        bbox_data = bbox_dict[label_filename]
        
        try:
            bbox_coords = [bbox_data[str(k)] for k in sorted(target_labels)]
            
        except KeyError as e:
            print(f"Error: Missing bounding box key {e} in {label_filename}. Skipping.")
            continue
        
        bbox_tensor = torch.tensor(bbox_coords, dtype=torch.float)
        
        max_label = len(target_labels)
        if bbox_tensor.size(0) < max_label:
            print(f"Warning: Not enough bounding boxes for {label_filename}. Skipping.")
            continue
        
        try:
            if all([torch.sum(bbox_tensor[idx]) != 0 for idx, lab in enumerate(target_labels)]):
                subjects.append({
                    'image': image_path,
                    'label': label_path,
                    'bbox': bbox_tensor,
                    # 'slice': slice_paths,
                })
            else:
                print(f"Warning: Some bounding boxes are zero for {label_filename}. Skipping.")
        except IndexError:
            print(f"Error: Bounding box indices out of range for {label_filename}. Skipping.")
            
    #print(subjects[0])
    return subjects

# def load_dataset_slices(configs, mode):
#     image_scale = str(configs.run.loader.dataset.image_scale)
#     label_scale = str(configs.run.loader.dataset.label_scale if configs.run.loader.dataset.label_scale else image_scale)
#     slice_scale = str(configs.run.loader.dataset.slice_scale if configs.run.loader.dataset.slice_scale else label_scale)
#     bbox_scale = str(configs.run.loader.dataset.bbox_scale if configs.run.loader.dataset.bbox_scale else slice_scale)
    
    
#     #image_root = os.path.join(configs.run.loader.dataset.root, image_scale)
#     label_root = os.path.join(configs.run.loader.dataset.root, label_scale)
#     bbox_root = os.path.join(configs.run.loader.dataset.root, bbox_scale)
#     slice_root = os.path.join(configs.run.loader.dataset.root, slice_scale)
    
#     filter_labels = [1,3,4,6,7,12,15]
#     target_labels = configs.run.loader.target_labels
    
#     assert os.path.exists(slice_root), f"Root directory does not exist: {slice_root}"
    
#     labels_dir = os.path.join(label_root, 'labelsTr' if mode == 'train' else 'labelsVa')
#     slices_dir = os.path.join(slice_root, 'slicesTr' if mode == 'train' else 'slicesVa')
    
#     label_files = sorted(os.listdir(labels_dir))
#     slice_files = sorted(os.listdir(slices_dir)) #amos_0000_x.png
    
#     label_list = [os.path.join(labels_dir, file) for file in label_files]
    
#     #processing slices
#     slice_dict = {}
#     for file in slice_files:
#         file_path = os.path.join(slices_dir, file)
#         direction = file.split('_')[-1].split('.')[0] # x, y, z
#         file_id = '_'.join(file.split('_')[:2])
        
#         if file_id not in slice_dict:
#             slice_dict[file_id] = {'x': None, 'y': None, 'z': None}
        
#         slice_dict[file_id][direction] = file_path
#     slice_list = [slice_dict[file_id] for file_id in sorted(slice_dict.keys())]
    
#     if mode in ['validation', 'inference', 'generation']:
#         if mode == 'validation':
#             label_list = label_list[0::2]
#             slice_list = slice_list[0::2]
#         elif mode in ['inference', 'generation']:
#             label_list = label_list[1::2]
#             slice_list = slice_list[1::2]
    
#     bbox_json_path = os.path.join(bbox_root, "bboxTr.json" if mode == 'train' else 'bboxVa.json')
#     assert os.path.exists(bbox_json_path), f"bboxTr.json not found at: {bbox_json_path}"
    
#     with open(bbox_json_path, 'r') as f:
#         bbox_dict = json.load(f)
    
#     subjects = []
    
#     for idx, label_path in enumerate(label_list):
#         slice_paths = slice_list[idx]
#         label_filename = os.path.basename(label_path)
        
#         if label_filename not in bbox_dict:
#             print(f"Warning: {label_filename} not found in bboxTr.json. Skipping.")
#             continue
        
#         bbox_data = bbox_dict[label_filename]
        
#         try:
#             bbox_coords = [bbox_data[str(k)] for k in sorted(target_labels)]
#         except KeyError as e:
#             print(f"Error: Missing bounding box key {e} in {label_filename}. Skipping.")
#             continue
        
#         bbox_tensor = torch.tensor(bbox_coords, dtype=torch.float)
        
#         max_label = len(target_labels)
#         if bbox_tensor.size(0) < max_label:
#             print(f"Warning: Not enough bounding boxes for {label_filename}. Skipping.")
#             continue
        
#         try:
#             if all([torch.sum(bbox_tensor[idx]) != 0 for idx, lab in enumerate(target_labels)]):
#                 subjects.append({
#                     'bbox': bbox_tensor,
#                     'slice': slice_paths,
#                 })
#             else:
#                 print(f"Warning: Some bounding boxes are zero for {label_filename}. Skipping.")
#         except IndexError:
#             print(f"Error: Bounding box indices out of range for {label_filename}. Skipping.")
#     # print(subjects[0])
#     target = []    
#     for sub in subjects:
#         for k in sub['slice'].keys():
#             if mode in ['generation']:
#                 box = sub['bbox'][:,(1,2,4,5)] if k=='x' else sub['bbox'][:,(0,2,3,5)] if k=='y' else sub['bbox'][:,(0,1,3,4)]
#             else:
#                 box = box_ops.box_xyxy_to_cxcywh(sub['bbox'][:,(1,2,4,5)]/128) if k=='x' else box_ops.box_xyxy_to_cxcywh(sub['bbox'][:,(0,2,3,5)]/128) if k=='y' else box_ops.box_xyxy_to_cxcywh(sub['bbox'][:,(0,1,3,4)]/128)
#             target.append({
#                     'image': sub['slice'][k],
#                     'label': box,
#                 })
            
#     return target

def load_dataset_slices(root, p_image_scale, p_label_scale, p_slice_scale,p_bbox_scale, p_target_labels, mode):
    image_scale = str(p_image_scale)
    label_scale = str(p_label_scale if p_label_scale else image_scale)
    slice_scale = str(p_slice_scale if p_slice_scale else label_scale)
    bbox_scale = str(p_bbox_scale if p_bbox_scale else slice_scale)

    label_root = os.path.join(root, label_scale)
    bbox_root = os.path.join(root, bbox_scale)
    slice_root = os.path.join(root, slice_scale)

    filter_labels = [1, 3, 4, 6, 7, 12, 15]
    target_labels = p_target_labels

    assert os.path.exists(slice_root), f"Root directory does not exist: {slice_root}"

    labels_dir = os.path.join(label_root, 'labelsTr' if mode == 'train' else 'labelsVa')
    slices_dir = os.path.join(slice_root, 'slicesTr' if mode == 'train' else 'slicesVa')

    label_files = sorted(os.listdir(labels_dir))
    slice_files = sorted(os.listdir(slices_dir))  # amos_0000_x.png

    label_list = [os.path.join(labels_dir, file) for file in label_files]

    # 슬라이스 처리
    slice_dict = {}
    for file in slice_files:
        file_path = os.path.join(slices_dir, file)
        direction = file.split('_')[-1].split('.')[0]  # x, y, z
        file_id = '_'.join(file.split('_')[:2])

        if file_id not in slice_dict:
            slice_dict[file_id] = {'x': None, 'y': None, 'z': None}

        slice_dict[file_id][direction] = file_path
    slice_list = [slice_dict[file_id] for file_id in sorted(slice_dict.keys())]

    if mode in ['validation', 'inference', 'generation']:
        if mode == 'validation':
            label_list = label_list[0::2]
            slice_list = slice_list[0::2]
        elif mode in ['inference', 'generation']:
            label_list = label_list[1::2]
            slice_list = slice_list[1::2]

    bbox_json_path = os.path.join(bbox_root, "bboxTr.json" if mode == 'train' else 'bboxVa.json')
    assert os.path.exists(bbox_json_path), f"bboxTr.json not found at: {bbox_json_path}"

    with open(bbox_json_path, 'r') as f:
        bbox_dict = json.load(f)

    subjects = []

    for idx, label_path in enumerate(label_list):
        slice_paths = slice_list[idx]
        label_filename = os.path.basename(label_path)

        if label_filename not in bbox_dict:
            print(f"Warning: {label_filename} not found in bboxTr.json. Skipping.")
            continue

        bbox_data = bbox_dict[label_filename]

        try:
            bbox_coords = [bbox_data[str(k)] for k in sorted(target_labels)]
        except KeyError as e:
            print(f"Error: Missing bounding box key {e} in {label_filename}. Skipping.")
            continue

        bbox_tensor = torch.tensor(bbox_coords, dtype=torch.float)

        max_label = len(target_labels)
        if bbox_tensor.size(0) < max_label:
            print(f"Warning: Not enough bounding boxes for {label_filename}. Skipping.")
            continue

        try:
            if all([torch.sum(bbox_tensor[idx]) != 0 for idx, lab in enumerate(target_labels)]):
                subjects.append({
                    'bbox': bbox_tensor,
                    'slice': slice_paths,
                })
            else:
                print(f"Warning: Some bounding boxes are zero for {label_filename}. Skipping.")
        except IndexError:
            print(f"Error: Bounding box indices out of range for {label_filename}. Skipping.")

    # 3D bbox와 각 방향의 슬라이스를 내보냅니다.
    target = []
    for sub in subjects:
        target.append({
            'bbox': sub['bbox'],
            'x': sub['slice']['x'],
            'y': sub['slice']['y'],
            'z': sub['slice']['z'],
        })

    return target

def image_loader(configs, type = 'seg', mode='train'):
    
    if type in ['det']:
        return load_dataset_slices(configs.run.loader.dataset.root, configs.run.loader.dataset.image_scale, configs.run.loader.dataset.label_scale, configs.run.loader.dataset.slice_scale, configs.run.loader.dataset.bbox_scale, configs.run.loader.target_labels, mode)
    else:
        return load_dataset_image(configs.run.loader.dataset.root, configs.run.loader.dataset.image_scale, configs.run.loader.dataset.label_scale,configs.run.loader.dataset.bbox_scale, configs.run.loader.target_labels, mode)
    
    #box = sub['bbox'][:,(1,2,4,5)] if k=='x' else sub['bbox'][:,(0,2,3,5)] if k=='y' else sub['bbox'][:,(0,1,3,4)]