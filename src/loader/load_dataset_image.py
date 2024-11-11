import os
import json
import torch
import torchio as tio

def get_slice_paths(root_dir):
    slice_paths = []
    
    x_dir = os.path.join(root_dir, 'x')
    y_dir = os.path.join(root_dir, 'y')
    z_dir = os.path.join(root_dir, 'z')
    
    # Fetch and sort file names from each directory
    x_files = sorted(os.listdir(x_dir))
    y_files = sorted(os.listdir(y_dir))
    z_files = sorted(os.listdir(z_dir))
    
    for x_file, y_file, z_file in zip(x_files, y_files, z_files):
        x_path = os.path.join(x_dir, x_file)
        y_path = os.path.join(y_dir, y_file)
        z_path = os.path.join(z_dir, z_file)
        
        slice_paths.append({"x": x_path, "y": y_path, "z": z_path})
    
    return slice_paths

def load_dataset_image(configs, mode='train'):
    root = os.path.join(configs.run.loader.dataset.root, str(configs.run.loader.dataset.scale))
    filter_labels = [1,3,4,6,7,12,15]
    
    assert os.path.exists(root), f"Root directory does not exist: {root}"
    
    images_dir = os.path.join(root, 'imagesTr' if mode == 'train' else 'imagesVa')
    labels_dir = os.path.join(root, 'labelsTr' if mode == 'train' else 'labelsVa')
    
    image_files = sorted(os.listdir(images_dir))
    label_files = sorted(os.listdir(labels_dir))
    
    image_list = [os.path.join(images_dir, file) for file in image_files]
    label_list = [os.path.join(labels_dir, file) for file in label_files]
    
    
    if mode in ['validation', 'inference', 'generation']:
        if mode == 'validation':
            image_list = image_list[0::2]
            label_list = label_list[0::2]
        elif mode in ['inference', 'generation']:
            image_list = image_list[1::2]
            label_list = label_list[1::2]
    
    bbox_json_path = os.path.join(root, "bboxTr.json" if mode == 'train' else 'bboxVa.json')
    assert os.path.exists(bbox_json_path), f"bboxTr.json not found at: {bbox_json_path}"
    
    with open(bbox_json_path, 'r') as f:
        bbox_dict = json.load(f)
    
    subjects = []
    
    for idx, label_path in enumerate(label_list):
        image_path = image_list[idx]
        label_filename = os.path.basename(label_path)
        
        if label_filename not in bbox_dict:
            print(f"Warning: {label_filename} not found in bboxTr.json. Skipping.")
            continue
        
        bbox_data = bbox_dict[label_filename]
        
        try:
            bbox_coords = [bbox_data[str(k)] for k in sorted(bbox_data.keys(), key=lambda x: int(x))]
        except KeyError as e:
            print(f"Error: Missing bounding box key {e} in {label_filename}. Skipping.")
            continue
        
        bbox_tensor = torch.tensor(bbox_coords, dtype=torch.float)
        
        max_label = max(filter_labels)
        if bbox_tensor.size(0) < max_label:
            print(f"Warning: Not enough bounding boxes for {label_filename}. Skipping.")
            continue
        
        try:
            if all([torch.sum(bbox_tensor[lab - 1]) != 0 for lab in filter_labels]):
                subjects.append({
                    'image': image_path,
                    'label': label_path,
                    'bbox': bbox_tensor
                })
            else:
                print(f"Warning: Some bounding boxes are zero for {label_filename}. Skipping.")
        except IndexError:
            print(f"Error: Bounding box indices out of range for {label_filename}. Skipping.")
    
    return subjects