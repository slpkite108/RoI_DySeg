import nibabel as nib
import numpy as np
import cupy as cp
import os, sys
import shutil

from tqdm import tqdm
from cucim.skimage.transform import resize
import json

#nib_reader = monai.data.NibabelReader()

def calculate_bounding_box(volume, labelmap):
    bboxList = []
    
    for l in labelmap:
        currentVol = cp.where(volume == l, 1, 0)
        
        nonzero = cp.nonzero(currentVol)
        if nonzero[0].size == 0:
            bbox = np.array([0,0,0,0,0,0], dtype=np.int32)#channel_boxes.append(monai.data.MetaTensor([0,0,0,0,0,0], dtype=torch.int64))
        else:
            d_min, h_min, w_min = [cp.min(dim_indices) for dim_indices in nonzero]
            d_max, h_max, w_max = [cp.max(dim_indices) for dim_indices in nonzero]
            
            bbox = cp.asnumpy(cp.array([d_min,h_min, w_min, d_max+1, h_max+1, w_max+1]))
        bboxList.append(bbox)
    return np.stack(bboxList, axis=0)

def make_bbox_list(
        root,
        name,
        label_list,
        target_list,
    ):
    #bbox.json파일을 label 폴더 옆에 생성
    #bbox는 dict로 구성되어 
    #path:{label number: bbox coo} 로 구성
    #label number =0 이면 전체 라벨에 대한 bbox
    #bbox는 xyz min max 순으로 작성
    #xyz 순으로 로드 [ 81,  65,  86, 165, 107, 165]
    bboxlist = {}
    with tqdm(label_list) as pbar:
        for filepath in pbar:
            bboxlist[os.path.basename(filepath)] = {}
            img = nib.load(filepath)
            cp_image = cp.asarray(img.dataobj)
            #print(cp_image.shape)
            
            lab_list = calculate_bounding_box(cp_image, labelmap=target_list)
            print(type(lab_list))
            bboxlist[os.path.basename(filepath)][0] = np.concatenate([lab_list.min(axis=0)[:3],lab_list.max(axis=0)[3:]],axis=0).tolist()
            for idx,lab in enumerate(lab_list):
                bboxlist[os.path.basename(filepath)][idx+1] = lab.tolist()
            pbar.set_postfix(shape = img.shape)
            
    with open(os.path.join(root,name if not name==None else 'bbox.json'),'w')as f:
        json.dump(bboxlist, f, indent=4)
    return

#root=origin_path,name='bboxTr.json', label_list = label_list, target_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
def downsample(file_path, target_size, order, mode, anti_aliasing, preserve_range,):
    img = nib.load(file_path)
    target_size = [target_size,target_size,target_size]
    cp_image = cp.asarray(img.dataobj)
    header = img.header.copy()
    affine = img.affine
    assert len(cp_image.shape) == 3
    
    downsampled = resize(cp_image, target_size, order=order, mode=mode,
                                        anti_aliasing=anti_aliasing,
                                        preserve_range=preserve_range,)
    
    if order == 0:
        downsampled = downsampled.astype(cp.uint8)
    else:
        downsampled = downsampled.astype(cp.float64)
        
    zoom_factors = [cp_image.shape[i] / target_size[i] for i in range(3)]
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * zoom_factors
    
    header.set_data_shape(target_size)
    header['pixdim'][1:4] = header['pixdim'][1:4] * zoom_factors
    
    return nib.Nifti1Image(cp.asnumpy(downsampled), new_affine, header)

def find_nii_gz_files(root_dir, target_dir):
    nii_gz_files = []
    for root, dirs, files in os.walk(root_dir):
        structure_path = root.replace(root_dir, target_dir)

        os.makedirs(structure_path, exist_ok=True)
        for file in files:
            if file.endswith('nii.gz'):
                full_path = os.path.join(root, file)
                if 'label' in full_path:
                    isLab = True
                else: 
                    isLab = False
                nii_gz_files.append((full_path, os.path.join(structure_path,file), isLab))
        
    return nii_gz_files

def reform_origin_folder(root_dir):
    folder_list = []
    for filename in os.listdir(root_dir):
        full_path = os.path.join(root_dir, filename)
        if os.path.isdir(full_path):
            folder_list.append(full_path)

    target_path = os.path.join(root_dir,'origin')
    os.makedirs(target_path, exist_ok=True)
    
    with tqdm(folder_list) as pbar:
        for folder in pbar:
            shutil.move(folder,target_path)
    

def main(root_path):
    
    assert os.path.exists(root_path)
    
    origin_path = os.path.join(str(root_path),'origin')
    print(root_path)
    
    target_size = [16, 32, 64, 128]
    
    if not os.path.exists(origin_path):
        reform_origin_folder(root_dir=root_path)
        
    if not os.path.exists(os.path.join(origin_path,"bboxTr.json")):
        print("generate bbox.json")
        label_list = [os.path.join(os.path.join(origin_path,"labelsTr"),file) for file in sorted(os.listdir(os.path.join(origin_path, "labelsTr")))]
        make_bbox_list(root=origin_path,name='bboxTr.json', label_list = label_list, target_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    
    if not os.path.exists(os.path.join(origin_path,"bboxVa.json")):
        print("generate bbox.json")
        label_list = [os.path.join(os.path.join(origin_path,"labelsVa"),file) for file in sorted(os.listdir(os.path.join(origin_path, "labelsVa")))]
        make_bbox_list(root=origin_path,name='bboxVa.json', label_list = label_list, target_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    
    for size in target_size:  
        target_path = os.path.join(root_path,str(size))
        
        if not os.path.exists(target_path):
            file_list = find_nii_gz_files(root_dir=origin_path, target_dir=target_path)
            
            with tqdm(file_list) as pbar:
                for origin, target, isLab in pbar:
                    if isLab:
                        img = downsample(
                            origin, 
                            target_size=size, 
                            order= 0, 
                            mode='edge',
                            anti_aliasing=False,
                            preserve_range=True,
                            )
                    else:
                        img = downsample(
                            origin, 
                            target_size=size, 
                            order= 1,
                            mode='edge',
                            anti_aliasing=True,
                            preserve_range=True,
                            )
                    nib.save(img,target)
                    pbar.set_postfix(shape = img.shape)
     
        if not os.path.exists(os.path.join(target_path,"bboxTr.json")):
            print("generate bboxTr.json")
            label_list = [os.path.join(os.path.join(target_path,"labelsTr"),file) for file in sorted(os.listdir(os.path.join(target_path, "labelsTr")))]
            make_bbox_list(root=target_path,name='bboxTr.json', label_list = label_list, target_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
                  
        if not os.path.exists(os.path.join(target_path,"bboxVa.json")):
            print("generate bboxVa.json")
            label_list = [os.path.join(os.path.join(target_path,"labelsVa"),file) for file in sorted(os.listdir(os.path.join(target_path, "labelsVa")))]
            make_bbox_list(root=target_path,name='bboxVa.json', label_list = label_list, target_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        
if __name__ == "__main__":
    device = 0
    root_path = '/home/work/.dataset/Amos22_RAS/'
    with cp.cuda.Device(device):
        main(root_path=root_path)
            