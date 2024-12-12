import nibabel as nib
import numpy as np
import cupy as cp
import os, sys
import shutil
from PIL import Image
from tqdm import tqdm
from cucim.skimage.transform import resize
import json

#def save_mean_slices(image_list, target):
def save_center_slices(image_list, target):
    os.makedirs(target, exist_ok=True)
    
    with tqdm(image_list) as pbar:
        for file_path in pbar:
            # 이미지 로드 (NumPy 배열로 변환)
            img = nib.load(file_path)
            np_image = np.array(img.dataobj, dtype=np.float32)
            
            # 먼저 볼륨 전체를 0-255로 스케일링
            np_image = np_image - np_image.min()  # 최소값을 0으로 맞추기
            if np_image.max() > 0:
                np_image = (np_image / np_image.max()) * 255.0
            else:
                # 이미지가 전부 같은 값이라면 그냥 0으로 채워진 볼륨일 가능성 있음.
                # 이 경우 그대로 두거나 특정 값으로 세팅 가능. 여기서는 그냥 0으로 둠.
                pass
            
            # 각 축에 대해 평균 슬라이스 계산
            # x 축 평균: 모든 x 방향에 대해 평균, 결과 shape: (Y, Z)
            x_mean_slice = np.mean(np_image, axis=0)
            # y 축 평균: 모든 y 방향에 대해 평균, 결과 shape: (X, Z)
            y_mean_slice = np.mean(np_image, axis=1)
            # z 축 평균: 모든 z 방향에 대해 평균, 결과 shape: (X, Y)
            z_mean_slice = np.mean(np_image, axis=2)
            
            # 각 평균 슬라이스에 대해 다시 0-255 스케일링(정규화)
            # x_mean_slice는 (Y, Z) 형태
            x_mean_slice = x_mean_slice - x_mean_slice.min()
            if x_mean_slice.max() > 0:
                x_mean_slice = (x_mean_slice / x_mean_slice.max()) * 255
            x_mean_slice = x_mean_slice.astype(np.uint8)
            
            # y_mean_slice는 (X, Z) 형태
            y_mean_slice = y_mean_slice - y_mean_slice.min()
            if y_mean_slice.max() > 0:
                y_mean_slice = (y_mean_slice / y_mean_slice.max()) * 255
            y_mean_slice = y_mean_slice.astype(np.uint8)
            
            # z_mean_slice는 (X, Y) 형태
            z_mean_slice = z_mean_slice - z_mean_slice.min()
            if z_mean_slice.max() > 0:
                z_mean_slice = (z_mean_slice / z_mean_slice.max()) * 255
            z_mean_slice = z_mean_slice.astype(np.uint8)
            
            # 파일 이름 설정
            base_name = os.path.basename(file_path).replace('.nii.gz', '')
            
            # 이미지 저장
            Image.fromarray(x_mean_slice).save(os.path.join(target, f"{base_name}_x.png"))
            Image.fromarray(y_mean_slice).save(os.path.join(target, f"{base_name}_y.png"))
            Image.fromarray(z_mean_slice).save(os.path.join(target, f"{base_name}_z.png"))
            
# def save_center_slices(image_list, target):
#     with tqdm(image_list) as pbar:
#         for file_path in pbar:
#             # 이미지 로드 (NumPy 배열로 변환)
#             img = nib.load(file_path)
#             np_image = np.array(img.dataobj)
            
#             # intensity scaling (0-255로 조정)
#             np_image = np_image - np_image.min()  # 최소값을 0으로 맞추기
#             np_image = (np_image / np_image.max() * 255).astype(np.uint8)  # 최대값을 255로 맞추기
            
#             # 슬라이스 위치 계산
#             x_center = np_image.shape[0] // 2
#             y_center = np_image.shape[1] // 2
#             z_center = np_image.shape[2] // 2

#             # 슬라이스 저장 경로 설정
#             os.makedirs(target, exist_ok=True)
            
#             # 파일 이름 설정
#             base_name = os.path.basename(file_path).replace('.nii.gz', '')

            # # x 축 중앙 슬라이스 저장
            # x_slice = np_image[x_center, :, :]
            # x_slice_img = Image.fromarray(x_slice)
            # x_slice_img.save(os.path.join(target, f"{base_name}_x.png"))

            # # y 축 중앙 슬라이스 저장
            # y_slice = np_image[:, y_center, :]
            # y_slice_img = Image.fromarray(y_slice)
            # y_slice_img.save(os.path.join(target, f"{base_name}_y.png"))

            # # z 축 중앙 슬라이스 저장
            # z_slice = np_image[:, :, z_center]
            # z_slice_img = Image.fromarray(z_slice)
            # z_slice_img.save(os.path.join(target, f"{base_name}_z.png"))

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
    
    if not os.path.exists(os.path.join(origin_path, 'slicesTr')):
        print('generate slices')
        image_list = [os.path.join(os.path.join(origin_path,"imagesTr"),file) for file in sorted(os.listdir(os.path.join(origin_path, "imagesTr")))]
        save_center_slices(image_list = image_list, target=os.path.join(origin_path, 'slicesTr'))
        
    if not os.path.exists(os.path.join(origin_path, 'slicesVa')):
        print('generate slices')
        image_list = [os.path.join(os.path.join(origin_path,"imagesVa"),file) for file in sorted(os.listdir(os.path.join(origin_path, "imagesVa")))]
        save_center_slices(image_list = image_list, target=os.path.join(origin_path, 'slicesVa'))
        
    if not os.path.exists(os.path.join(origin_path, 'slicesTs')):
        print('generate slices')
        image_list = [os.path.join(os.path.join(origin_path,"imagesTs"),file) for file in sorted(os.listdir(os.path.join(origin_path, "imagesTs")))]
        save_center_slices(image_list = image_list, target=os.path.join(origin_path, 'slicesTs'))
    
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
            
        if not os.path.exists(os.path.join(target_path, 'slicesTr')):
            print('generate slices')
            image_list = [os.path.join(os.path.join(target_path,"imagesTr"),file) for file in sorted(os.listdir(os.path.join(target_path, "imagesTr")))]
            save_center_slices(image_list = image_list, target=os.path.join(target_path, 'slicesTr'))
            
        if not os.path.exists(os.path.join(target_path, 'slicesVa')):
            print('generate slices')
            image_list = [os.path.join(os.path.join(target_path,"imagesVa"),file) for file in sorted(os.listdir(os.path.join(target_path, "imagesVa")))]
            save_center_slices(image_list = image_list, target=os.path.join(target_path, 'slicesVa'))
            
        if not os.path.exists(os.path.join(target_path, 'slicesTs')):
            print('generate slices')
            image_list = [os.path.join(os.path.join(target_path,"imagesTs"),file) for file in sorted(os.listdir(os.path.join(target_path, "imagesTs")))]
            save_center_slices(image_list = image_list, target=os.path.join(target_path, 'slicesTs'))
        
if __name__ == "__main__":
    device = 0
    root_path = '/home/work/.dataset/Amos22_RAS/'
    with cp.cuda.Device(device):
        main(root_path=root_path)
            