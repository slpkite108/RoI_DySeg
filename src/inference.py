import torch
import json
from model.fcos_3d import FCOS3D, SimpleBackbone3D
from util.data_loader import get_data_loaders
from util.visualize import plot_slice

def inference(model_path='model.pth', config_path='config/config.json', data_path='path/to/data'):
    # 설정 파일 로드
    with open(config_path) as config_file:
        config = json.load(config_file)

    # 모델 로드
    backbone = SimpleBackbone3D()  # 실제 사용 시, 적절한 백본 모델로 교체 필요
    model = FCOS3D(backbone, num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    # 데이터 로드 및 전처리
    dataloader = get_data_loaders([data_path], [data_path], batch_size=1)  # 예시로 동일 경로 사용

    with torch.no_grad():
        for images, _ in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()

            outputs = model(images)
            # 추론 결과 처리 (예: 시각화, 저장 등)
            plot_slice(images.cpu().numpy(), outputs.cpu().numpy())

if __name__ == '__main__':
    inference()
