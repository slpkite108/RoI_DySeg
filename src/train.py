import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from model.fcos_3d import FCOS3D, SimpleBackbone3D
from util.data_loader import NiiGzDataset
from metric.evaluation import dice_coefficient, calculate_precision_recall, calculate_accuracy

def train_model(config_path='config/config.json'):
    # 설정 파일 로드
    with open(config_path) as config_file:
        config = json.load(config_file)

    # 데이터셋 준비
    train_dataset = NiiGzDataset(
        image_paths=config['data']['train_images'],
        mask_paths=config['data']['train_masks']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

    # 모델 초기화
    backbone = SimpleBackbone3D()  # 실제 사용 시, 적절한 백본 모델로 교체 필요
    model = FCOS3D(backbone, num_classes=config['model']['num_classes'])
    if torch.cuda.is_available():
        model.cuda()

    # 옵티마이저 및 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()  # 이진 분류 문제에 적합한 손실 함수

    # 학습 루프
    for epoch in range(config['train']['num_epochs']):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{config["train"]["num_epochs"]}, Loss: {total_loss/len(train_loader)}')

        # 여기에 검증 로직 추가 (옵션)

    # 모델 저장
    torch.save(model.state_dict(), 'model.pth')
    print('Training complete. Model saved as model.pth')

if __name__ == '__main__':
    train_model()
