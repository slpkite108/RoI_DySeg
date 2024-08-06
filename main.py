import json
from train import train_model
# from validate import validate_model  # 검증 함수가 구현되어 있다고 가정

def main(config_path='config/config.json'):
    # 설정 파일 로드
    with open(config_path) as config_file:
        config = json.load(config_file)

    # 모델 학습
    train_model(config_path)

    # 모델 검증 (선택적)
    # 검증 결과를 기반으로 최적의 모델 선택 및 저장 로직 구현
    # validate_model(config_path)

if __name__ == '__main__':
    main()
