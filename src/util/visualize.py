import matplotlib.pyplot as plt
import numpy as np

def plot_slice(image, mask, prediction=None, slice_number=None):
    """
    주어진 슬라이스 번호에 대해 이미지, 마스크, 예측 결과를 시각화합니다.
    Args:
        image (numpy.ndarray): 원본 이미지 데이터.
        mask (numpy.ndarray): 실제 마스크 데이터.
        prediction (numpy.ndarray, optional): 모델에 의한 예측 마스크.
        slice_number (int, optional): 시각화할 슬라이스 번호. 지정하지 않으면 중간 슬라이스가 선택됩니다.
    """
    if slice_number is None:
        slice_number = image.shape[0] // 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image[slice_number], cmap='gray')
    axs[0].set_title('Image')
    axs[1].imshow(mask[slice_number], cmap='gray')
    axs[1].set_title('Mask')
    
    if prediction is not None:
        axs[2].imshow(prediction[slice_number], cmap='gray')
        axs[2].set_title('Prediction')
    else:
        axs[2].imshow(np.zeros_like(image[slice_number]), cmap='gray')
        axs[2].set_title('Prediction (N/A)')
    
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_metrics(history):
    """
    학습과정에서의 손실과 메트릭을 시각화합니다.
    Args:
        history (dict): 에포크별 손실 및 메트릭 값이 포함된 딕셔너리.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
