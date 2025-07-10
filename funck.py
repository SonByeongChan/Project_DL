# 딥 러닝을 위한 필수 라이브러리 임포트
import os                                         # 파일 경로와 관련된 함수 제공
import random                                     # 랜덤 수 생성
import cv2                                        # 이미지 처리 라이브러리
from PIL import Image                             # 이미지 파일 처리
from tqdm.notebook import tqdm                    # Jupyter Notebook에서 진행 상황 표시
import torch                                      # PyTorch 기본 라이브러리
import torchvision                                # PyTorch의 이미지 데이터셋 및 변환
from torch.utils.data import DataLoader, Dataset  # 데이터 로딩 및 데이터셋 처리
from torchvision import transforms                # 이미지 변환 기능 제공
from torch.autograd import Variable               # 자동 미분 기능
from torch import optim                           # 최적화 알고리즘
import torch.nn as nn                             # 신경망 구성 요소
import torch.nn.functional as F                   # 신경망의 활성화 함수 및 다양한 기능
from matplotlib import pyplot as plt              # 데이터 시각화
import time                                       # 소요 시간 계산산


# 1. 디바이스 확인하기 : 사용 가능한 장치를 반환하는 함수 (GPU 또는 CPU)
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2. 학습 모델 정의 

def train_model(model, dataloader_dict, criterion, optimizer, num_epoch, device, model_dir):
    """
    모델 학습 함수

    매개변수:
    - model: 학습할 모델
    - dataloader_dict: 데이터로더 사전 (훈련 및 검증 데이터)
    - criterion: 손실 함수
    - optimizer: 옵티마이저
    - num_epoch: 학습할 에폭 수
    - device: 학습에 사용할 장치 (CPU 또는 GPU)
    - model_dir: 모델 가중치를 저장할 디렉터리

    반환값:
    - 학습된 모델
    """
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epoch):
        print(f'epoch {epoch + 1}/{num_epoch}')
        print('=' * 28)

        for phase in ['train', 'val']:
            # 모델 모드 설정
            model.train() if phase == 'train' else model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in dataloader_dict[phase]:
                inputs = inputs.to(device)  # 입력 데이터를 장치로 이동
                labels = labels.to(device)  # 레이블 데이터를 장치로 이동

                optimizer.zero_grad()  # 기울기 초기화

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # 모델 예측
                    _, preds = torch.max(outputs, 1)  # 예측값 추출
                    loss = criterion(outputs, labels)  # 손실 계산

                    if phase == 'train':
                        loss.backward()  # 역전파
                        optimizer.step()  # 가중치 업데이트

                # 손실 및 정확도 계산
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            # 평균 손실 및 정확도
            epoch_loss /= len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.8f}, ACC: {epoch_acc:.8f}')

            # 최적의 모델 가중치 저장
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

                # 모델 가중치 저장
                torch.save(model.state_dict(), f'{model_dir}fashion_weights_epoch{epoch}_{epoch_acc:.4f}.pt')
                print('모델 가중치 저장됨^ㅋㅋ^')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초')
    print(f'Best val ACC: {best_acc:.8f}')

    return model


# 3. 조기 종료
def end(self, SCHDULER):
    SCHDULER.step(validLoss)
    
    if SCHDULER.num_bad_epochs >= SCHDULER.patience:
        EARLY_STOP -= 1

    if not EARLY_STOP:
        print(f'{Epoch}EPOCH : 성능 개선이 없어 강종~~>.<')

# 조기 종료를 위한 기준값 저장