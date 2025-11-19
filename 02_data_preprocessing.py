"""
피부 질환 이미지 데이터 전처리 스크립트
- 이미지 로딩 및 전처리
- 데이터 증강
- 데이터셋 클래스 정의
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# 데이터 경로
TRAINING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Training'
LABELING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Labeling'

class SkinDiseaseDataset(Dataset):
    """피부 질환 이미지 데이터셋 클래스"""
    
    def __init__(self, df, image_dir, transform=None, task='classification'):
        """
        Args:
            df: 데이터프레임 (identifier, category, bbox 정보 포함)
            image_dir: 이미지 디렉토리 경로
            transform: 이미지 변환 함수
            task: 'classification', 'detection', 'segmentation'
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.task = task
        
        # 카테고리를 숫자로 매핑
        self.categories = sorted(self.df['category'].unique())
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        identifier = row['identifier']
        category = row['category']
        category_name = row['category']
        
        # 이미지 경로 찾기
        image_path = None
        for cat_dir in self.image_dir.iterdir():
            if cat_dir.name.replace('TS_', '') == category_name:
                image_path = cat_dir / f"{identifier}.png"
                break
        
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {identifier}")
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 변환 적용
        if self.transform:
            if self.task == 'detection':
                # 객체 탐지를 위한 변환
                bbox = [
                    row['bbox_x'],
                    row['bbox_y'],
                    row['bbox_x'] + row['bbox_width'],
                    row['bbox_y'] + row['bbox_height']
                ]
                transformed = self.transform(image=image, bboxes=[bbox], class_labels=[self.category_to_idx[category]])
                image = transformed['image']
                bbox = transformed['bboxes'][0] if transformed['bboxes'] else None
            else:
                # 분류를 위한 변환
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # 태스크별 반환값
        if self.task == 'classification':
            label = self.category_to_idx[category]
            return image, label
        
        elif self.task == 'detection':
            label = self.category_to_idx[category]
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32) if bbox else None
            return image, {'boxes': bbox_tensor, 'labels': torch.tensor([label])}
        
        else:  # segmentation
            # 세그멘테이션의 경우 마스크 생성 필요
            label = self.category_to_idx[category]
            return image, label  # 실제로는 마스크를 반환해야 함

def get_classification_transforms(image_size=512, is_train=True):
    """분류를 위한 이미지 변환 함수"""
    if is_train:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            # A.GaussNoise(var_limit=10.0, p=0.3),  # 최신 버전에서 지원하지 않음
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform

def get_detection_transforms(image_size=512, is_train=True):
    """객체 탐지를 위한 이미지 변환 함수"""
    if is_train:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    return transform

def load_and_split_data(test_size=0.15, val_size=0.15, random_state=42):
    """데이터 로드 및 분할"""
    # Labeling 데이터 로드
    data = []
    
    for category_dir in Path(LABELING_PATH).iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category_name = category_dir.name.replace('TL_', '')
        
        for json_file in category_dir.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            for annotation in json_data.get('annotations', []):
                bbox = annotation.get('bbox', {})
                photo = annotation.get('photograph', {})
                
                data.append({
                    'identifier': annotation.get('identifier'),
                    'category': category_name,
                    'file_path': photo.get('file_path'),
                    'bbox_x': bbox.get('xpos'),
                    'bbox_y': bbox.get('ypos'),
                    'bbox_width': bbox.get('width'),
                    'bbox_height': bbox.get('height'),
                })
    
    df = pd.DataFrame(data)
    
    # Train/Test 분할
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['category'], 
        random_state=random_state
    )
    
    # Train/Val 분할
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size_adjusted,
        stratify=train_df['category'],
        random_state=random_state
    )
    
    print(f"Train: {len(train_df)}개 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)}개 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)}개 ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def create_dataloaders(train_df, val_df, test_df, batch_size=32, num_workers=4, task='classification'):
    """데이터로더 생성"""
    
    # 변환 함수 생성
    train_transform = get_classification_transforms(is_train=True) if task == 'classification' else get_detection_transforms(is_train=True)
    val_transform = get_classification_transforms(is_train=False) if task == 'classification' else get_detection_transforms(is_train=False)
    test_transform = get_classification_transforms(is_train=False) if task == 'classification' else get_detection_transforms(is_train=False)
    
    # 데이터셋 생성
    train_dataset = SkinDiseaseDataset(train_df, TRAINING_PATH, transform=train_transform, task=task)
    val_dataset = SkinDiseaseDataset(val_df, TRAINING_PATH, transform=val_transform, task=task)
    test_dataset = SkinDiseaseDataset(test_df, TRAINING_PATH, transform=test_transform, task=task)
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset

def main():
    """메인 실행 함수"""
    print("데이터 전처리 시작...")
    
    # 데이터 로드 및 분할
    train_df, val_df, test_df = load_and_split_data()
    
    # 데이터 저장
    train_df.to_csv('train_data.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv('val_data.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('test_data.csv', index=False, encoding='utf-8-sig')
    print("\n데이터 분할 완료 및 CSV 파일로 저장되었습니다.")
    
    # 데이터로더 생성 예시
    print("\n데이터로더 생성 중...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        train_df, val_df, test_df, 
        batch_size=16, 
        task='classification'
    )
    
    print(f"\n카테고리 수: {len(dataset.categories)}")
    print(f"카테고리: {dataset.categories}")
    
    # 샘플 확인
    print("\n샘플 데이터 확인...")
    sample_image, sample_label = dataset[0]
    print(f"이미지 shape: {sample_image.shape}")
    print(f"레이블: {sample_label} ({dataset.idx_to_category[sample_label]})")
    
    print("\n전처리 완료!")

if __name__ == "__main__":
    main()

