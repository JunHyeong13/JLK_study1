"""
Validation 데이터로 모델 평가 스크립트
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
import cv2
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 한글 폰트 설정 (macOS)
import matplotlib.font_manager as fm
import matplotlib
# 한글이 네모로 보이면 다양한 한글 폰트를 순차적으로 적용
def set_korean_font():
    font_list = [f.name for f in fm.fontManager.ttflist]
    tried_fonts = []
    preferred_fonts = [
        'Apple SD Gothic Neo',
        'AppleGothic',
        'NanumGothic',
        'Malgun Gothic',
        '돋움',
        'Arial Unicode MS',
        'Noto Sans CJK KR',
        'Pretendard'
    ]
    for font_name in preferred_fonts:
        if font_name in font_list:
            plt.rcParams['font.family'] = font_name
            print(f"[폰트 적용] 한글 폰트: {font_name}")
            break
        else:
            tried_fonts.append(font_name)
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print(f"[경고] 한글 폰트가 적용되지 않을 수 있습니다. 시도된 폰트: {tried_fonts}")

    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 데이터 경로
VALIDATION_RESOURCE_PATH = '/Users/jonabi/Downloads/JLK_DATA/Validation/Validation_Resource'
VALIDATION_LABELING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Validation/Validation_Labeling'
TRAINING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Training'

# 모델 클래스 정의 (03_train_classification_model.py와 동일)
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EfficientNetClassifier(nn.Module):
    """EfficientNet 기반 분류 모델"""
    
    def __init__(self, num_classes=6, model_name='efficientnet_b3', pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        if model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def get_classification_transforms(image_size=512, is_train=True):
    """분류를 위한 이미지 변환 함수"""
    if is_train:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
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

class ValidationDataset(Dataset):
    """Validation 데이터셋 클래스"""
    
    def __init__(self, df, image_dir, transform=None, category_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.category_to_idx = category_to_idx or self._create_category_mapping()
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
    
    def _create_category_mapping(self):
        """카테고리 매핑 생성"""
        categories = sorted(self.df['category'].unique())
        return {cat: idx for idx, cat in enumerate(categories)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        identifier = row['identifier']
        category = row['category']
        
        # 이미지 경로 찾기
        resource_dir = self.image_dir / f"VS_{category}"
        image_path = resource_dir / f"{identifier}.png"
        
        if not image_path.exists():
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {identifier} ({category})")
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 변환 적용
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # 레이블
        label = self.category_to_idx[category]
        return image, label

def load_validation_data():
    """Validation 데이터 로드"""
    # Training에 있는 카테고리만 사용
    training_categories = set()
    for cat_dir in Path(TRAINING_PATH).iterdir():
        if cat_dir.is_dir() and not cat_dir.name.startswith('.'):
            category_name = cat_dir.name.replace('TS_', '')
            training_categories.add(category_name)
    
    data = []
    for category_dir in Path(VALIDATION_LABELING_PATH).iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category_name = category_dir.name.replace('VL_', '')
        
        # Training에 없는 카테고리는 스킵
        if category_name not in training_categories:
            continue
        
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
    
    return pd.DataFrame(data)

def load_model(checkpoint_path, num_classes, device):
    """학습된 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 생성 (카테고리 수에 맞게)
    model = EfficientNetClassifier(
        num_classes=num_classes,
        model_name='efficientnet_b3',
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, dataloader, device, class_names):
    """모델 평가"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 메트릭 계산
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 카테고리별 메트릭
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support': support,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='validation_confusion_matrix.png'):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '개수'})
    plt.title('Validation Set 혼동 행렬', fontsize=14, fontweight='bold')
    plt.ylabel('실제 레이블', fontsize=12)
    plt.xlabel('예측 레이블', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n혼동 행렬이 '{save_path}'로 저장되었습니다.")

def print_detailed_results(results, class_names):
    """상세 결과 출력"""
    print("\n" + "=" * 50)
    print("Validation Set 평가 결과")
    print("=" * 50)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    
    print(f"\n카테고리별 성능:")
    print(f"{'카테고리':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {results['precision_per_class'][i]:<10.4f} "
              f"{results['recall_per_class'][i]:<10.4f} "
              f"{results['f1_per_class'][i]:<10.4f} "
              f"{results['support'][i]:<10}")
    
    # Classification Report
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    report = classification_report(
        results['labels'], 
        results['predictions'],
        target_names=class_names,
        digits=4
    )
    print(report)

def main():
    """메인 실행 함수"""
    # 설정
    checkpoint_path = 'checkpoints/best_model.pth'
    batch_size = 32  # GPU 사용 시 더 큰 배치 사이즈 가능
    # Device 설정 (MPS > CUDA > CPU 순서)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU)")
    
    # Validation 데이터 로드
    print("\nValidation 데이터 로딩 중...")
    validation_df = load_validation_data()
    print(f"총 {len(validation_df)}개의 Validation 데이터 로드됨")
    
    # 카테고리 정보
    categories = sorted(validation_df['category'].unique())
    num_classes = len(categories)
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    print(f"\n카테고리: {categories}")
    print(f"카테고리 수: {num_classes}")
    
    # 변환 함수
    transform = get_classification_transforms(is_train=False)
    
    # 데이터셋 및 데이터로더 생성
    validation_dataset = ValidationDataset(
        validation_df, 
        VALIDATION_RESOURCE_PATH,
        transform=transform,
        category_to_idx=category_to_idx
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 모델 로드
    print(f"\n모델 로딩 중: {checkpoint_path}")
    model = load_model(checkpoint_path, num_classes, device)
    print("모델 로드 완료!")
    
    # 평가
    print("\n모델 평가 시작...")
    results = evaluate_model(model, validation_loader, device, categories)
    
    # 결과 출력
    print_detailed_results(results, categories)
    
    # 혼동 행렬 시각화
    plot_confusion_matrix(
        results['labels'], 
        results['predictions'], 
        categories
    )
    
    # 결과 저장
    results_dict = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
        'precision_per_class': results['precision_per_class'].tolist(),
        'recall_per_class': results['recall_per_class'].tolist(),
        'f1_per_class': results['f1_per_class'].tolist(),
        'support': results['support'].tolist(),
        'class_names': categories
    }
    
    with open('validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print("\n결과가 'validation_results.json'으로 저장되었습니다.")
    print("\n" + "=" * 50)
    print("Validation 평가 완료!")
    print("=" * 50)

if __name__ == "__main__":
    main()

