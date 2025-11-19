"""
베이스라인 분류 모델 학습 스크립트
EfficientNet-B4를 사용한 피부 질환 이미지 분류
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import json
from datetime import datetime
import cv2

# 데이터 경로 (multiprocessing 호환성을 위해 직접 정의)
TRAINING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Training'
LABELING_PATH = '/Users/jonabi/Downloads/JLK_DATA/Labeling'

# Dataset 클래스 (multiprocessing 호환성을 위해 직접 정의)
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
            if not cat_dir.is_dir() or cat_dir.name.startswith('.'):
                continue
            if cat_dir.name.replace('TS_', '') == category_name:
                image_path = cat_dir / f"{identifier}.png"
                if image_path.exists():
                    break
                else:
                    image_path = None
        
        if image_path is None or not image_path.exists():
            # 더 자세한 디버깅 정보
            available_dirs = [d.name for d in self.image_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            raise FileNotFoundError(
                f"이미지를 찾을 수 없습니다: {identifier}\n"
                f"  카테고리: {category_name}\n"
                f"  찾은 경로: {image_path}\n"
                f"  사용 가능한 폴더: {available_dirs}"
            )
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 변환 적용
        if self.transform:
            # 분류를 위한 변환
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # 분류 태스크
        label = self.category_to_idx[category]
        return image, label

def get_classification_transforms(image_size=512, is_train=True):
    """분류를 위한 이미지 변환 함수"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
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

def load_and_split_data(test_size=0.15, val_size=0.15, random_state=42):
    """데이터 로드 및 분할"""
    from sklearn.model_selection import train_test_split
    
    # Training 폴더에 있는 카테고리 목록 가져오기
    available_categories = set()
    for cat_dir in Path(TRAINING_PATH).iterdir():
        if cat_dir.is_dir() and not cat_dir.name.startswith('.'):
            category_name = cat_dir.name.replace('TS_', '')
            available_categories.add(category_name)
    
    print(f"Training에 있는 카테고리: {sorted(available_categories)}")
    
    # Labeling 데이터 로드 (Training에 있는 카테고리만)
    data = []
    
    for category_dir in Path(LABELING_PATH).iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category_name = category_dir.name.replace('TL_', '')
        
        # Training에 없는 카테고리는 스킵
        if category_name not in available_categories:
            print(f"  스킵: {category_name} (Training에 없음)")
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
    
    df = pd.DataFrame(data)
    print(f"\n총 {len(df)}개의 데이터 로드됨 (Training에 있는 카테고리만)")
    
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
    from torch.utils.data import DataLoader
    
    # 변환 함수 생성
    train_transform = get_classification_transforms(is_train=True)
    val_transform = get_classification_transforms(is_train=False)
    test_transform = get_classification_transforms(is_train=False)
    
    # 데이터셋 생성
    train_dataset = SkinDiseaseDataset(train_df, TRAINING_PATH, transform=train_transform, task=task)
    val_dataset = SkinDiseaseDataset(val_df, TRAINING_PATH, transform=val_transform, task=task)
    test_dataset = SkinDiseaseDataset(test_df, TRAINING_PATH, transform=test_transform, task=task)
    
    # 카테고리 정보 동기화 (첫 번째 데이터셋 기준)
    val_dataset.categories = train_dataset.categories
    val_dataset.category_to_idx = train_dataset.category_to_idx
    val_dataset.idx_to_category = train_dataset.idx_to_category
    test_dataset.categories = train_dataset.categories
    test_dataset.category_to_idx = train_dataset.category_to_idx
    test_dataset.idx_to_category = train_dataset.idx_to_category
    
    # 데이터로더 생성 (macOS에서는 num_workers=0 권장)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # macOS MPS에서는 pin_memory 지원 안 함
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset

# 한글 폰트 설정 (macOS)
import matplotlib.font_manager as fm
import matplotlib

# 한글 폰트 설정 함수 (한글 깨짐 방지)
def set_korean_font():
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    font_list = [f.name for f in fm.fontManager.ttflist]
    # 선호 순서대로 폰트 리스트
    preferred_fonts = [
        'Apple SD Gothic Neo', 'AppleGothic', 'NanumGothic', 'Malgun Gothic',
        '돋움', 'Arial Unicode MS', 'Noto Sans CJK KR', 'Pretendard'
    ]
    for font_name in preferred_fonts:
        if font_name in font_list:
            matplotlib.rc('font', family=font_name)
            print(f'[폰트적용] 한글 폰트: {font_name}')
            break
    else:
        matplotlib.rc('font', family='DejaVu Sans')
        print('[경고] 한글 폰트가 없어서 기본 폰트(DejaVu Sans)가 적용됩니다. 한글이 네모(□)로 보일 수 있습니다.')
    matplotlib.rcParams['axes.unicode_minus'] = False

set_korean_font()

class EfficientNetClassifier(nn.Module):
    """EfficientNet 기반 분류 모델"""
    
    def __init__(self, num_classes=6, model_name='efficientnet_b4', pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        # EfficientNet 모델 로드
        if model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        # 분류 헤드
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Progress bar 업데이트
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # 상세 메트릭
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return epoch_loss, epoch_acc, precision, recall, f1

def plot_training_history(history, save_path='training_history.png'):
    """학습 히스토리 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision, Recall, F1
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].plot(history['val_f1'], label='F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n학습 히스토리가 '{save_path}'로 저장되었습니다.")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '개수'})
    plt.title('혼동 행렬', fontsize=14, fontweight='bold')
    plt.ylabel('실제 레이블', fontsize=12)
    plt.xlabel('예측 레이블', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"혼동 행렬이 '{save_path}'로 저장되었습니다.")

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    learning_rate=1e-4,
    device='cuda',
    save_dir='checkpoints'
):
    """모델 학습"""
    
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss 및 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 학습 히스토리
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\n학습 시작: {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Learning rate 조정
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # 결과 출력
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"✓ Best model saved! (Val Acc: {best_val_acc:.4f})")
        
        # 주기적으로 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n" + "=" * 50)
    print(f"학습 완료!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print("=" * 50)
    
    return history, best_val_acc

def evaluate_model(model, test_loader, device, class_names):
    """테스트 세트 평가"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 메트릭 계산
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 카테고리별 메트릭
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print("\n" + "=" * 50)
    print("테스트 세트 평가 결과")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-Score: {f1:.4f}")
    print("\n카테고리별 성능:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall: {recall_per_class[i]:.4f}")
        print(f"    F1-Score: {f1_per_class[i]:.4f}")
    
    # 혼동 행렬
    plot_confusion_matrix(all_labels, all_preds, class_names)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }

def main():
    """메인 실행 함수"""
    # 설정
    config = {
        'batch_size': 16,  # GPU 사용 시 더 큰 배치 사이즈 가능
        'num_epochs': 5,  # 테스트용으로 에폭 수 줄임 (실제 학습 시 20 이상 권장)
        'learning_rate': 1e-4,
        'num_workers': 0,  # macOS multiprocessing 이슈로 0으로 설정 (필요시 2-4로 변경 가능)
        'image_size': 512,
        'model_name': 'efficientnet_b3',  # B3가 B4보다 빠름 (실제 학습 시 B4 권장)
        'save_dir': 'checkpoints',
        'random_state': 42
    }
    
    # Device 설정 (MPS > CUDA > CPU 순서)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU)")
    
    # 데이터 로드
    print("\n데이터 로딩 중...")
    train_df, val_df, test_df = load_and_split_data(
        test_size=0.15, 
        val_size=0.15, 
        random_state=config['random_state']
    )
    
    # 데이터로더 생성
    print("\n데이터로더 생성 중...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        task='classification'
    )
    
    class_names = dataset.categories
    num_classes = len(class_names)
    
    print(f"\n카테고리: {class_names}")
    print(f"카테고리 수: {num_classes}")
    
    # 모델 생성
    print(f"\n모델 생성 중... ({config['model_name']})")
    model = EfficientNetClassifier(
        num_classes=num_classes,
        model_name=config['model_name'],
        pretrained=True
    )
    model = model.to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 학습
    history, best_val_acc = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device,
        save_dir=config['save_dir']
    )
    
    # 학습 히스토리 시각화
    plot_training_history(history, save_path='training_history.png')
    
    # Best model 로드
    print("\nBest model 로딩 중...")
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 세트 평가
    test_results = evaluate_model(model, test_loader, device, class_names)
    
    # 결과 저장
    results = {
        'config': config,
        'best_val_acc': float(best_val_acc),
        'test_results': test_results,
        'class_names': class_names,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n결과가 'training_results.json'으로 저장되었습니다.")
    print("\n학습 완료!")

if __name__ == "__main__":
    main()

