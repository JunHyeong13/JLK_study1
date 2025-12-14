# 피부 질환 이미지 분석 프로젝트

## 프로젝트 개요
피부 질환 이미지 데이터를 활용한 딥러닝 기반 분류, 객체 탐지, 세그멘테이션 모델 개발

## 데이터 구조
- **Training 데이터**: 4,800개의 PNG 이미지 (6개 카테고리)
- **Labeling 데이터**: 4,800개의 JSON 파일 (6개 카테고리, 바운딩 박스 정보 포함)

## 설치 방법

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 탐색 (EDA)
```bash
python 01_data_exploration.py
```
- 데이터 분포 분석
- 바운딩 박스 통계
- 시각화 그래프 생성

### 2. 데이터 전처리
```bash
python 02_data_preprocessing.py
```
- 데이터 분할 (Train/Val/Test)
- 데이터로더 생성
- 데이터 증강 설정

## 다음 단계
1. 베이스라인 모델 구현 (EfficientNet 분류 모델)
2. 객체 탐지 모델 구현 (YOLOv8)
3. 세그멘테이션 모델 구현 (U-Net)

자세한 내용은 `분석_계획서.md`를 참고하세요.

