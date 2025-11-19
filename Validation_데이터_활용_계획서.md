# Validation 데이터 활용 계획서

## 1. Validation 데이터 개요

### 1.1 데이터 구조

- **Validation_Resource**: `/Users/jonabi/Downloads/JLK_DATA/Validation/Validation_Resource`

  - 형식: PNG 이미지 파일
  - 총 700개 이미지
  - 카테고리별 분포 (각 100개):
    - 광선각화증 (Actinic Keratosis): 100개
    - 기저세포암 (Basal Cell Carcinoma): 100개
    - 멜라닌세포모반 (Melanocytic Nevus): 100개
    - 보웬병 (Bowen's Disease): 100개
    - 악성흑색종 (Malignant Melanoma): 100개
    - 지루각화증 (Seborrheic Keratosis): 100개 ⚠️ **라벨링 없음**
    - 혈관종 (Hemangioma): 100개

- **Validation_Labeling**: `/Users/jonabi/Downloads/JLK_DATA/Validation/Validation_Labeling`
  - 형식: JSON 파일 (주석 및 바운딩 박스 정보 포함)
  - 총 600개 JSON 파일
  - 카테고리별 분포 (각 100개):
    - 광선각화증: 100개
    - 기저세포암: 100개
    - 멜라닌세포모반: 100개
    - 보웬병: 100개
    - 악성흑색종: 100개
    - 혈관종: 100개
    - ⚠️ **지루각화증은 라벨링 데이터 없음**

### 1.2 Training vs Validation 비교

| 항목            | Training               | Validation                          |
| --------------- | ---------------------- | ----------------------------------- |
| **이미지 개수** | 4,000개                | 700개                               |
| **라벨링 개수** | 4,800개 (6개 카테고리) | 600개 (6개 카테고리)                |
| **카테고리 수** | 5개 (보웬병 없음)      | 7개 (지루각화증 포함, 라벨링은 6개) |
| **용도**        | 모델 학습              | 모델 평가 및 검증                   |
| **데이터 분포** | 균등 (각 800개)        | 균등 (각 100개)                     |

## 2. Validation 데이터 활용 전략

### 2.1 권장 활용 방법 (3가지 옵션)

#### **옵션 1: 독립적인 테스트 세트로 활용 (권장) ⭐**

**개념**: Validation 데이터를 완전히 독립적인 테스트 세트로 사용

**장점**:

- Training 데이터와 완전히 분리되어 모델의 일반화 성능을 더 정확히 평가
- 실제 배포 환경과 유사한 조건에서 모델 성능 검증
- 데이터 누수(data leakage) 방지

**구현 방법**:

1. Training 데이터로만 모델 학습 (기존 방식 유지)
2. Validation 데이터를 최종 테스트 세트로 사용
3. Training에서 분할한 Test 세트는 개발 중간 평가용으로만 사용

**데이터 분할 구조**:

```
Training 데이터 (4,000개)
├── Train: 2,800개 (70%)
└── Val: 1,200개 (30%)  ← 하이퍼파라미터 튜닝용

Validation 데이터 (600개)
└── Final Test: 600개  ← 최종 성능 평가용
```

**평가 지표**:

- 최종 테스트 세트에서의 Accuracy, Precision, Recall, F1-Score
- 카테고리별 성능 분석
- 혼동 행렬 분석

---

#### **옵션 2: 교차 검증 세트로 활용**

**개념**: Validation 데이터를 추가 검증 세트로 사용하여 모델 안정성 검증

**장점**:

- 여러 데이터셋에서 일관된 성능 확인
- 모델의 안정성 및 신뢰성 검증
- 과적합(overfitting) 여부 확인

**구현 방법**:

1. Training 데이터로 모델 학습
2. Training의 Val 세트와 Validation 데이터 모두에서 평가
3. 두 세트의 성능 차이 분석

**평가 지표**:

- Training Val vs Validation 성능 비교
- 성능 차이 분석 (일관성 확인)

---

#### **옵션 3: 지루각화증 카테고리 추가 학습**

**개념**: Validation의 지루각화증 데이터를 활용하여 새로운 카테고리 학습

**장점**:

- 모델이 더 많은 카테고리를 분류할 수 있음
- 지루각화증은 라벨링이 없지만, 자가 지도 학습(self-supervised learning) 또는 전이 학습 활용 가능

**구현 방법**:

1. 지루각화증 이미지만 별도로 수집
2. 라벨링 없이 클러스터링 또는 전이 학습 활용
3. 또는 전문가 라벨링 후 추가 학습

**주의사항**:

- 라벨링 데이터가 없어서 지도 학습이 어려움
- 반지도 학습(semi-supervised learning) 기법 필요

---

### 2.2 추천 전략: 옵션 1 (독립적인 테스트 세트)

**이유**:

1. **과학적 엄밀성**: 완전히 독립적인 테스트 세트는 모델의 실제 성능을 가장 정확히 반영
2. **실무 적용성**: 실제 배포 환경과 유사한 조건
3. **구현 용이성**: 기존 코드 수정 최소화

## 3. 구체적인 구현 계획

### 3.1 데이터 로딩 스크립트 작성

**파일**: `04_load_validation_data.py`

**기능**:

1. Validation_Resource에서 이미지 로드
2. Validation_Labeling에서 JSON 로드
3. Training에 있는 카테고리만 필터링 (보웬병 제외)
4. 데이터 통합 및 DataFrame 생성

**주의사항**:

- 지루각화증은 라벨링이 없으므로 제외
- 보웬병은 Training에 없으므로 제외 (또는 별도 처리)
- 최종적으로 5개 카테고리만 사용

### 3.2 평가 스크립트 작성

**파일**: `05_evaluate_on_validation.py`

**기능**:

1. 학습된 모델 로드
2. Validation 데이터로 평가
3. 상세 메트릭 계산:
   - Overall Accuracy, Precision, Recall, F1-Score
   - 카테고리별 성능
   - 혼동 행렬
   - Classification Report
4. Training Test vs Validation 성능 비교

### 3.3 성능 비교 분석

**비교 항목**:

1. **Training Test Set 성능** (기존)
2. **Validation Set 성능** (새로운)
3. **성능 차이 분석**:
   - 두 세트 간 성능 차이
   - 카테고리별 성능 차이
   - 오분류 패턴 분석

## 4. 코드 수정 사항

### 4.1 기존 코드 수정

**`03_train_classification_model.py`**:

- Validation 데이터 경로 추가
- 평가 함수에 Validation 데이터셋 평가 옵션 추가

### 4.2 새로운 파일 생성

1. **`04_load_validation_data.py`**

   - Validation 데이터 로딩 함수
   - 데이터 전처리 및 필터링

2. **`05_evaluate_on_validation.py`**

   - Validation 세트 평가 스크립트
   - 성능 비교 및 시각화

3. **`Validation_데이터_활용_계획서.md`** (현재 파일)
   - 전체 계획 문서

## 5. 실행 순서

### Phase 1: 데이터 준비

```bash
# 1. Validation 데이터 탐색
python 04_load_validation_data.py

# 2. 데이터 구조 확인 및 통계
```

### Phase 2: 모델 학습 (기존)

```bash
# Training 데이터로 모델 학습
python 03_train_classification_model.py
```

### Phase 3: Validation 평가

```bash
# Validation 데이터로 최종 평가
python 05_evaluate_on_validation.py
```

### Phase 4: 성능 비교 및 분석

- Training Test vs Validation 성능 비교
- 카테고리별 성능 분석
- 오분류 패턴 분석

## 6. 예상 결과 및 활용

### 6.1 평가 결과

- **Validation Accuracy**: 예상 80-90% (Training Test와 유사할 것으로 예상)
- **카테고리별 성능**: 각 카테고리별 Precision, Recall, F1-Score
- **혼동 행렬**: 어떤 카테고리를 자주 혼동하는지 분석

### 6.2 활용 방안

1. **모델 검증**: 실제 배포 전 최종 검증
2. **성능 보고서**: 연구 논문 또는 보고서 작성 시 사용
3. **모델 개선**: Validation에서 낮은 성능을 보인 카테고리 개선
4. **하이퍼파라미터 튜닝**: Validation 성능을 기준으로 최적화

## 7. 주의사항

### 7.1 데이터 불일치

- **보웬병**: Validation에는 있지만 Training에는 없음
  - 해결: Validation 평가 시 보웬병 제외 또는 별도 처리
- **지루각화증**: Validation Resource에는 있지만 라벨링 없음
  - 해결: 평가에서 제외

### 7.2 카테고리 매칭

- Training과 Validation의 카테고리 이름이 일치하는지 확인
- 카테고리 매핑 딕셔너리 생성 필요

### 7.3 데이터 분포

- Validation은 각 카테고리당 100개로 Training(800개)보다 적음
- 소규모 데이터셋에서의 성능 변동성 고려

## 8. 다음 단계

### 즉시 구현 가능한 작업

1. ✅ Validation 데이터 구조 분석 (완료)
2. ⏳ Validation 데이터 로딩 스크립트 작성
3. ⏳ Validation 평가 스크립트 작성
4. ⏳ 기존 학습 스크립트에 Validation 평가 옵션 추가

### 단기 목표 (1주)

1. Validation 데이터로 모델 평가
2. Training Test vs Validation 성능 비교
3. 성능 차이 분석 및 보고서 작성

### 중기 목표 (2-4주)

1. Validation 성능을 기준으로 모델 개선
2. 하이퍼파라미터 최적화
3. 앙상블 모델 구축 및 평가

## 9. 코드 예시

### Validation 데이터 로딩 함수 예시

```python
def load_validation_data():
    """Validation 데이터 로드"""
    validation_resource_path = '/Users/jonabi/Downloads/JLK_DATA/Validation/Validation_Resource'
    validation_labeling_path = '/Users/jonabi/Downloads/JLK_DATA/Validation/Validation_Labeling'

    # Training에 있는 카테고리만 사용
    training_categories = ['광선각화증', '기저세포암', '멜라닌세포모반', '악성흑색종', '혈관종']

    data = []
    for category in training_categories:
        # JSON 파일 로드
        labeling_dir = Path(validation_labeling_path) / f"VL_{category}"
        resource_dir = Path(validation_resource_path) / f"VS_{category}"

        for json_file in labeling_dir.glob('*.json'):
            # JSON 파싱 및 데이터 추가
            ...

    return pd.DataFrame(data)
```

### Validation 평가 함수 예시

```python
def evaluate_on_validation(model, validation_loader, device, class_names):
    """Validation 세트에서 모델 평가"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 메트릭 계산
    accuracy = accuracy_score(all_labels, all_preds)
    # ... 추가 메트릭

    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }
```

## 10. 요약

**핵심 전략**: Validation 데이터를 **독립적인 최종 테스트 세트**로 활용하여 모델의 실제 성능을 검증

**주요 작업**:

1. Validation 데이터 로딩 및 전처리
2. 학습된 모델로 Validation 평가
3. Training Test vs Validation 성능 비교
4. 성능 분석 및 보고서 작성

**예상 소요 시간**: 1-2일 (코드 작성 및 실행)
