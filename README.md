# MachineLearningClass

**머신러닝 수업·Kaggle InClass 대회를 위한 실습·경쟁 코드 모음**

국민대학교 AI빅데이터융합경영학과 머신러닝 과정에서 수행한 **팀 그룹과제(임금 회귀·Pipeline)** 와 **KML Challenge 2023F(패널 설문 응답 예측)** 작업을 프로젝트별 폴더로 분리해 정리한 저장소이다. 여러 실험 노트북이 한 디렉터리에 섞여 있던 상태를 해소하고, 재현·열람이 가능한 구조로 재배치하였다.

**프로젝트 상태: 완료** — 수업 제출·대회 제출 시점의 코드와 제출 CSV가 보존된 아카이브이다.

**작성:** 배성윤

---

## 저장소 구성 (2개 프로젝트)

| 프로젝트 | 폴더 | 문제 | 평가 지표 |
|----------|------|------|-----------|
| 수업 그룹과제 (Pipeline·앙상블) | [`01_course_salary_regression/`](01_course_salary_regression/) | 설명변수 → **Salary** 회귀 | RMSE |
| KML Challenge 2023F | [`02_kml_challenge_2023f/`](02_kml_challenge_2023f/) | 패널·설문 매칭 → **STATUS**(응답 여부) 분류 | Accuracy |

---

## 01. 수업: 임금 회귀 & sklearn Pipeline

### 목적

과제 1·2에서 다룬 임금 예측 문제에 대해, **피처 엔지니어링 → ColumnTransformer·Pipeline → OOF·Voting 앙상블**까지 한 흐름으로 구현하는 것이 목표였다. CatBoost·선형모형·OneHot+트리/MLP 등 서로 다른 Pipeline 패턴을 비교한다.

### 주요 노트북

| 파일 | 내용 |
|------|------|
| `01_feature_engineering.ipynb` | 범주 인코딩·파생 변수·결측 처리 |
| `02_modeling_oof_ensemble.ipynb` | OOF, Voting, 가중 평균 제출 |
| `03_pipeline_catboost.ipynb` | CatBoost + Pipeline (PowerTransformer, OrdinalEncoder) |
| `04_pipeline_linear_models.ipynb` | Ridge / Lasso / ElasticNet Pipeline |
| `05_onehot_encoder_tree_mlp.ipynb` | OneHot + RandomForest·MLP 회귀 |

### 데이터

수업 제공 CSV(`X_train`, `y_train`, `X_test`)는 **저작권·서약으로 미포함**. [`01_course_salary_regression/data/README.md`](01_course_salary_regression/data/README.md)를 참고해 로컬에 배치한다.

---

## 02. KML Challenge 2023F (설문 응답 예측)

### 목적

*"각 패널에게 어떤 온라인 설문을 보내야 응답할까?"* — 패널 인구·가입 설문(SQ/DQ), 설문 메타(IR, LOI, TITLE, CPI 등)를 이용해 **STATUS(응답 여부)** 를 예측한다. 베이스라인(v1)에서 출발해 피처(응답률, 제목 토큰, KMeans·SHAP 등)·**Hard Voting·OOF**·하이퍼파라미터 튜닝·DNN 실험까지 확장하였다.

### 권장 읽기 순서

| 순서 | 노트북 | 설명 |
|------|--------|------|
| 0 | `00_official_baseline_v1.ipynb` | 대회 공식 베이스라인(LGBM) |
| 1 | `01_baseline_v2_best_snapshot.ipynb` | v2 + SHAP·KMeans 요약 스냅샷 |
| 2 | `02_feature_engineering.ipynb` | 피처 생성(1218) |
| 3 | `03_tuned_models_hard_voting.ipynb` | 튜닝 모델 + Hard Voting OOF |
| 4 | `04_reference_consolidated.ipynb` | 팀 참고·정리본(다중 모델·가중 평균) |
| 5 | `05_dnn_ensemble.ipynb` | Keras Tuner + 앙상블 시도 |
| 6 | `06_oof_hard_voting.ipynb` | OOF Hard Voting 변형 |
| 7 | `07_voting_ensemble_minimal.ipynb` | Voting 최소 예제 |

`notebooks/experiments/`에는 동일 대회의 **중간 실험본**(Optuna, MinMaxScaler, Log 변환, 시각별 제출본 등)을 보관한다.

### 제출물

[`02_kml_challenge_2023f/submissions/`](02_kml_challenge_2023f/submissions/)에 베이스라인·OOF Hard Voting 등 **32개 제출 CSV**가 타임스탬프 파일명으로 저장되어 있다.

### 데이터

대회 `train.csv` / `test.csv`는 [Kaggle 대회 페이지](https://www.kaggle.com/competitions/kml-challenge-2023f)에서 받아 [`02_kml_challenge_2023f/data/`](02_kml_challenge_2023f/data/)에 둔다. 자세한 안내는 [`data/README.md`](02_kml_challenge_2023f/data/README.md)를 본다.

---

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

- **수업 과제:** `01_course_salary_regression/notebooks/`에서 실행 (`../data/` 기준).
- **KML 대회:** `02_kml_challenge_2023f/notebooks/`에서 실행. `experiments/` 하위는 `../../data/` 기준.

---

## 디렉터리 구조

```
MachineLearningClass/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_course_salary_regression/
│   ├── data/
│   │   └── README.md
│   └── notebooks/
│       ├── 01_feature_engineering.ipynb
│       ├── 02_modeling_oof_ensemble.ipynb
│       ├── 03_pipeline_catboost.ipynb
│       ├── 04_pipeline_linear_models.ipynb
│       └── 05_onehot_encoder_tree_mlp.ipynb
└── 02_kml_challenge_2023f/
    ├── data/
    │   └── README.md
    ├── notebooks/
    │   ├── 00_official_baseline_v1.ipynb
    │   ├── 01_baseline_v2_best_snapshot.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_tuned_models_hard_voting.ipynb
    │   ├── 04_reference_consolidated.ipynb
    │   ├── 05_dnn_ensemble.ipynb
    │   ├── 06_oof_hard_voting.ipynb
    │   ├── 07_voting_ensemble_minimal.ipynb
    │   └── experiments/          # 중간 실험 노트북 (10개)
    └── submissions/              # 대회 제출 CSV (32개)
```

---

## 공개 범위·민감 정보

- 수업·대회 **원시 학습 데이터**는 저장소에 포함하지 않는다.
- 팀원 실명이 파일명에 있던 노트북은 **역할 기준 영문 파일명**으로 변경하였다.
- 수업 제출 규정(가상대학 제출·감점 조항 등)은 README에서 제거하고, 포트폴리오용 설명만 남겼다.

---

## 기술 스택

- **Python:** pandas, scikit-learn, category_encoders, LightGBM, CatBoost, XGBoost, Optuna, SHAP(일부 노트북), PyTorch/Keras( DNN 노트북)
- **R:** (본 저장소에는 미포함)
