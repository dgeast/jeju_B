# Sales Data Analysis & Decision Dashboard

## 📌 프로젝트 소개
본 프로젝트는 판매 데이터를 종합적으로 분석하여 매출 증대 전략을 수립하기 위한 **올인원 데이터 분석 및 대시보드 솔루션**입니다.
데이터 전처리부터 EDA(탐색적 데이터 분석), K-Means 클러스터링, 그리고 Streamlit 기반의 인터랙티브 대시보드까지 포함되어 있습니다.

## 📂 파일 구조
```
salesdata_B/
├── data/
│   ├── data_total_non.csv          # 원본 데이터
│   └── data_classified_unified.csv # 전처리 및 분류 완료된 데이터
├── docs/
│   ├── comprehensive_analysis_report.md # 종합 분석 보고서 (자동 생성)
│   └── github_deployment_guide.md       # GitHub 배포 가이드
├── app_decision_dashboard.py       # Streamlit 대시보드 메인 앱
├── eps_clustering.py               # EDA 및 보고서 생성 자동화 스크립트
├── preprocess_unified.py           # 데이터 전처리 및 규칙 기반 분류 스크립트
├── requirements.txt                # 필수 라이브러리 목록
└── README.md                       # 프로젝트 설명서 (본 파일)
```

## 🚀 사용 방법

### 1. 환경 설정
필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리 (Preprocessing)
원본 데이터(`data_total_non.csv`)를 정제하고, 상품명/옵션/등급 분류 규칙을 적용하여 `data_classified_unified.csv`를 생성합니다.
```bash
python preprocess_unified.py
```
> **적용된 규칙**:
> - 감귤류 통합 (하우스/노지/조생 등 → '감귤')
> - 옵션 분류 (이벤트, 선물세트, 일반, 가정용)
> - 등급 단순화 (로얄과, 기타)

### 3. 분석 보고서 생성 (Analysis)
정제된 데이터를 기반으로 EDA를 수행하고 클러스터링 분석 결과를 포함한 마크다운 보고서를 자동 생성합니다.
```bash
python eda_clustering.py
```
- 결과물: `/docs/comprehensive_analysis_report.md`

### 4. 대시보드 실행 (Dashboard)
Streamlit 대시보드를 실행하여 브라우저에서 분석 결과를 시각적으로 탐색합니다.
```bash
python -m streamlit run app_decision_dashboard.py
```
- **주요 기능**: 매출 개요, 고객 세분화(3D), 채널/셀러 분석, 상품 상세 분석, 시계열 분석, VIP 고객 관리

## 📊 주요 분석 내용
- **고객 세분화**: 구매 패턴(금액, 수량, 무게)에 따른 K-Means 클러스터링 및 그룹별 공략 전략 제안
- **상품 분석**: 등급별, 옵션별 매출 기여도 및 트렌드 파악
- **채널/셀러**: 주요 판매 경로 및 우수 셀러 성과 분석

## 🛠 배포
GitHub 및 Streamlit Cloud 배포 방법은 `docs/github_deployment_guide.md`를 참고하세요.
