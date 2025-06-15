# colorlog

# 🎨 감정 추정 시스템 (Emotion Estimation System)

이 프로젝트는 이미지의 색상 조합을 바탕으로 감정을 추정하는 웹 애플리케이션입니다.  
Flask 백엔드와 OpenCV 기반의 K-means 알고리즘을 활용하여, 업로드한 3장의 이미지에서 주요 감정을 분석합니다.

---

## 📌 주요 기능

- 세 장의 이미지를 업로드하면, 주요 감정 두 가지를 추출해 결과로 보여줍니다.
- 색상 분석 기반 K-means 클러스터링
- Flask 웹 서버로 간편하게 웹에서 사용 가능

---

## ⚙️ 사용 기술

- Python 3
- Flask
- OpenCV
- Numpy
- Matplotlib

---

## 🚀 실행 방법

1. 프로젝트 클론

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. 가상환경 생성 및 설치

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

3. 서버 실행

python server.py

4. 웹 브라우저에서 http://127.0.0.1:5000 접속

📁 디렉토리 구조

emotion-project/
├── server.py                 # Flask 서버 및 감정 분석 로직
├── final_project.ipynb       # 프로젝트 분석 및 테스트 기록
├── templates/
│   ├── page1.html            # 이미지 업로드 페이지
│   └── result.html           # 결과 페이지
├── static/
│   ├── style.css
├── image/
│   └── .gitkeep              # 이미지 저장용 (빈 폴더 유지용)
├── requirements.txt          # 필요한 파이썬 패키지
└── README.md                 # 프로젝트 설명서

🙋‍♀️ 제작자
작성자: 안진영, 박지혜

---

### ✅ `requirements.txt`

```txt
Flask==2.3.2
opencv-python==4.9.0.80
numpy==1.24.3
matplotlib==3.7.1
