# HF Spaces는 포트 7860 고정
# backend/ 폴더 기준으로 빌드됨

FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치 (캐시 효율)
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# 나머지 코드 복사
COPY . .

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
