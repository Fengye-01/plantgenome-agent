# ============================================================
# PlantGenome Agent - Dockerfile
# ============================================================
# 前端只需要 Streamlit、HTTP 客户端和表格展示依赖，不加载模型运行栈。
FROM python:3.12-slim AS frontend
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10
COPY requirements-frontend.txt .
RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements-frontend.txt
COPY frontend ./frontend
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# 后端与 Worker 共用完整运行环境。
FROM python:3.12-slim AS backend
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0
COPY requirements-backend.txt .
RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements-backend.txt
COPY . .
RUN mkdir -p data/pdfs data/chroma docs/screenshots
EXPOSE 8000
CMD ["python", "-m", "app.api.main"]
