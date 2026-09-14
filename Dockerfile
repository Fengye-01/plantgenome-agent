# ============================================================
# PlantGenome Agent - Dockerfile
# ============================================================
# 基础镜像：Python 3.12 slim（轻量级）
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（pymupdf / sentence-transformers 可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt，利用 Docker 缓存层
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p data/pdfs data/chroma docs/screenshots

# 暴露端口
# 8000: FastAPI 后端
# 8501: Streamlit 前端
EXPOSE 8000 8501

# 默认启动命令（docker-compose 中会分别覆盖 backend 和 frontend 的 command）
# 后端启动：python -m app.api.main
# 前端启动：streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
CMD ["python", "-m", "app.api.main"]
