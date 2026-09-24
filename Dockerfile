# GeoFlare -- SIH2K26 production image
# CPU-only torch (free-tier friendly). Models + deps are pre-baked.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch CPU wheel first (keeps image far smaller than the CUDA build).
# cffi/build-base are needed to compile the psycopg binary wheel if Redis/compilers are required.
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# App + models + frontend (data/raw images excluded via .dockerignore)
COPY . .

WORKDIR /app/src

EXPOSE ${PORT:-10000}

# Render sets $PORT; default 10000 keeps local `docker run` simple.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 300 server:app"]