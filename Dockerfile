FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d

ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="VascFusion" \
      org.opencontainers.image.description="Reproducible inference environment for the MICCAI 2026 GAVE2 Challenge solution" \
      org.opencontainers.image.url="https://github.com/KawhiQaQ/GAVE2-Challenge-8th-Solution" \
      org.opencontainers.image.source="https://github.com/KawhiQaQ/GAVE2-Challenge-8th-Solution" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    VASCFUSION_ROOT=/opt/vascfusion \
    PYTHONPATH=/opt/vascfusion/code/gave2_solution

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${VASCFUSION_ROOT}

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY LICENSE THIRD_PARTY_NOTICES.md README.md README_zh-CN.md ./
COPY configs ./configs
COPY code ./code
COPY weights/README.md ./weights/README.md

RUN chmod +x code/run_inference.sh code/docker_entrypoint.sh \
    && python code/verify_runtime.py

ENTRYPOINT ["/opt/vascfusion/code/docker_entrypoint.sh"]
CMD ["help"]
