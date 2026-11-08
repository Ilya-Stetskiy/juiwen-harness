# Base image tag is pinned; all dependencies are pinned in requirements.txt.
FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so the layer is cached; everything is downloaded at build time only.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY run_agent.py .
COPY harness harness
COPY profiles profiles

# The organizers start it with: --entrypoint python image /app/run_agent.py --tasks-dir /data/tasks
ENTRYPOINT ["python", "/app/run_agent.py"]
