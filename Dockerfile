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

# Fighting profile (review R-K1): the organizers' launch command passes no HARNESS_* variables, so what was measured in
# the lab must be the default of the image. A variable given with `docker run -e` still wins (the mode profile uses setdefault).
# economy: budget is probably not enough for all tasks (call of 2026-11-09). Lab runs set their own env and do not use this file.
ENV HARNESS_RUN_MODE=economy     HARNESS_LIMITS=on     HARNESS_SANDBOX=scrub     HARNESS_HARD_CAP=on

# The organizers start it with: --entrypoint python image /app/run_agent.py --tasks-dir /data/tasks
ENTRYPOINT ["python", "/app/run_agent.py"]
