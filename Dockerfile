FROM python:3.10-slim AS build

WORKDIR /app

COPY requirements.txt .

RUN python -m venv venv && \
    /app/venv/bin/python -m pip install --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=build /app/venv /app/venv

ENV PATH="/app/venv/bin:$PATH"

COPY . .

CMD ["python", "training.py"]