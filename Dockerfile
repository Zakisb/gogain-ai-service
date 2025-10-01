FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random
COPY minreqs.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --retries 3 -r minreqs.txt
RUN pip install "uvicorn[standard]"


COPY . .

EXPOSE 8000

# Original uvicorn command with streaming fixes
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--http", "httptools", "--workers", "1"]

# Alternative: Use hypercorn if uvicorn still buffers
# CMD ["hypercorn", "app.main:app", "--bind", "0.0.0.0:8000", "--worker-class", "asyncio"]



