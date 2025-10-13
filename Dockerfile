##########################################
# Stage 1: Builder (Node used only to run the Python generator)
##########################################
FROM node:18-bullseye AS builder
WORKDIR /app

# Copy Prisma schema files
COPY prisma /app/prisma
COPY prisma/schema/schema_combined.prisma /app/prisma/schema_combined.prisma

# Install Python + pip so we can use the Python Prisma generator
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Python Prisma + dotenv (includes the Python generator)
RUN pip3 install --no-cache-dir prisma==0.15.0 python-dotenv

# Generate Prisma Python client (code)
RUN python3 -m prisma generate --schema=prisma/schema/schema_combined.prisma

# Fetch the matching Prisma engine binary using the Python CLI
RUN python3 -m prisma fetch --schema=prisma/schema/schema_combined.prisma --platform debian-openssl-3.5.x || true


##########################################
# Stage 2: Runtime (pure Python)
##########################################
#FROM python:3.11-slim
FROM python:3.11-bullseye
WORKDIR /app

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random

# Copy and install dependencies
COPY minreqs.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --retries 3 -r minreqs.txt && \
    pip install --no-cache-dir "uvicorn[standard]"

# Copy generated Prisma Python client from builder
COPY --from=builder /usr/local/lib/python3.9/dist-packages/prisma /usr/local/lib/python3.11/site-packages/prisma

# Copy Prisma binaries and schema
COPY --from=builder /root/.cache/prisma-python /root/.cache/prisma-python
COPY --from=builder /app/prisma /app/prisma

# Copy your source code
COPY . .

EXPOSE 8000

# Launch your FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--http", "httptools", "--workers", "1"]

