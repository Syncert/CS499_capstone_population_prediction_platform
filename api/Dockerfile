# api/Dockerfile
FROM python:3.11-slim
WORKDIR /app

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends curl jq \
 && rm -rf /var/lib/apt/lists/*

# Copy metadata + source first for better caching
COPY pyproject.toml README.md /app/
COPY src /app/src

# Install ALL extras so ETL/ML CLIs are available in this image
RUN pip install --no-cache-dir -e ".[api,etl,ml]"

# Create a non-root user and run the app as that user
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "ppp_api.main:app", "--host", "0.0.0.0", "--port", "8000"]