FROM python:3.11-slim
WORKDIR /app

# copy dependency metadata + source
COPY pyproject.toml README.md /app/
COPY src /app/src

# install API deps
RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000
CMD ["uvicorn", "ppp_api.main:app", "--host", "0.0.0.0", "--port", "8000"]