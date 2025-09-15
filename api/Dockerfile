FROM python:3.11-slim
WORKDIR /app

# copy just dependency metadata first for caching
COPY pyproject.toml /app/
COPY src /app/src
#copy README
COPY pyproject.toml README.md /app/
COPY src /app/src


# install API extras (brings in base + api deps)
RUN pip install --no-cache-dir -e ".[api]"

#expose port for API
EXPOSE 8000
CMD ["uvicorn", "ppp_api.main:app", "--host", "0.0.0.0", "--port", "8000"]