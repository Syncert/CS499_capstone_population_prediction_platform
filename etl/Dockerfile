FROM python:3.11-slim
WORKDIR /app

# copy just dependency metadata first for caching
COPY pyproject.toml /app/
COPY src /app/src
# copy README
COPY pyproject.toml README.md /app/
COPY src /app/src


# install ETL extras (base + etl deps)
RUN pip install --no-cache-dir -e ".[etl]"

# run a module as the container's command
CMD ["python", "-m", "ppp_etl.jobs.sample_census_ingest"]