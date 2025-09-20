FROM python:3.11-slim
WORKDIR /app

# copy just dependency metadata first for caching
COPY pyproject.toml /app/
COPY src /app/src
# copy README
COPY pyproject.toml README.md /app/
COPY src /app/src

# Install your package; this creates console scripts like ppp-etl-acs-pop
RUN python -m pip install --upgrade pip && \
    pip install -e ".[etl]"

# Simple entrypoint that chains the ETLs
COPY docker/etl_entrypoint.sh /usr/local/bin/etl_entrypoint.sh
RUN chmod +x /usr/local/bin/etl_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/etl_entrypoint.sh"]