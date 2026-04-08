# Application image: lightweight layer built on top of hypothesis-base
FROM hypothesis-base:latest

WORKDIR /app

# install uv for Python dependency management
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# copy project metadata and build the flask venv
COPY pyproject.toml .
ENV UV_PROJECT_ENVIRONMENT=/opt/flask-venv
ENV PATH="/opt/flask-venv/bin:$PATH"
RUN uv sync --no-cache --extra r-integration

# application source code
COPY . .
RUN chmod +x /app/entrypoint.sh

# create separate harmonizer venv
RUN python3 -m venv /opt/harmonizer-venv
ENV UV_HARMONIZER_VENV=/opt/harmonizer-venv
RUN uv pip install --python=/opt/harmonizer-venv/bin/python -r gwas-sumstats-harmoniser/environments/requirements.txt

# minimal harmonizer deps in main venv
RUN uv pip install 'duckdb>=0.9.2' 'pyliftover>=0.4'

ENTRYPOINT ["/app/entrypoint.sh"]
EXPOSE 5000
