# hypothesis-generation

## Overview

This project is designed to generate hypotheses for the causal relationship between genetic variants and phenotypes using various bioinformatics tools and machine learning models. The system integrates data from multiple sources, performs enrichment analysis, and uses a large language model (LLM) to summarize and generate hypotheses.

### Prerequisites

- Python 3.8+
- MongoDB
- SWI-Prolog
- Docker and Docker Compose

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/rejuve-bio/hypothesis-generation-demo.git
    cd hypothesis-generation-demo
    ```

2. Compile the Prolog KB (Biocypher export on disk). Use `config/kb.yaml` for Prolog KB **v2** and `config/kb_v3.yaml` for **v3**. Set `--path-prefix` to the root folder that contains those KB trees (the same directories you mount as `prolog_out_v2` / `prolog_out_v3` in `docker/docker-compose.yml`).
    ```sh
    python scripts/compile_kb.py --config-path config/kb.yaml --compile-script pl/compile.pl --path-prefix /mnt/hdd_1/biocypher-kg/output/human/prolog_out_v2 --hook-script pl/hook.pl
    python scripts/compile_kb.py --config-path config/kb_v3.yaml --compile-script pl/compile.pl --path-prefix /mnt/hdd_1/biocypher-kg/output/human/prolog_out_v3 --hook-script pl/hook.pl
    ```

3. Set up environment variables:
    Create a `.env` file in the project root and add the following variables:
    ```env
    MONGODB_URI=<your_mongodb_uri>
    DB_NAME=<your_db_name>
    OPENAI_API_KEY=<your_openai_api_key>
    ANTHROPIC_API_KEY=<your_anthropic_api_key>
    HF_TOKEN=<your_huggingface_token>
    JWT_SECRET=<your_jwt_secret>
    PREFECT_SERVICE_TOKEN=<prefect_service_jwt>
    ```

4. Run the Docker containers:
    ```sh
    docker compose -f docker/docker-compose.yml up --build
    ```

5. With the default `docker/docker-compose.yml` port mappings, the API (FastAPI) is on host port **5008**, Prolog on **4242**, and Prefect on **4292**.
