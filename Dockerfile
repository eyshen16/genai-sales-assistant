FROM python:3.9.24-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py ./streamlit_app.py
COPY src ./src
COPY data/documents ./data/documents
COPY data/structured/compatibility_matrix.csv ./data/structured/compatibility_matrix.csv
COPY consulting/knowledge_source_registry.csv ./consulting/knowledge_source_registry.csv
COPY consulting/source_of_truth_mapping.csv ./consulting/source_of_truth_mapping.csv
COPY consulting/knowledge_domain_taxonomy.csv ./consulting/knowledge_domain_taxonomy.csv
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY start.sh ./start.sh

RUN chmod +x ./start.sh

EXPOSE 8501

CMD ["./start.sh"]
