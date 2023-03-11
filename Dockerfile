FROM python:3.9-slim-buster

# Install the Google Cloud SDK and dependencies
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*

# Install gsutil
RUN gcloud components install gsutil

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
