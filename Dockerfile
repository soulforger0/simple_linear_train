FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    rm -rf /var/lib/apt/lists/*

# Install the Google Cloud SDK and gsutil
RUN curl https://sdk.cloud.google.com | bash && \
    exec $SHELL && \
    gcloud components install gsutil

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
