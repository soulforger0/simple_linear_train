FROM python:3.9-slim-buster

# Install the Google Cloud SDK and dependencies
RUN sudo apt-get install google-cloud-sdk

# Install gsutil
RUN gcloud components install gsutil

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
