FROM gcr.io/google.com/cloudsdktool/google-cloud-cli

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y python

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
