FROM gcr.io/google.com/cloudsdktool/google-cloud-cli

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
