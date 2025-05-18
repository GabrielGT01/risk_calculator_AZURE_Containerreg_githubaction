
FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

RUN apt update -y && apt install -y ca-certificates curl gnupg
# Azure CLI installation can be added here if needed:
# RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5001

CMD ["python3", "app.py"]
