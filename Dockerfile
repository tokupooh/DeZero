FROM python:3.8
USER root

RUN apt-get update && apt-get install -y graphviz

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
