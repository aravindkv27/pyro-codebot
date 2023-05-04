FROM python:3.11.0a4-slim-buster

ADD . ./app

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]