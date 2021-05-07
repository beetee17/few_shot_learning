FROM python:3.9.4

WORKDIR /docker

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY ./app ./app

CMD ["python", "./app/docker.py"]