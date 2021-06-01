FROM python:3.9.4

WORKDIR /docker

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY ./app ./app

CMD ["python", "./app/main.py"]


# cd C:\Users\Admin\few_shot_learning\aircraft\docker
# docker build -t <image name> .
# docker run -p 8000:8000 <image name>