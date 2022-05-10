FROM python:3.8

ADD rest-server.py /

RUN pip3 install --upgrade pandas pika jsonpickle requests flask flask-cors flask-jwt-extended 
RUN pip3 install pika redis json pickle uuid time re
RUN pip3 install numpy scipy tensorflow pyspark matplotlib hashlib

CMD mkdir /app
WORKDIR /app

COPY . /app

EXPOSE 5000
CMD cd /app && python3 rest-server.py