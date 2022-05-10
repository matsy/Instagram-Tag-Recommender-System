FROM python:3.6-slim
LABEL MAINTAINER="arpitaambavane89"

RUN  pip3 install pika

WORKDIR /srv
COPY    logs.py /srv
CMD [ "python3", "logs.py"]
