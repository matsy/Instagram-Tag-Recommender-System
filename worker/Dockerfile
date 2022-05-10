FROM python:3.8

ADD worker-server.py /

RUN pip3 install --upgrade pandas pika jsonpickle requests flask img2pdf astrapy flask-cors flask-jwt-extended google-api-core google-api-python-client google-auth google-auth-httplib2 google-cloud google-cloud-core google-cloud-storage google-cloud-vision jsonify pillow

#COPY . /root/sentiment_analysis

CMD cd /root/worker && \
    python3 worker-server.py