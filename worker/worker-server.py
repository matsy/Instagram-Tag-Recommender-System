#
# Worker server
#
import pickle
import platform
import io
import os
import sys
import pika
import redis
import hashlib
import json
import requests
import re
# from google.cloud import vision
# from google.cloud import storage
# import img2pdf
#from PIL import Image
# import google.auth
# import google.oauth2.service_account as service_account
# import uuid

# hostname = platform.node()
## Configure test vs. production
rabbitMQHost = os.getenv("RABBITMQ_HOST") or "localhost"

print(f"Connecting to rabbitmq({rabbitMQHost})")

## Set up rabbitmq connection
rabbitMQ = pika.BlockingConnection(
        pika.ConnectionParameters(host=rabbitMQHost))
rabbitMQChannel = rabbitMQ.channel()
toWorkerResult = rabbitMQChannel.queue_declare(queue='toWorkerQueue')

# credential_path = "keyFile-credentials.json"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path


def enqueueDataToLogsExchange(message,messageType):
    rabbitMQ = pika.BlockingConnection(
            pika.ConnectionParameters(host=rabbitMQHost))
    rabbitMQChannel = rabbitMQ.channel()

    rabbitMQChannel.exchange_declare(exchange='logs', exchange_type='topic')

    infoKey = f"{platform.node()}.worker.info"
    debugKey = f"{platform.node()}.worker.debug"

    if messageType == "info":
        key = infoKey
    elif messageType == "debug":
        key = debugKey

    rabbitMQChannel.basic_publish(
        exchange='logs', routing_key='logs', body=json.dumps(message))

    print(" [x] Sent %r:%r" % (key, message))

    rabbitMQChannel.close()
    rabbitMQ.close()


def callback(ch, method, properties, body):
    enqueueDataToLogsExchange('Worker processing images','info')
    print("rest to worker working")
    response = "worker to rest working"
    ch.basic_publish(exchange='',
                     routing_key=properties.reply_to,
                     properties=pika.BasicProperties(correlation_id = properties.correlation_id), body=str(response))
    
    ch.basic_ack(delivery_tag=method.delivery_tag)
  



print("Waiting for messages:")
rabbitMQChannel.basic_qos(prefetch_count=1)
rabbitMQChannel.basic_consume(queue='toWorkerQueue', on_message_callback=callback)
rabbitMQChannel.start_consuming()
