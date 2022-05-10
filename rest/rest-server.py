##
from flask import Flask, request, Response, jsonify
import platform, hashlib
import io, os, sys
import pika, redis, json, pickle, uuid, time, re

import sys
sys.path.append("..")

from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager
from flask_cors import CORS
from functools import reduce
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from sklearn.model_selection import train_test_split
from functions import prepare_image, extract_features
from collections import defaultdict

np.random.seed(0)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

app.config["JWT_SECRET_KEY"] = "hvhvgcgc675765bhbj"  # Change this!
jwt = JWTManager(app)

# credential_path = "keyFile-credentials.json"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

rabbitMQHost = os.getenv("RABBITMQ_HOST") or "localhost"
print("Connecting to rabbitmq({})".format(rabbitMQHost))

###############################################################
#                     Redis code                              #
###############################################################
redisHost = os.getenv("REDIS_HOST") or "localhost"
db = redis.Redis(host=redisHost, db=1)  
print("Connecting to redis({})".format(redisHost))
json_file_names = os.listdir('metadata')

# Remove the 5 char .json file ending to isolate hashtag name
hashtags = [hashtag[:-5] for hashtag in json_file_names]

# remove '.DS_', '.ipynb_checkp'
non_hashtags = ['.DS_', '.ipynb_checkp']
for non_hashtag in non_hashtags:
    try:
        hashtags.remove(non_hashtag)
    except:
        pass # If we can't remove it, it's already gone

# convert hashtag metadata into dataframe
hashtag_metadata = []
for hashtag in hashtags: 
    hashtag_metadata.append(pd.read_json(f'metadata/{hashtag}.json'))
hashtag_metadata = reduce(lambda x, y: pd.concat([x, y]), hashtag_metadata)
pd.DataFrame.reset_index(hashtag_metadata, drop=True, inplace=True)
hashtag_metadata.tail()

#### store the tag data in redis, so that we can use it as a cache.
def store_tag_data(hashtags):
    tag_recommender = {}
    #print(hashtag_metadata)
    for hashtag in hashtags:
        df = hashtag_metadata.loc[hashtag_metadata['search_hashtag'] == hashtag]
        hash_tags = defaultdict(int)
        for tags in df['hashtags']:
            for tag in tags:
                hash_tags[tag] += 1
        tag_recommender[hashtag] = sorted(hash_tags.items(), key=lambda item: item[1], reverse=True)[:15]
    #print(tag_recommender)
    return tag_recommender

def store_tag_redis(db):
    tag_recommender = store_tag_data(hashtags)
    #print(tag_recommender)
    ### Push the data into redis so, that we can take hashtags from cache ####
    #### push the username and password and activites into redis cache
    for hashtag in hashtags:
        db.set(hashtag,pickle.dumps(tag_recommender[hashtag]))

###############################################################
#                          ML Code                           #
###############################################################
file_name = "./all_hashtags.pkl"
pics = pd.read_pickle("./tag_pics.pkl")
all_hashtags = pd.read_pickle(file_name)
hashtag_metadata = pd.read_pickle("./hashtag_metadata.pkl")
hashtag_lookup = pd.read_pickle("./hashtag_lookup.pkl")

### create our neural network ###
img_shape = (160, 160, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
neural_network = tf.keras.Sequential([
  base_model,
  global_average_layer,
])

### ALS MODEL #####            
spark = SparkSession.builder.master('local').getOrCreate()

als_model = ALSModel.load("./ALS_MODEL/als/")
recs = als_model.recommendForAllUsers(numItems=10).toPandas()
img_features = als_model.userFactors.toPandas()
hashtag_features = als_model.itemFactors.toPandas()

hashtag_index = list(all_hashtags)
def lookup_hashtag(hashtag_id):
    return hashtag_index[hashtag_id]

def lookup_hashtag_recs(rec_scores):
    return [lookup_hashtag(rec) for (rec, score) in rec_scores]

def recommender_dataframe(recs, hashtag_lookup, als_model):
  recs['recommended_hashtags'] = recs['recommendations'].apply(lookup_hashtag_recs)
  recs.index = recs['image_id']
  recs = recs.join(hashtag_metadata, how='left')[['recommendations',
                                                  'recommended_hashtags',
                                                  'hashtags',
                                                  'image_local_name',
                                                  'search_hashtag']]
  recs.drop('recommendations', axis=1, inplace=True)
  image_factors = als_model.userFactors.toPandas()
  image_factors.index = image_factors['id']
  recs.join(image_factors);

  # Add deep features information to recs dataframe
  recs_deep = recs.join(pics, on='image_local_name', how='inner')

  hashtags_df = pd.DataFrame.from_dict(hashtag_lookup, orient='index')
  hashtags_df.head()
  hashtags_df = hashtags_df.reset_index()
  hashtags_df.columns = ['hashtag', 'id']
  hashtags_df.index = hashtags_df['id']
  hashtags_df.drop('id', axis=1, inplace=True)
  

  # Only use certain columns
  recs_deep_clean = recs_deep[['image_local_name', 'hashtags', 'deep_features']]

  img_features.index = img_features['id']
  img_features.drop(['id'], axis=1)

  # Add image feature into dataframe
  recommender_df = recs_deep_clean.join(img_features, how='inner')
  
  return recommender_df, hashtags_df


recommender_df, hashtags_df = recommender_dataframe(recs,hashtag_lookup, als_model)
                        
# Function that finds k nearest neighbors by cosine similarity
def find_neighbor_vectors(image_path, k=5, recommender_df=recommender_df):
    """Find image features (user vectors) for similar images."""
    prep_image = prepare_image(image_path, where='local')
    pics = extract_features(prep_image, neural_network)
    rdf = recommender_df.copy()
    rdf['dist'] = rdf['deep_features'].apply(lambda x: cosine(x, pics))
    rdf = rdf.sort_values(by='dist')
    return rdf.head(k)


def generate_hashtags(image_path):
    fnv = find_neighbor_vectors(image_path, k=5, recommender_df=recommender_df)
    # Find the average of the 5 user features found based on cosine similarity.
    features = []
    for item in fnv.features.values:
        features.append(item)

    avg_features = np.mean(np.asarray(features), axis=0)
    
    # Add new column to the hashtag features which will be the dot product with the average image(user) features
    hashtag_features['dot_product'] = hashtag_features['features'].apply(lambda x: np.asarray(x).dot(avg_features))

    # Find the 10 hashtags with the highest feature dot products
    final_recs = hashtag_features.sort_values(by='dot_product', ascending=False).head(10)
    # Look up hashtags by their numeric IDs
    output = []
    for hashtag_id in final_recs.id.values:
        output.append(hashtags_df.iloc[hashtag_id]['hashtag'])
    return output


def show_results(test_image):
    img = mpimg.imread(f'{test_image}.jpg')
    '''
    plt.figure(figsize=(9, 9))
    plt.title(f'Original Hashtag: {test_image.upper()}', fontsize=32)        
    plt.imshow(img)
    '''
    recommended_hashtags = generate_hashtags(f'{test_image}.jpg')
    print(', '.join(recommended_hashtags))
    return recommended_hashtags

###############################################################
#                     RabbitMQ Part                           #
###############################################################
def enqueueDataToLogsExchange(message,messageType):
    rabbitMQ = pika.BlockingConnection(
            pika.ConnectionParameters(host=rabbitMQHost))
    rabbitMQChannel = rabbitMQ.channel()

    rabbitMQChannel.exchange_declare(exchange='logs', exchange_type='topic')

    infoKey = f"{platform.node()}.rest.info"
    debugKey = f"{platform.node()}.rest.debug"

    if messageType == "info":
        key = infoKey
    elif messageType == "debug":
        key = debugKey

    rabbitMQChannel.basic_publish(
        exchange='logs', routing_key='logs', body=json.dumps(message))

    print(" [x] Sent %r:%r" % (key, message))

    rabbitMQChannel.close()
    rabbitMQ.close()

class enqueueWorker(object):
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=rabbitMQHost))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.onResponse,
            auto_ack=True)
    
    def onResponse(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    # Producer is rest-server and sending data to RabbitMQ Queue 'toWorkerQueue' ie Consumer is worker-server
    def enqueueDataToWorker(self,message):
        self.response = None
        self.corr_id = str(uuid.uuid4())    
        self.channel.basic_publish(
            exchange='', routing_key='toWorkerQueue',properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ), 
            body=json.dumps(message))
        while self.response is None:
            self.connection.process_data_events()
        #print(self.response)        
        return str(self.response.decode('utf-8'))
        #return Response(response=json.dumps(self.response), status=200, mimetype="application/json")
        #print(" [x] Sent %r:%r" % ('toWorker', message))

    
###############################################################
#                 REST Request Handling                       #
###############################################################
@app.route('/api/upload/captionimage', methods=['POST','GET'])
def handle_captionform():
    # Adding code for rest handling - Trail
    try:
        print(" Inside Caption API Upload ")
        enqueueDataToLogsExchange('Call to api /api/upload/captionimage','info')
        print(request.files['file'])
        file = request.files['file']
 
        dataToWorker = enqueueWorker()
        response1 = dataToWorker.enqueueDataToWorker(file)
        print(response1)
        # billvalue = re.sub('[^\d\.]', '',response1)
        # response = {'bill_value':str(billvalue)}
       
    
        # client = storage.Client()
        # BUCKET_NAME = 'projectexpensegenerator'
        # bucket = client.get_bucket(BUCKET_NAME)
        # destination_blob_name = name2
        # blob1 = bucket.blob(destination_blob_name)
        # blob1.upload_from_filename(name2)
 
        data = {'user_details':'abc'}
        
        dataToWorker = enqueueWorker()
        response1 = dataToWorker.enqueueDataToWorker(data) #queue
        print(response1)
  
        response = "worked succesfully"
        return Response(response, status=200, mimetype="application/json")

    except Exception as e:
        print("Something went wrong" + str(e))
        enqueueDataToLogsExchange('Error occured in api /api/upload/captionimage','info')
        return Response(response="Something went wrong!", status=500, mimetype="application/json")

@app.route('/api/upload/tagimage', methods=['GET','POST'])
def handle_tagform():
    try:
        # Adding code for rest handling - Trail
        print(" Inside Tag API Upload ")
        enqueueDataToLogsExchange('Call to api /api/upload/tagimage','info')
        print(request.files['file'])
        file = request.files['file']
 
        dataToWorker = enqueueWorker()
        response1 = dataToWorker.enqueueDataToWorker(file)
        print(response1)
        # billvalue = re.sub('[^\d\.]', '',response1)
        # response = {'bill_value':str(billvalue)}
       
    
        # client = storage.Client()
        # BUCKET_NAME = 'projectexpensegenerator'
        # bucket = client.get_bucket(BUCKET_NAME)
        # destination_blob_name = name2
        # blob1 = bucket.blob(destination_blob_name)
        # blob1.upload_from_filename(name2)
 
        data = {'user_tag':'Tagabc'}
        
        dataToWorker = enqueueWorker()
        response1 = dataToWorker.enqueueDataToWorker(data) #queue
        print(response1)
  
        response = "worked succesfully"
        return Response(response, status=200, mimetype="application/json")
    except Exception as e:
        print("Something went wrong" + str(e))
        enqueueDataToLogsExchange('Error occured in api /api/upload/tagimage','info')
        return Response(response="Something went wrong!", status=500, mimetype="application/json")

# start flask app
app.run(host="0.0.0.0", port=5000,debug=True)
