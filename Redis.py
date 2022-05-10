
from flask import Flask, jsonify, render_template, request
import pandas as pd
import os
from functools import reduce


import redis, pickle


app = Flask(__name__)

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
    from collections import defaultdict

    for hashtag in hashtags:
        df = hashtag_metadata.loc[hashtag_metadata['search_hashtag'] == hashtag]
        hash_tags = defaultdict(int)
        for tags in df['hashtags']:
            for tag in tags:
                hash_tags[tag] += 1

        tag_recommender[hashtag] = sorted(hash_tags.items(), key=lambda item: item[1], reverse=True)[:15]
        
    return tag_recommender


def store_tag_redis(db):
    tag_recommender = store_tag_data(hashtags)

    

    ### Push the data into redis so, that we can take hashtags from cache ####

    #### push the username and password and activites into redis cache
    for hashtag in hashtags:
        db.set(hashtag,pickle.dumps(tag_recommender[hashtag]))
    

#store_tag_redis(db)



### get the recommended tags for a hashtag
@app.route('/redis/recommend', methods = ['POST'])
def recommend_tags(hashtag):
    details = pickle.loads(db.get(hashtag))

    import re
    recommendation = []   
    for tag in details:
        recommendation.append(tag[0])
    return json.dumps(recommendation)


app.run(host="0.0.0.0", port=5000,use_reloader=True, debug=True)