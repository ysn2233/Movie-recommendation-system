from pre import data_preprocess
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json

_, _, data = data_preprocess()
data = data[['uid', 'mid', 'rating']]

def createUserItemDict(x):
    user_movie_rate_dict = defaultdict(dict)
    for _, row in x.iterrows():
        uid = str(row['uid'])
        mid = str(row['mid'])
        rating = int(row['rating'])
        user_movie_rate_dict[uid][mid] = rating
    return user_movie_rate_dict

user_moving_rating_dict = createUserItemDict(data)
print(user_moving_rating_dict)
f = open('user-item.json', 'w')
json.dump(dict(user_moving_rating_dict), f)
