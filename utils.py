import math
import pickle
import pandas as pd


# load movie info and convert to a dict
def get_movie_info_dict(filename):
    movies_title = ['MovieID', 'Title']
    movies_frame = pd.read_csv(filename, sep='|', header=None, names=movies_title, usecols=[0, 1], engine='python')
    movie_info = {}
    for i, movie in movies_frame.iterrows():
        movie_info[int(movie['MovieID'])] = movie['Title']
    return movie_info


# load rating frame
def get_rating_frame(filename):
    rating_columns = ['uid', 'mid', 'rating']
    ratings_frame = pd.read_csv(filename, sep='\t', header=None, names=rating_columns, usecols=[0, 1, 2],
                                engine='python')
    return ratings_frame


def crete_search_table(frame):
    movie_search_dict = {}
    user_search_dict = {}
    for i, row in frame.iterrows():
        uid = row['uid']
        mid = row['mid']
        rating = row['rating']
        # create a rating search dict easier to find the user and its rating by given a movie
        ur = (uid, rating)
        if mid in movie_search_dict:
            movie_search_dict[mid].append(ur)
        else:
            movie_search_dict[mid] = [ur]

        # create a rating search dict easier to find the movie and its rating by given a user
        mr = (mid, rating)
        if uid in user_search_dict:
            user_search_dict[uid].append(mr)
        else:
            user_search_dict[uid] = [mr]
    return movie_search_dict, user_search_dict


def split(frame, split_ratio):
    n = frame.shape[0]
    train_size = int(split_ratio*n)
    return frame[:train_size], frame[train_size:]


def pearson_sim(user1, user2):
    avg1 = 0.0
    avg2 = 0.0
    n = 0
    for m1, r1 in user1:
        for m2, r2 in user2:
            if m1 == m2:
                avg1 += r1
                avg2 += r2
                n += 1
    avg1 /= n
    avg2 /= n
    sum1 = 0.0
    sum2 = 0.0
    numerator = 0.0
    for m1, r1 in user1:
        for m2, r2 in user2:
            if m1 == m2:
                numerator += (r1 - avg1) * (r2 - avg2)
                sum2 += (r2 - avg2) ** 2
                sum1 += (r1 - avg1) ** 2

    if numerator == 0.0:
        return 0
    denominator = math.sqrt(sum1 * sum2)
    return numerator / denominator


def adjusted_cosine_sim(movie1, movie2):
    avg1 = 0.0
    avg2 = 0.0
    for u1, r1 in movie1:
        avg1 += r1
    for u2, r2 in movie2:
        avg2 += r2
    avg1 /= len(movie1)
    avg2 /= len(movie2)
    sum1 = 0.0
    sum2 = 0.0
    numerator = 0.0
    for u1, r1 in movie1:
        for u2, r2 in movie2:
            if u1 == u2:
                numerator += (r1 - avg1) * (r2 - avg2)
        sum1 += (r1 - avg1) ** 2
    for u2, r2 in movie2:
        sum2 += (r2 - avg2) ** 2
    if numerator == 0.0:
        return 0
    denominator = math.sqrt(sum1 * sum2)
    return numerator / denominator


def normalization(dists):
    norm_dists = []
    sum = 0.0
    for _, sim in dists:
        sum += sim
    for uid, sim in dists:
        norm_dists.append((uid, sim / sum))
    return norm_dists


def serialize_dict(d, filename):
    f = open(filename, 'wb+')
    pickle.dump(d, f)
    f.close()


def deserialize_dict(filename):
    f = open(filename, 'rb')
    d = pickle.load(f)
    f.close()
    return d


def get_user_avg(user_search_dict):
    user_avg = {}

    for uid in user_search_dict:
        user_avg[uid] = 0
        for _, r in user_search_dict[uid]:
            user_avg[uid] += r
        user_avg[uid] /= len(user_search_dict[uid])
    return user_avg


def get_movie_avg(movie_search_dict):
    movie_avg = {}

    for mid in movie_search_dict:
        movie_avg[mid] = 0
        for _, r in movie_search_dict[mid]:
            movie_avg[mid] += r
        movie_avg[mid] /= len(movie_search_dict[mid])
    return movie_avg