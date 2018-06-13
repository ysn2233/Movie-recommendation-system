from utils import *
import math

def evaluate_baselines():
    train = get_rating_frame('dataset/ml-100k/u1.base')
    test = get_rating_frame('dataset/ml-100k/u1.test')

    user_avg = {}
    movie_avg = {}

    movie_search_dict, user_search_dict = crete_search_table(train)

    for mid in movie_search_dict:
        movie_avg[mid] = 0
        for _, r in movie_search_dict[mid]:
            movie_avg[mid] += r
        movie_avg[mid] /= len(movie_search_dict[mid])

    for uid in user_search_dict:
        user_avg[uid] = 0
        for _, r in user_search_dict[uid]:
            user_avg[uid] += r
        user_avg[uid] /= len(user_search_dict[uid])

    error_sum_movie = 0
    error_sum_user = 0
    for i, row in test.iterrows():
        uid = row['uid']
        mid = row['mid']
        r = row['rating']
        if mid in movie_avg:
            error_sum_movie += (movie_avg[mid] - r) ** 2
        if uid in user_avg:
            error_sum_user += (user_avg[uid] - r) ** 2

    rmse_avg_movie = math.sqrt(error_sum_movie / test.shape[0])
    rmse_avg_user = math.sqrt(error_sum_user / test.shape[0])

    return rmse_avg_movie, rmse_avg_user

