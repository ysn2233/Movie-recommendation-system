import numpy as np
import pandas as pd
import os

def data_preprocess_100k(datapath):
    """
        Load and preprocess data
    """

    # load user data
    user_columns = ['uid', 'age', 'gender', 'occupation']
    filename = os.path.join(datapath, 'u.user')
    users = pd.read_csv(filename, sep='|', header=None, names=user_columns, usecols=[0, 1, 2, 3], engine='python')

    # update user's gender to int
    gender_map = {'F': 0, 'M': 1}
    users.replace(gender_map, inplace=True)

    # update user's age range to int
    age_map = {age: i for i, age in enumerate(sorted(set(users['age'])))}
    users.replace(age_map, inplace=True)

    users['age'] = np.where(users['age'] <=16, 0, users['age'])
    users['age'] = np.where(np.logical_and(users['age'] > 16, users['age'] <= 25) , 1, users['age'])
    users['age'] = np.where(np.logical_and(users['age'] > 25, users['age'] <= 32), 2, users['age'])
    users['age'] = np.where(np.logical_and(users['age'] > 32, users['age'] <= 39), 3, users['age'])
    users['age'] = np.where(np.logical_and(users['age'] > 39, users['age'] <= 46), 4, users['age'])
    users['age'] = np.where(np.logical_and(users['age'] > 46, users['age'] <= 53), 5, users['age'])
    users['age'] = np.where(np.logical_and(users['age'] > 53, users['age'] <= 60), 6, users['age'])
    users['age'] = np.where(users['age'] >= 60, 7, users['age'])

    # convert occupation to digits
    filename = os.path.join(datapath, 'u.occupation')
    occ_frame = pd.read_csv(filename, names=['occupation'])
    occ_map = {}
    for i, row in occ_frame.iterrows():
        occ_map[row['occupation']] = i
    users.replace(occ_map, inplace=True)

    # load movie data
    filename = os.path.join(datapath, 'u.item')
    movie_columns = ['mid', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    movies = pd.read_csv(filename, sep='|', header=None, names=movie_columns,
                         usecols=[0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], engine='python')

    # load train rating data
    filename = os.path.join(datapath, 'u1.base')
    rating_columns = ['uid', 'mid', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='\t', header=None, names=rating_columns, usecols=[0,1,2], engine='python')

    frame = pd.merge(pd.merge(users, ratings), movies)
    X_train = frame.drop(columns='rating', axis=1)
    y_train = frame['rating']

    # load test rating data
    filename = os.path.join(datapath, 'u1.test')
    rating_columns = ['uid', 'mid', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='\t', header=None, names=rating_columns, usecols=[0,1,2], engine='python')

    frame = pd.merge(pd.merge(users, ratings), movies)
    X_test = frame.drop(columns='rating', axis=1)
    y_test = frame['rating']

    return X_train, y_train, X_test, y_test


def data_preprocess_1m():
    """
    Load and preprocess data
    """

    # load user data
    user_columns = ['uid', 'gender', 'age', 'jobid', 'zip-code']
    users = pd.read_csv('../dataset/users.dat', sep='::', header=None, names=user_columns, engine='python')
    users = users.drop(columns='zip-code', axis=1)

    # update user's gender to int
    gender_map = {'F': 0, 'M': 1}
    users.replace(gender_map, inplace=True)

    # update user's age range to int
    age_map = {age: i for i, age in enumerate(sorted(set(users['age'])))}
    users.replace(age_map, inplace=True)

    # load movie data
    movie_columns = ['mid', 'title', 'genres']
    movies = pd.read_csv('../dataset/movies.dat', sep='::', header=None, names=movie_columns, engine='python')
    # movies = movies.drop(columns=['title'], axis=1)

    # process genres to int features
    genres_set = set()
    for genres in movies['genres']:
        genres_list = genres.split('|')
        for genre in genres_list:
            genres_set.add(genre)
    genres_map = {g: i for i, g in enumerate(genres_set)}

    num_genres = len(genres_set)
    for i, row in movies.iterrows():
        genres = row['genres']
        genres_list = genres.split('|')
        g_int_list = [0] * num_genres
        for genre in genres_list:
            g_int_list[genres_map[genre]] = 1
        movies['genres'].iat[i] = g_int_list

    # load rating data
    rating_columns = ['uid', 'mid', 'rating', 'timestamp']
    ratings = pd.read_csv('../dataset/ratings.dat', sep='::', header=None, names=rating_columns, engine='python')
    ratings = ratings.drop(columns='timestamp', axis=1)

    frame = pd.merge(pd.merge(users, ratings), movies)
    X = frame.drop(columns='rating', axis=1)
    y = frame['rating']
    return X, y, frame