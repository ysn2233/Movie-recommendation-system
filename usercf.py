import os
from utils import *


class UserCF:

    def __init__(self):
        self.movie_info = get_movie_info_dict('dataset/ml-100k/u.item')


    def get_neighbors(self, uid, movie_search_dict, user_search_dict, k):
        """
        get the similarity of neighbors of user(uid)
        :param uid:
        :param movie_search_dict: dict {mid: (uid, rating)}
        :param user_search_dict: dict {uid: (mid: rating)}
        :param k: k nearest neighbors
        :return: list of neighbors' similarity
        """
        neighbors = []
        for m, _ in user_search_dict[uid]:
            for u, _ in movie_search_dict[m]:
                if u != uid and u not in neighbors:
                    neighbors.append(u)

        usims = []
        for u in neighbors:
            sim = pearson_sim(user_search_dict[uid], user_search_dict[u])
            usims.append((u, sim))
        usims.sort(key=lambda tup: tup[1], reverse=True)
        if len(usims) < k or k == 0:
            return usims
        else:
            return usims[:k]


    def get_neighbors_dict(self, movie_search_dict, user_search_dict, k):
        """
        calculate all neighbors' similarity of all users
        :param movie_search_dict: dict {mid: (uid, rating)}
        :param user_search_dict: dict {uid: (mid: rating)}
        :param k: k nearest neighbors
        :return: similarity matrix
        """
        neighbors_dict = {}
        for uid in user_search_dict:
            neighbors_dict[uid] = self.get_neighbors(uid, movie_search_dict, user_search_dict, k)
        return neighbors_dict


    def get_rcmd_movies(self, usims, user_search_dict, n):
        """
        get recommended movies
        :param usims: user neighbor's similarity
        :param user_search_dict: dict {uid: (mid: rating)}
        :param n: n movies recommended
        :return: list of name of recommended movies
        """
        rcmd_dict = {}
        for u, sim in usims:
            for m, r in user_search_dict[u]:
                if m not in rcmd_dict:
                    rcmd_dict[m] = sim * r
                else:
                    rcmd_dict[m] += sim * r
        rcmd_list = []
        for mid in rcmd_dict:
            rcmd_list.append((mid, rcmd_dict[mid]))
        rcmd_list.sort(key=lambda tup: tup[1], reverse=True)
        rcmd_movies = []
        for i in range(n):
            rcmd_movies.append(self.movie_info[rcmd_list[i][0]])
        return rcmd_movies


    def recommend(self, target_uid, k=10, n=10):
        movie_search_dict, user_search_dict = crete_search_table(get_rating_frame('dataset/ml-100k/u.data'))
        usims = self.get_neighbors(target_uid, movie_search_dict, user_search_dict, k)
        rcmd_movies = self.get_rcmd_movies(usims, user_search_dict, n)
        return rcmd_movies


    def predict_rating(self, uid, mid, usims, user_search_dict):
        """
        predict the rating for specific uid and mid
        :param uid:
        :param mid:
        :param usims: user neighbor's similarity
        :param user_search_dict: dict {uid: (mid: rating)}
        :return: predicted rating
        """
        rating = 0
        sim_sum = 0
        for u, sim in usims:
            for m, r in user_search_dict[u]:
                if m == mid:
                    sim_sum += abs(sim)
                    rating += sim * (r - self.user_avg[uid])
        if sim_sum == 0:
            return (self.user_avg[uid] + rating)
        else:
            return (self.user_avg[uid] + rating / sim_sum)


    def evaluate(self, split_ratio=0.8, k=0, matrix='user_similarity_matrix.pk'):
        train = get_rating_frame('dataset/ml-100k/u1.base')
        test = get_rating_frame('dataset/ml-100k/u1.test')
        movie_search_dict, user_search_dict = crete_search_table(train)
        self.movie_avg = get_movie_avg(movie_search_dict)
        self.user_avg = get_user_avg(user_search_dict)
        if not os.path.exists(matrix):
            neighbors_dict = self.get_neighbors_dict(movie_search_dict, user_search_dict, k)
            serialize_dict(neighbors_dict, matrix)
        else:
            neighbors_dict = deserialize_dict(matrix)

        error_sum = 0
        count = 0
        for _, row in test.iterrows():
            uid = row['uid']
            mid = row['mid']
            r = row['rating']
            if mid not in movie_search_dict:
                continue
            if uid not in user_search_dict:
                continue
            else:
                usims = neighbors_dict[uid]
                predicted_r = self.predict_rating(uid, mid, usims, user_search_dict)
            error_sum += (predicted_r - r) ** 2
            count += 1
        rmse = math.sqrt(error_sum / test.shape[0])
        return rmse


cf = UserCF()
rm = cf.recommend(1, 10)
print(rm)
#rmse = cf.evaluate()
#print(rmse)