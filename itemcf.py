import os
from utils import *

class ItemCF:

    def __init__(self):
        self.movie_info = get_movie_info_dict('dataset/ml-100k/u.item')


    def get_neighbors(self, mid, movie_search_dict, user_search_dict, k):
        """
        get the similarity of neighbors of movie(mid)
        :param uid:
        :param movie_search_dict: dict {mid: (uid, rating)}
        :param user_search_dict: dict {uid: (mid: rating)}
        :param k: k nearest neighbors
        :return: list of neighbors' similarity
        """
        neighbors = []
        for u, _ in movie_search_dict[mid]:
            for m, _ in user_search_dict[u]:
                if m != mid and m not in neighbors:
                    neighbors.append(m)

        msims = []
        for m in neighbors:
            sim = adjusted_cosine_sim(movie_search_dict[mid], movie_search_dict[m])
            msims.append((m, sim))
        msims.sort(key=lambda tup: tup[1], reverse=True)
        if len(msims) < k or k ==0:
            return msims
        else:
            return msims[:k]


    def get_neighbors_dict(self, movie_search_dict, user_search_dict, k):
        """
        calculate all neighbors' similarity of all movies
        :param movie_search_dict: dict {mid: (uid, rating)}
        :param user_search_dict: dict {uid: (mid: rating)}
        :param k: k nearest neighbors
        :return: similarity matrix
        """
        neighbors_dict = {}
        for mid in movie_search_dict:
            neighbors_dict[mid] = self.get_neighbors(mid, movie_search_dict, user_search_dict, k)
        return neighbors_dict


    def get_rcmd_movies(self, uid, msims, movie_search_dict, n):
        """
        get recommended movies
        :param usims: user neighbor's similarity
        :param movie_search_dict: dict {mid: (uid: rating)}
        :param n: n movies recommended
        :return: list of name of recommended movies
        """
        rcmd_dict = {}
        for m, sim in msims:
            for u, r in movie_search_dict[m]:
                if u == uid:
                    if m in rcmd_dict:
                        rcmd_dict[m] += sim * r
                    else:
                        rcmd_dict[m] = sim * r
        rcmd_list = []
        print(rcmd_dict)
        for mid in rcmd_dict:
            rcmd_list.append((mid, rcmd_dict[mid]))
        rcmd_list.sort(key=lambda tup: tup[1], reverse=True)
        rcmd_movies = []
        for i in range(n):
            rcmd_movies.append(self.movie_info[rcmd_list[i][0]])
        return rcmd_movies


    def recommend(self, target_uid, k=10, n=10):
        movie_search_dict, user_search_dict = crete_search_table(get_rating_frame('dataset/ml-100k/u.data'))
        msims = self.get_neighbors(target_uid, movie_search_dict, user_search_dict, k)
        rcmd_movies = self.get_rcmd_movies(target_uid, msims, user_search_dict, n)
        return rcmd_movies


    def predict_rating(self, uid, mid, msims, movie_search_dict):
        rating = 0
        sim_sum = 0
        for m, sim in msims:
            for u, r in movie_search_dict[m]:
                if u == uid:
                    sim_sum += abs(sim)
                    rating += sim * (r - self.movie_avg[mid])
        if sim_sum == 0:
            return (self.movie_avg[mid] + rating)
        else:
            return (self.movie_avg[mid] + rating  / sim_sum)


    def evaluate(self, split_ratio=0.8, k=0, matrix='movie_similarity_matrix.pk'):
        train = get_rating_frame('dataset/ml-100k/u1.base')
        test = get_rating_frame('dataset/ml-100k/u1.test')
        movie_search_dict, user_search_dict = crete_search_table(train)
        self.user_avg = get_user_avg(user_search_dict)
        self.movie_avg = get_movie_avg(movie_search_dict)

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
            if uid not in user_search_dict:
                continue
            if mid not in neighbors_dict:
                predicted_r = self.user_avg[uid]
            else:
                msims = neighbors_dict[mid]
                predicted_r = self.predict_rating(uid, mid, msims, movie_search_dict)
            error_sum += (predicted_r - r) ** 2
            count += 1
        rmse = math.sqrt(error_sum / count)
        return rmse


cf = ItemCF()
rm = cf.recommend(1, 10)
print(rm)