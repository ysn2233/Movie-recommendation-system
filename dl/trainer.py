import math
import torch
import torch.nn as nn
import torch.optim as optim
from dl.model import MovieRsNN
from dl.config import *
from dl.preprocessing import data_preprocess_100k


def get_embedding_matrix(frame):
    uid = torch.from_numpy(frame['uid'].as_matrix())
    age = torch.from_numpy(frame['age'].as_matrix())
    gender = torch.from_numpy(frame['gender'].as_matrix())
    occupation = torch.from_numpy(frame['occupation'].as_matrix())
    u, a, g, o = get_user_embedding(uid, age, gender, occupation)

    mid = torch.from_numpy(frame['mid'].as_matrix())
    genres = torch.from_numpy(frame.drop(columns=['uid', 'age', 'gender', 'occupation', 'mid']).as_matrix())
    m, ge = get_movie_embedding(mid, genres)
    if torch.cuda.is_available():
        u = u.cuda()
        a = a.cuda()
        g = g.cuda()
        o = o.cuda()
        m = m.cuda()
        ge = ge.cuda()
    return u, a, g, o, m, ge


def get_user_embedding(uid, age, gender, occupation):
    uid_embed_layer = nn.Embedding(UID_100K_NUM + 1, 32)
    age_embed_layer = nn.Embedding(AGE_100K_GROUP_NUM + 1, 16)
    gender_embed_layer = nn.Embedding(2, 16)
    occupation_embed_layer = nn.Embedding(OCCUPATION_100K_NUM + 1, 16)
    uid_embed_matrix = uid_embed_layer(uid).float()
    age_embed_matrix = age_embed_layer(age).float()
    gender_embed_matrix = gender_embed_layer(gender).float()
    occupation_embed_matrix = occupation_embed_layer(occupation).float()
    return uid_embed_matrix, age_embed_matrix, gender_embed_matrix, occupation_embed_matrix


def get_movie_embedding(mid, genres):
    mid_embed_layer = nn.Embedding(MOVIE_100K_NUM + 1, 32)
    mid_embed_matrix = mid_embed_layer(mid)
    genres_embed_layer = nn.Embedding(GENRES_100K_NUM + 1, 32).float()
    genres_embed_matrix = genres_embed_layer(genres)
    genres_embed_matrix = genres_embed_matrix.sum(dim=1).float()
    return mid_embed_matrix, genres_embed_matrix


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_preprocess_100k('../dataset/ml-100k')
    use_gpu = torch.cuda.is_available()
    net = MovieRsNN()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    for epoch in range(50):
        running_loss = 0.0
        i = 0
        while i < X_train.shape[0]:
            optimizer.zero_grad()
            batch_end = i + BATCH_SIZE
            if batch_end >= X_train.shape[0]:
                batch_end = X_train.shape[0]
            u, a, g, o, m, ge = get_embedding_matrix(X_train[i: batch_end])
            y = y_train[i: batch_end]
            y = torch.from_numpy(y.as_matrix()).float()
            if use_gpu:
                y = y.cuda()
            out = net.forward(u, a, g, o, m, ge)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            #print('[%d], loss is %f' % (epoch, loss.data[0]))
            running_loss += loss.item()
            i = batch_end
        print('epoch [%d] finished, the average loss is %f' % (epoch, running_loss))

    loss_sum = 0.0
    i = 0
    u, a, g, o, m, ge = get_embedding_matrix(X_test)
    y = torch.from_numpy(y_test.as_matrix()).float()
    if use_gpu:
        y = y.cuda()
    for i in range(X_test.shape[0]):
        out = net.forward(u[i].unsqueeze(0), a[i].unsqueeze(0), g[i].unsqueeze(0), o[i].unsqueeze(0), m[i].unsqueeze(0), ge[i].unsqueeze(0))
        loss_sum += (out - y[i]) ** 2
    rmse = math.sqrt(loss_sum / X_test.shape[0])
    print(rmse)