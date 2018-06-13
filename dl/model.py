import torch
import torch.nn as nn
import torch.nn.functional as F

class MovieRsNN(nn.Module):

    def __init__(self):
        super(MovieRsNN, self).__init__()
        self.fc_u = nn.Linear(32, 32)
        self.fc_a = nn.Linear(16, 32)
        self.fc_g = nn.Linear(16, 32)
        self.fc_o = nn.Linear(16, 32)
        self.fc_m = nn.Linear(32, 32)
        self.fc_ge = nn.Linear(32, 32)

        self.fc_u2 = nn.Linear(128, 200)
        self.fc_m2 = nn.Linear(64, 200)

        self.fc_final = nn.Linear(400, 1)


    def forward(self, u, a, g, o, m, ge):
        x_u = F.relu(self.fc_u(u))
        x_a = F.relu(self.fc_a(a))
        x_g = F.relu(self.fc_g(g))
        x_o = F.relu(self.fc_o(o))
        x_m = F.relu(self.fc_m(m))
        x_ge = F.relu(self.fc_ge(ge))

        x_user = torch.cat((x_u, x_a), 1)
        x_user = torch.cat((x_user, x_g), 1)
        x_user = torch.cat((x_user, x_o), 1)
        x_movie = torch.cat((x_m, x_ge), 1)

        x_user = F.tanh(self.fc_u2(x_user))
        x_movie = F.tanh(self.fc_m2(x_movie))

        output = self.fc_final(torch.cat((x_user, x_movie), 1))
        return output[:,0]


