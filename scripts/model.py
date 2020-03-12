import torch
import torch.nn as nn


class SimpleLSTMEncoder(nn.Module):
    def __init__(self,
                 input_dim=64, h_dim=64, linear_dim=1024, num_layers=1, dropout=0.0):
        super(SimpleLSTMEncoder, self).__init__()

        self.linear_dim = linear_dim
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        # LSTM and Fully Connected Layer
        self.encoder = nn.LSTM(
            input_dim, h_dim, num_layers, dropout=dropout)
        self.fc = nn.Linear(2, input_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        '''

        :param obs_traj: Input shape (obs_traj_len, batch, 2)
        :return: shape (num_layers, batch, h_dim)
        '''
        batch_size = obs_traj.size(1)
        obs_traj_embedding = self.fc(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch_size, self.input_dim
        )

        hidden_state_tuple = self.init_hidden(batch_size)
        output, hidden_state = self.encoder(obs_traj_embedding, hidden_state_tuple)
        final_h = hidden_state[0]
        return final_h


class SimpleLSTMDecoder(nn.Module):
    def __init__(self,
                 input_dim=64, h_dim=64, linear_dim=1024, num_layers=1, dropout=0.0):
        super(SimpleLSTMDecoder, self).__init__()

