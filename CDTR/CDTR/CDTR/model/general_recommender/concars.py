from audioop import bias
from genericpath import exists
import torch
import torch.nn as nn
import numpy as np
from CDTR.model.init import xavier_normal_initialization
from CDTR.model.loss import BPRLoss
from CDTR.utils import InputType
from CDTR.model.abstract_recommender import GeneralRecommender
from CDTR.model.layers import MLPLayers, CNNLayers


class CoNCARS(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CoNCARS, self).__init__(config, dataset)

        self.task = config['task']
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        self.TIME = config['TIME_FIELD']
        self.cnn_channels = config['cnn_channels']
        self.cnn_kernels = config['cnn_kernels']
        self.cnn_strides = config['cnn_strides']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_times = config['K']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.u_t = []
        self.i_t = []
        for i in range(self.n_times):
            self.u_t.append(nn.Embedding(self.n_users, self.embedding_size).to(self.device))
            self.i_t.append(nn.Embedding(self.n_items, self.embedding_size).to(self.device))
     
        self.wcu = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size, bias=True)
        self.wci = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.cnn_layers_uy = CNNLayers(self.cnn_channels, self.cnn_kernels, self.cnn_strides, activation='relu',
                                       init_method='norm')
        self.cnn_layers_ui = CNNLayers(self.cnn_channels, self.cnn_kernels, self.cnn_strides, activation='relu',
                                       init_method='norm')
        self.cnn_layers_xy = CNNLayers(self.cnn_channels, self.cnn_kernels, self.cnn_strides, activation='relu',
                                       init_method='norm')
        self.cnn_layers_xi = CNNLayers(self.cnn_channels, self.cnn_kernels, self.cnn_strides, activation='relu',
                                       init_method='norm')
        self.predict_layers = nn.Linear(self.embedding_size * 2, 1)
        # self.predict_layers = MLPLayers([self.cnn_channels[-1], 1], self.dropout_prob, activation='none')
        if self.task == 'ps':
            self.loss = nn.BCELoss(reduce='mean')
            self.sigmoid = nn.Sigmoid()
        else:
            self.loss = nn.MSELoss(reduce=False)
            self.sigmoid = None

        # parameters initialization
        # self.init_weights()
        self.apply(xavier_normal_initialization)

    def init_weights(self):
        initrange = 1e-9
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.item_embedding.weight.data.uniform_(-initrange, initrange)
        for i in range(self.n_times):
            self.u_t[i].weight.data.uniform_(-initrange, initrange)
            self.i_t[i].weight.data.uniform_(-initrange, initrange)
      

    def get_user_embedding(self, user):
        # idx = user * self.n_times + time
        return self.user_embedding(user)

    def get_item_embedding(self, item):

        return self.item_embedding(item)

    def get_time_embedding(self, time):

        return self.time_embedding(time)

    def forward(self, user, item, time):
        time = time.long()
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        utl = [self.u_t[i](user) for i in range(self.n_times)]
        itl = [self.i_t[i](item) for i in range(self.n_times)]
        ua = torch.stack(utl)
        ia = torch.stack(itl)
        atte_u = torch.empty(len(user), self.n_times).to(self.device)
        atte_i = torch.empty(len(user), self.n_times).to(self.device)
        for i in range(self.n_times):
            ut = self.u_t[i](user)
            hu = self.wcu(ut)
            hu = self.tanh(hu)
            atte_u[range(len(user)), i] = torch.sum(torch.mul(hu, ua[time]))
            it = self.i_t[i](item)
            hi = self.wcu(it)
            hi = self.tanh(hi)
            atte_i[range(len(user)), i] = torch.sum(torch.mul(hi, ia[time]))
        atte_u = self.softmax(atte_u)
        atte_i = self.softmax(atte_i)
        xu = torch.zeros(len(user), self.embedding_size).to(self.device)
        yi = torch.zeros(len(user), self.embedding_size).to(self.device)
        for i in range(self.n_times):
            xu += atte_u[:, i].view(-1, 1) * self.u_t[i](user)
            yi += atte_i[:, i].view(-1, 1) * self.i_t[i](item)
        uy = torch.bmm(user_e.unsqueeze(2), yi.unsqueeze(1))
        uy = uy.unsqueeze(1)
        cnn_output = self.cnn_layers_uy(uy)
        cnn_output_uy = cnn_output.sum(axis=(2, 3))
        ui = torch.bmm(user_e.unsqueeze(2), item_e.unsqueeze(1))
        ui = ui.unsqueeze(1)
        cnn_output = self.cnn_layers_ui(ui)
        cnn_output_ui = cnn_output.sum(axis=(2, 3))
        xy = torch.bmm(xu.unsqueeze(2), yi.unsqueeze(1))
        xy = xy.unsqueeze(1)
        cnn_output = self.cnn_layers_xy(xy)
        cnn_output_xy = cnn_output.sum(axis=(2, 3))
        xi = torch.bmm(xu.unsqueeze(2), item_e.unsqueeze(1))
        xi = xi.unsqueeze(1)
        cnn_output = self.cnn_layers_uy(xi)
        cnn_output_xi = cnn_output.sum(axis=(2, 3))
        z = torch.cat([cnn_output_uy, cnn_output_ui, cnn_output_xy, cnn_output_xi], dim=1)
        output = self.predict_layers(z)

        return output.squeeze()

    def calculate_loss(self, interaction, weight=None):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        if self.task == 'ps':
            label = interaction[self.LABEL]
        else:
            label = interaction[self.RATING]

        output = self.forward(user, item, time)
        loss = self.loss(output, label)
        if weight != None:
            loss *= weight
        if self.task != 'ps':
            loss = torch.sum(loss) / len(loss)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        score = self.forward(user, item, time)
        return score

    def get_p(self, user, item, time):
        output = self.forward(user, item, time)
        return output

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
