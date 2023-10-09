
import torch
import torch.nn as nn

from CDTR.model.init import xavier_normal_initialization
from CDTR.model.loss import BPRLoss
from CDTR.utils import InputType
from CDTR.model.abstract_recommender import GeneralRecommender


class TMF(GeneralRecommender):
    r"""
        MF model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(TMF, self).__init__(config, dataset)

        self.task = config['task']
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        self.TIME = config['TIME_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_times = config['K']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.time_embedding = nn.Embedding(self.n_times, 1)
        self.user_Dyn_embedding = nn.Embedding(self.n_users * self.n_times, self.embedding_size)

        self.b_u = nn.Embedding(self.n_users, 1)
        self.b_i = nn.Embedding(self.n_items, 1)
        tt=torch.zeros((1,))
        self.b = nn.Parameter(tt)
        if self.task == 'ps':
            self.loss = nn.BCELoss(reduce='mean')
            self.sigmoid = nn.Sigmoid()
        else:
            self.loss = nn.MSELoss(reduce=False)
            self.sigmoid = None

        # parameters initialization
        self.init_weights()
        #self.apply(xavier_normal_initialization)

    def init_weights(self):
        initrange = 0.1
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.item_embedding.weight.data.uniform_(-initrange, initrange)
        self.time_embedding.weight.data.uniform_(-initrange, initrange)
    def get_user_embedding(self, user,time):
        idx = user * self.n_times + time
        return self.user_Dyn_embedding(idx)

    def get_item_embedding(self, item):

        return self.item_embedding(item)

    def get_time_embedding(self, time):

        return self.time_embedding(time)

    def forward(self, user, item, time):
        user_e = self.get_user_embedding(user,time.long())
        item_e = self.get_item_embedding(item)
        time_e = self.get_time_embedding(time.long())
        output=torch.mul(user_e, item_e).sum(dim=1)
        output+=time_e.squeeze()+self.b+self.b_u(user).squeeze()+self.b_i(item).squeeze()
        if self.sigmoid==None:
            return output
        output=self.sigmoid(output)
        return output

    def calculate_loss(self, interaction, weight=None):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        if self.task=='ps':
            label = interaction[self.LABEL]
        else:
            label = interaction[self.RATING]

        output = self.forward(user, item, time)
        loss = self.loss(output, label)
        if weight != None:
            loss *= weight
        if self.task!='ps':
            loss = torch.sum(loss) / len(loss)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        score = self.forward(user, item, time)
        return score

    def get_p(self,user,item,time):
        output = self.forward(user,item,time)
        return output


    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
