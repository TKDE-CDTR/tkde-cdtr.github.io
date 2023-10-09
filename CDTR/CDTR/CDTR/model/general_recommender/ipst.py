import torch
import torch.nn as nn

from CDTR.model.init import xavier_normal_initialization
from CDTR.model.layers import MLPLayers
from CDTR.model.loss import BPRLoss
from CDTR.utils import InputType
from CDTR.model.abstract_recommender import GeneralRecommender


class IPST(nn.Module):

    def __init__(self, config,psmodel,n_users,n_items):
        super(IPST, self).__init__()

        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.n_users=n_users
        self.n_items=n_items
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = nn.MSELoss()
        self.gamma= config['gamma_t']
        self.psmodel=psmodel
        self.mlp=MLPLayers([self.embedding_size*2+1,self.embedding_size,1],activation='sigmoid')
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user,item,time):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        po = self.psmodel.get_p(user, item, time)
        po[po<0.25]=0.25
        invp = torch.reciprocal(po)
        time=time.unsqueeze(dim=-1)
        input=torch.cat([user_e,item_e,time],dim=1)
        w=self.mlp(input).squeeze()

        low=1+(invp-1)/self.gamma

        up=1+(invp-1)*self.gamma
        w.data*=(up-low)
        w.data += low
        return w

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.sigmoid(self.forward(user, item))
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.sigmoid(self.forward(user, item))
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
