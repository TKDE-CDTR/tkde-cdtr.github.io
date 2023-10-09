
import math

import torch
import torch.nn as nn

from CDTR.model.abstract_recommender import GeneralRecommender
from CDTR.model.init import xavier_normal_initialization
from CDTR.model.loss import BPRLoss
from CDTR.utils import InputType
from CDTR.model.layers import MLPLayers

class TimeProp(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(TimeProp, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.TIME = config['TIME_FIELD']
        self.K = config['K']

        # self.Ra
        self.RATING = config['RATING_FIELD']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.mlp=MLPLayers([self.embedding_size*2,self.embedding_size,self.embedding_size,7],activation='relu')
        self.softmax = nn.Softmax(dim=1)
        self.loss=nn.CrossEntropyLoss()
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

    def smooth(self, t, b):
        s = 1 + torch.exp(self.T * (t - b))
        s = 1 / s
        return s

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        input=torch.cat([user_e,item_e],dim=1)
        output=self.mlp(input)
        return output


    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        time=time.long()
        output=self.forward(user,item)
        loss=self.loss(output,time.long())
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        time = interaction[self.TIME]
        time = time.long()
        output = self.softmax(self.forward(user, item))

        #p=torch.argmax(output,dim=1)
        return output[range(len(user)), time]

    def get_p(self,user,item,time):
        time = time.long()
        output = self.softmax(self.forward(user, item))
        return output[range(len(user)),time]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        time = interaction[self.TIME]
        # time_origin = interaction[self.TIME_ORIGIN]
        puit = self.forward(user, all_item_e, time.unsqueeze(1))

        return puit
