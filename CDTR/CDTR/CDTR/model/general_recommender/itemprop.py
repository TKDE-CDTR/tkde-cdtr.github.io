
import torch
import torch.nn as nn

from CDTR.model.init import xavier_normal_initialization
from CDTR.model.loss import BPRLoss
from CDTR.utils import InputType
from CDTR.model.abstract_recommender import GeneralRecommender


class ItemProp(GeneralRecommender):
    r"""
        ItemProp model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(ItemProp, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # self.loss = nn.MSELoss()
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.T=100
        # parameters initialization
        self.apply(xavier_normal_initialization)

        #self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embedding.weight.data.uniform_(-initrange, initrange)
        self.item_embedding.weight.data.uniform_(-initrange, initrange)
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

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        all_item_e = self.item_embedding.weight
        all_score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        all_score=all_score/self.T
        all_score=torch.softmax(all_score,dim=1)
        score=all_score.gather(1, item.unsqueeze(dim=1)).squeeze()
        return score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]
        output = self.forward(user, item)
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        score = self.forward(user, item)
        return score
    def get_p(self,user,item):
        output = self.forward(user,item)
        return output


    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
