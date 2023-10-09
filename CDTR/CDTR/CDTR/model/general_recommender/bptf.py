
import torch
import torch.nn as nn

from CDTR.model.init import xavier_normal_initialization
from CDTR.model.loss import BPRLoss
from CDTR.utils import InputType
from CDTR.model.abstract_recommender import GeneralRecommender


class BPTF(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(BPTF, self).__init__(config, dataset)

        self.task = config['task']
        self.LABEL = config['LABEL_FIELD']
        self.RATING = config['RATING_FIELD']
        self.TIME = config['TIME_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_times=config['K']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.time_embedding = nn.Embedding(self.n_times, self.embedding_size)
        if self.task == 'ps':
            self.loss = nn.BCELoss(reduce='mean')
            self.sigmoid = nn.Sigmoid()
        else:
            self.loss = nn.MSELoss(reduce=False)
            self.sigmoid = None
      

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

    def get_time_embedding(self, time):
       
        return self.time_embedding(time)

    def forward(self, user, item,day):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        time_e = self.get_time_embedding(day.long())
        return torch.mul(user_e, item_e+time_e).sum(dim=1)


    def calculate_loss(self, interaction, weight=None):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        day=interaction[self.TIME].long()
        if self.task=='ps':
            label = interaction[self.LABEL]
        else:
            label = interaction[self.RATING]
        output = self.forward(user, item,day)
        loss = self.loss(output, label)
        if weight != None:
            loss *= weight
        if self.task!='ps':
            loss = torch.sum(loss) / len(loss)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        day = interaction[self.TIME].long()
        score = self.forward(user, item,day)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
