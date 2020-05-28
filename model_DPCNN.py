import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import class_num,SEQ_LEN,NUM_FILTERS,embeddingdim,NUM_CHANNELS,class_num
from data_generate import pretrained_matrix
torch.manual_seed(1)


class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.channel_size = NUM_FILTERS
        #down sampling
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        ).cuda()
        #conv without increasing feature_map or decreasing demension
        #input shape[batch,channel,length]
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        ).cuda()

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x


class DPCNN(nn.Module):
    """
    DPCNN model, 3
    1. region embedding: using TextCNN to generate
    2. two 3 conv(padding) block
    3. maxpool->3 conv->3 conv with resnet block(padding) feature map: len/2
    """

    # max_features, opt.EMBEDDING_DIM, opt.SENT_LEN, embedding_matrix):
    def __init__(self):
        super(DPCNN, self).__init__()
        self.channel_size = NUM_FILTERS
        self.num_class = class_num
        self.model_name = "DPCNN"
        self.seq_len = SEQ_LEN
        self.init_channel = NUM_CHANNELS
        self.embedding_size = embeddingdim
        self.filter_size = 3
        self.embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=True).cuda()
        self.embedding_nonstatic = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False).cuda()
        self.shortcut = nn.Linear(self.channel_size,self.channel_size,bias=True).cuda()
        # region embedding
        self.region_embedding = nn.Sequential(nn.Conv2d(self.init_channel, self.channel_size,
                      kernel_size=(self.filter_size,self.embedding_size),padding=(1,0)),
                      nn.BatchNorm2d(num_features=self.channel_size),
                      nn.ReLU(),
                      nn.Dropout(0.2)).cuda()
        #two conv layer with pre-activation
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size,
                      kernel_size=(self.filter_size,1),
                      padding=(1,0)),
            nn.BatchNorm2d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size,
                      kernel_size=(self.filter_size,1),
                      padding=(1,0)),
        ).cuda()

        resnet_block_list = []
        while (self.seq_len > 2):
            resnet_block_list.append(ResnetBlock())
            self.seq_len = self.seq_len // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list).cuda()
        self.fc = nn.Sequential(
            nn.Linear(self.channel_size*self.seq_len, self.num_class),
            nn.BatchNorm1d(self.num_class),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.num_class, self.num_class)
        ).cuda()

    def forward(self, x):
        if self.init_channel == 2:
            x_static = self.embedding(x)
            x_nonstatic = self.embedding_nonstatic(x)
            x = torch.stack((x_static,x_nonstatic),dim=2)
            x = x.permute(0, 2, 1, 3)
        if self.init_channel == 1:
            X = self.embedding(X).unsqueeze(1)
        #pre_act for short cut
        #shape [batch,length,channel_size]
        pre_act = F.relu(self.region_embedding(x)).squeeze(3).permute(0,2,1)
        x = self.region_embedding(x)
        x = self.conv_block(x).squeeze(3).permute(0,2,1)
        #merge into mainstreet
        #tranfer the dimension back
        #shape[batch,channel_size,seq_len_1,1]
        x = (self.shortcut(pre_act) + x).permute(0,2,1)
        #to the resblock part
        x = self.resnet_layer(x)
        #x = x.permute(0, 2, 1)
        x = x.contiguous().view(x.shape[0], -1)
        out = self.fc(x)
        return out