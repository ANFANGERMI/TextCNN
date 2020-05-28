import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import embeddingdim,filterSizes,class_num,SEQ_LEN,NUM_FILTERS,DROP_OUT, NUM_CHANNELS,K_MAX
from data_generate import pretrained_matrix,vocab_size
class TextCNN(nn.Module):
    """
    Baseline模型：一个浅层的CNN
    具体参数：
    """
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_channel = NUM_CHANNELS
        self.embedding_size = embeddingdim
        self.seq_len = SEQ_LEN
        self.class_num = class_num
        self.num_filters = NUM_FILTERS
        self.num_filters_total = self.num_filters * len(filterSizes)
        # 使用预训练词向量
        self.embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=True).cuda()
        self.embedding_nonstatic = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False).cuda()
        self.dropout = nn.Dropout(DROP_OUT).cuda()
        self.fc_1 = nn.Linear(self.num_filters_total,256)
        self.fc_out = nn.Linear(256, self.class_num, bias=False).cuda()
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.num_channel, self.num_filters, (filter_size, self.embedding_size)).cuda() for filter_size in filterSizes])
        self.k_max = K_MAX
    
    def k_max_pool(self,X,k):
        """
        param X:the feature map
        shape[batch,channel,seq]
        param k:how many feature to leave
        """
        X_pooled = X.topk(k = k,dim = 2)[0]
        X_pooled = torch.reshape(X_pooled,(X.shape[0],-1))
        return X_pooled
          

    def forward(self, X):
        """

        :param X:shape[batch,seq_len,embed_size]
        :return:
        """
        #shape [batchsize,seq_len,channels,embedding_size]
        if self.num_channel == 2:
            X_static = self.embedding(X)
            X_nonstatic = self.embedding_nonstatic(X)
            X = torch.stack((X_static,X_nonstatic),dim=2)
            X = X.permute(0, 2, 1, 3)
        if self.num_channel == 1:
            X = self.embedding(X).unsqueeze(1)
        #匹配一维池化的参数
        X = [F.relu(conv(X)).squeeze(3) for conv in self.convs]
        pooled_output = []
        #max over time pooling每次仅取得featuremap的最大值
        if self.k_max == False:
            pooled_output = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in X]
        #k-max pool
        if self.k_max == True:
            pooled_output = [k_max_pool(item,4) for item in x]
        #shape [batch,channel,seq_len]
        h_pool = torch.cat(pooled_output,1)
        h_pool_flat = torch.reshape(h_pool,[-1, self.num_filters_total])
        h_pool_flat = self.dropout(h_pool_flat)
        fc_1_out = F.relu(self.fc_1(h_pool_flat))
        fc_1_out = self.dropout(fc_1_out)
        model = self.fc_out(fc_1_out)
        return model
    
    

class BiLSTM_CNN(nn.Module):
    """
    Baseline模型：单层LSTM + 一个浅层的CNN
    具体参数：
    """
    def __init__(self):
        super(LSTM_CNN, self).__init__()
        self.num_channel = NUM_CHANNELS
        self.embedding_size = embeddingdim
        self.seq_len = SEQ_LEN
        self.class_num = class_num
        self.num_filters = NUM_FILTERS
        self.num_filters_total = self.num_filters * len(filterSizes)
        # 使用预训练词向量
        self.embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=True).cuda()
        #self.embedding_nonstatic = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False).cuda()
        self.lstm = nn.LSTM(input_size=self.embedding_size,hidden_size=self.embedding_size,bias=True,
                            batch_first=True,bidirectional=True).cuda()
        self.context = nn.Linear(2*self.embedding_size, self.embedding_size).cuda()
        self.dropout = nn.Dropout(DROP_OUT).cuda()
        self.fc = nn.Linear(self.num_filters_total, self.class_num, bias=True).cuda()
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.num_channel, self.num_filters, (filter_size, self.embedding_size)).cuda() for filter_size in filterSizes])

    def forward(self, X):
        """

        :param X:shape[batch,seq_len,embed_size]
        :return:
        """
        #shape [batchsize,seq_len,channels,embedding_size]
        X = self.embedding(X)
        #X = torch.stack((self.embedding(X), self.embedding_nonstatic(X)), dim=2)
        #X = X.permute(0, 2, 1, 3)
        lstm_output = self.lstm(X)[0]
        context_lstm = self.context(lstm_output).unsqueeze(1)
        
        #匹配一维池化的参数
        activation = [F.relu(conv(context_lstm)).squeeze(3) for conv in self.convs]
        #max over time pooling每次仅取得featuremap的最大值
        pooled_output = [F.max_pool1d(item, item.size(2)) for item in activation]
        #shape [batch,channel,seq_len]
        h_pool = torch.cat(pooled_output,1)
        h_pool_flat = torch.reshape(h_pool,[-1, self.num_filters_total])
        h_pool_flat = self.dropout(h_pool_flat)
        model = self.fc(h_pool_flat)
        return model

    
    
    
class LSTM_CNN(nn.Module):
    """
    Baseline模型：单层LSTM + 一个浅层的CNN
    具体参数：
    """
    def __init__(self):
        super(LSTM_CNN, self).__init__()
        self.num_channel = NUM_CHANNELS
        self.embedding_size = embeddingdim
        self.seq_len = SEQ_LEN
        self.class_num = class_num
        self.num_filters = NUM_FILTERS
        self.num_filters_total = self.num_filters * len(filterSizes)
        # 使用预训练词向量
        self.embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=True).cuda()
        #self.embedding_nonstatic = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False).cuda()
        self.lstm = nn.LSTM(input_size=self.embedding_size,hidden_size=self.embedding_size,bias=True,
                            batch_first=True,bidirectional=False).cuda()
        self.context = nn.Linear(self.embedding_size, self.embedding_size).cuda()
        self.dropout = nn.Dropout(DROP_OUT).cuda()
        self.fc = nn.Linear(self.num_filters_total, self.class_num, bias=True).cuda()
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.num_channel, self.num_filters, (filter_size, self.embedding_size)).cuda() for filter_size in filterSizes])

    def forward(self, X):
        """

        :param X:shape[batch,seq_len,embed_size]
        :return:
        """
        #shape [batchsize,seq_len,channels,embedding_size]
        X = self.embedding(X)
        #X = torch.stack((self.embedding(X), self.embedding_nonstatic(X)), dim=2)
        #X = X.permute(0, 2, 1, 3)
        lstm_output = self.lstm(X)[0]
        context_lstm = self.context(lstm_output).unsqueeze(1)
        
        #匹配一维池化的参数
        activation = [F.relu(conv(context_lstm)).squeeze(3) for conv in self.convs]
        #max over time pooling每次仅取得featuremap的最大值
        pooled_output = [F.max_pool1d(item, item.size(2)) for item in activation]
        #shape [batch,channel,seq_len]
        h_pool = torch.cat(pooled_output,1)
        h_pool_flat = torch.reshape(h_pool,[-1, self.num_filters_total])
        h_pool_flat = self.dropout(h_pool_flat)
        model = self.fc(h_pool_flat)
        return model

    
    
if __name__ == "__main__":
    print("该模块定义模型结构")





