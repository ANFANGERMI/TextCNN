from torchtext.data import TabularDataset
from torchtext.data import Field
from torchtext import vocab
from torchtext.data import BucketIterator, Iterator
import jieba
import re
import logging
from torch import device
from settings import BATCH_SIZE, SEQ_LEN

jieba.setLogLevel(logging.INFO)
#加载停用词表
stop_words = []
with open('./中文停用词表.txt', mode='r', encoding='utf-8') as f:
    for text in f:
        stop_words.append(text.strip('\n'))

#定义给torchtext使用的分词器
def tokenize(text):
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9]')
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]

#定义torchtext使用的字段对象
TEXT = Field(sequential=True, use_vocab=True,tokenize=tokenize,stop_words=stop_words, fix_length=SEQ_LEN)
LABEL = Field(sequential=False, use_vocab=False)

#读取文件
tv_datafields = [('id', None), ('content', TEXT), ('label', LABEL)]
tst_datafields = [('id', None), ('content', TEXT), ('label', LABEL)]
params = {'delimiter': '\t'}
trn = TabularDataset(path='./data/train_alloversample.csv', format='CSV', fields=tv_datafields, skip_header=True)
tst = TabularDataset(path='../data/test.csv', format='CSV', fields=tst_datafields, skip_header=False,
                    csv_reader_params=params)

#划分验证集
trn, vld = trn.split(split_ratio=0.9, stratified=True)


#产生供训练使用的迭代器
train_iter, val_iter = BucketIterator.splits(
 (trn, vld), # we pass in the datasets we want the iterator to draw data from
 batch_sizes=(BATCH_SIZE, BATCH_SIZE),
 device=device('cuda'), # if you want to use the GPU, specify the GPU number here
 sort_key=lambda x: len(x.content), # the BucketIterator needs to be told what function it should use to group the data.
 sort_within_batch=False,
 repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)
test_iter = Iterator(tst, batch_size=64, device=0, sort=False, sort_within_batch=False, repeat=False)

#产生预训练的词向量矩阵
vectors = vocab.Vectors(name='wv_f.word', cache='./wordvec')
TEXT.build_vocab(trn, vld, vectors=vectors)
pretrained_matrix = TEXT.vocab.vectors
vocab_size = len(TEXT.vocab)


if __name__ == "__main__":
    print("该文件产生迭代器，以及预训练矩阵，大小为：", pretrained_matrix.shape)
  