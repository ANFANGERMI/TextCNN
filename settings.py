"""
模型参数设置
"""
embeddingdim = 300#词向量维度
filterSizes = [3,4,5]
class_num = 7#输出标签种类
NUM_CHANNELS = 1
SEQ_LEN = 256
EPOCHS = 3000
FILE_PATH = r"./wv.txt"
MODEL_SAVE_PATH = r"./checkpoint_BiLSTM"
BATCH_SIZE = 128
NUM_FILTERS =256
DROP_OUT = 0.85
L2_Norm = 3.0
K_MAX = False


if __name__ == "__main__":
    print("这是一个参数文档")
