from model_define import TextCNN,LSTM_CNN
from model_DPCNN import DPCNN
from data_generate import test_iter
from sklearn.metrics import f1_score,roc_auc_score,precision_score,recall_score
import torch
device = torch.device('cuda')
model = LSTM_CNN().to(device)
#保存的参数路径
model.load_state_dict(torch.load('./checkpoint_BiLSTM/TextCNN_Params_87.0891.pth'))

def predict(model,test_set):
    model.eval()
    total_corrects = 0
    total_num = 0
    ground_truth = torch.rand(0).long().to(device)
    prediction = torch.rand(0).long().to(device)
    #prob = torch.rand(0).to(device)
    for batch in test_set:
        input_batch = batch.content.t().to(device)
        input_label = batch.label.to(device)
        output = model(input_batch)
        pred = (torch.max(output,1))[1]
        total_corrects += (pred == input_label).sum()
        ground_truth = torch.cat((ground_truth,input_label),dim=0)
        prediction = torch.cat((prediction,pred),dim=0)
        #prob = torch.cat((prob,output),dim=1)
        total_num += batch.batch_size
    
    total_Acc = 100.0 * total_corrects / total_num
    labels = [0,1,2,3,4,5,6]
    f1_macro = f1_score(y_true = ground_truth.cpu(),y_pred = prediction.cpu(),average = 'macro')
    f1_micro = f1_score(y_true = ground_truth.cpu(),y_pred = prediction.cpu(),average = 'micro')
    recall = recall_score(y_true = ground_truth.cpu(),y_pred = prediction.cpu(),average = 'macro')
    precision = precision_score(y_true = ground_truth.cpu(),y_pred = prediction.cpu(),average = 'macro')
    #auc_score = roc_auc_score(y_true = ground_truth,y_score = prob,multi_class = 'ovr',labels = labels)
    print("The Micro F1-score:%.3f \nThe Macro F1-score:%.3f" % (f1_micro,f1_macro))
    print("The recall Score:%.3f\nThe precision:%.3f" % (recall, precision))
    return total_Acc,total_num,total_corrects

if __name__ == '__main__':
    print("用于模型测试")
    acc, num, corrects = predict(model,test_iter)
    print("测试集上准确率:%.4f 测试用例:%d  正确用例:%d" % (acc, num, corrects))