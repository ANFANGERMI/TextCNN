import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from model_define import LSTM_CNN,TextCNN
from model_DPCNN import DPCNN
from data_generate import train_iter, test_iter, val_iter
from settings import EPOCHS,BATCH_SIZE,MODEL_SAVE_PATH,L2_Norm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def confusion_matrices(truth,pred,label):
    matrix = 0
    acc = {}
    batch_size = len(pred)
    for i in range(batch_size):
        matrix += confusion_matrix(truth[i],pred[i],labels=[i for i in range(7)])
    print("confusion matrix:\n")
    for i in matrix:
        print(i)
    for i in range(label):
        if matrix[i].sum() != 0:
            acc['label_' + str(i)] = matrix[i][i]/(matrix[i].sum())
        else:
            acc['label_' + str(i)] = "label not included"
    print("categorical acc:\n")
    for key, value in acc.items():
        if type(value) == str:
             print('%s: %s'% (key,value))
        else:
            print('%s: %.4f'% (key,value))
def train(model):
    """

    :param model:TextCNN模型
    :return:
    """
    model.to(device)
    #model.train()
    #optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=L2_Norm)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9,weight_decay=1e-5)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    steps = 0
    bestAcc = 0
    bestAcc_step = 0

    for epoch in range(1, EPOCHS + 1):
        print("Epoch:%d" % epoch)
        for batch in train_iter:
            #原本是[seq_len,batch] -> [batch,seq_len]
            model.train()
            input_batch = batch.content.t().to(device)
            input_label = batch.label.to(device)
            optimizer.zero_grad()
            output = model(input_batch)
            loss = F.cross_entropy(output, input_label)
            loss.backward()
            optimizer.step()
            steps+=1

            #每50步输出一次训练集的情况
            if steps % 50 == 0:
                corrects = ((torch.max(output, 1))[1] == input_label).sum()
                train_acc = 100.0 * corrects / BATCH_SIZE
                print('\rBatch[{}] Training Set: - loss: {:.6f} - acc: {:.4f}%=({}/{})'.format(steps,
                                                                                             loss.item(),
                                                                                             train_acc,
                                                                                             corrects,
                                                                                             BATCH_SIZE))

            #每500步在验证集上进行一次验证
            if steps % 200 == 0:
                valid_acc = eval(model)
                if valid_acc > bestAcc:
                    bestAcc = valid_acc
                    bestAcc_step = steps
                    print('Best model acc: {:.4f}%\n'.format(bestAcc))
                    if valid_acc > 80.0:
                        save(model, 'TextCNN_Baseline', bestAcc)


def save(model,model_name,acc):
    """

    :param model:model to save
    :param acc: the accuracy
    :return:
    """
    """  
    save_name = '{}_{}_{:.4f}%.pt.pkl'.format(model_name, time.strftime("%Y-%m-%d %H:%M", time.localtime()), acc)
    """
    save_name = "TextCNN_Params_{:.4f}.pth".format(acc)
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, save_name))



def eval(model,valid_set=val_iter):
    """

    :param valid_set:测试集
    :param model:模型
    :return:
    acc
    """
    model.eval()
    corrects, avg_loss = 0,0
    valid_size = 0
    pred_label = []
    truth_label = []
    for batch in valid_set:
        valid_batch = batch.content.t().to(device)
        val_label = batch.label.to(device)
        output = model(valid_batch)
        loss = F.cross_entropy(output,val_label)
        avg_loss += loss.item()
        corrects += (torch.max(output, 1)[1] == val_label).sum()
        valid_size += batch.batch_size
        pred_label.append(torch.max(output,1)[1].cpu())
        truth_label.append(val_label.cpu())

    #valid_size = len(valid_set)
    avg_loss /= valid_size
    acc = 100.0 * corrects / valid_size
    confusion_matrices(truth_label,pred_label,7)
    print('\n         Evaluation Set - loss: {:.6f} - acc: {:.4f}% = ({}/{}) \n'.format(avg_loss,
                                                                                             acc,
                                                                                        corrects,
                                                                                        valid_size))
    return acc

if __name__ == '__main__':
    #model = DPCNN().to(device)
    #model = TextCNN().to(device)
    model = LSTM_CNN().to(device)
    print("开始训练")
    train(model)