# -*-coding:utf-8-*-
import torch
import numpy as np
import torch.utils.data as Data

from torch import nonzero

def find_one_zero(RS):01
    one_postive = np.argwhere(RS == 1)
    one_postive = np.array(one_postive)
    np.random.shuffle(one_postive)

    zero_postive = np.argwhere(RS == 0)
    zero_postive = np.array(zero_postive)
    np.random.shuffle(zero_postive)  

    return one_postive, zero_postive


def index_dp(A, k):01
    index_drug_p = np.transpose(np.nonzero(A))
    np.random.shuffle(index_drug_p)
    data_1 = np.array_split(index_drug_p, k, 0)

    index_dp_zero = np.argwhere(A == 0)
    np.random.shuffle(index_dp_zero)
    data_0 = np.array_split(index_dp_zero, k, 0)

    return data_0, data_1



def Split_data(data_1, data_0, fold, k, drug_p):01
    X_train = []  
    Y_train = []
    X_test = []
    Y_test = []
    X_t = []
    for i in range(k):  
        num = 0
        if i != fold:  
            for j in range(len(data_1[i])):
                X_train.append(data_1[i][j])
            for t in range(len(data_0[i])):
                if t < len(data_1[i]):
                    X_train.append(data_0[i][t])
                else:
                    x = int(data_0[i][t][0])
                    y = int(data_0[i][t][1])
                    X_test.append([x, y])

        else:
            for t1 in range(len(data_1[i])):
                x = int(data_1[i][t1][0])  
                y = int(data_1[i][t1][1])  
                X_test.append([x, y])
                if t1<2500:
                    X_t.append([x,y])
                    
            for t2 in range(len(data_0[i])):  
                x = int(data_0[i][t2][0])  
                y = int(data_0[i][t2][1])
                X_test.append([x, y])
                if num<2500:
                    X_t.append([x, y])
                    num = num + 1
    np.random.shuffle(X_train)
    np.random.shuffle(X_t)

    return X_train, X_test, X_t

def load_data(id, BATCH_SIZE, RS):01
    x = []
    y = []
    for j in range(id.shape[0]):  
        temp_save = []
        x_A = int(id[j][0])  
        y_A = int(id[j][1])  
        temp_save.append([x_A, y_A])  
        
        label = RS[x_A, y_A]
        
        x.append(temp_save)
        y.append(label)
    x = torch.FloatTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    torch_dataset = Data.TensorDataset(x, y)  
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        drop_last=False
    )  
    return x,y,data2_loader



def Preproces_Data(RS, test_id):01
    copy_RS = RS / 1
    for i in range(test_id.shape[0]):
        x = int(test_id[i][0])
        y = int(test_id[i][1])
        copy_RS[x, y] = 0
    return copy_RS

