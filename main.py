import torch
import math
from numpy import *
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import auc
import datetime
import os
import time
import sys
from tools import fold_5, curve, calculate_TPR_FPR, calculate_AUC_AUPR, EarlyStopping, caculate_TPR_FPR_my, curve_my
sys.path.append("..")
import data_loader

data_file = "./results/"
result_file = "./results/"
early_stop_file = "./results/"


### Relation transformer

#### graph—level attention

class SemanticAttention(nn.Module):
    def __init__(self, in_size, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.num_head = num_head  
        self.att_layers = nn.ModuleList()
        
        for i in range(num_head):
            self.att_layers.append(
                nn.Sequential(
                    nn.Linear(in_size // self.num_head, hidden_size // self.num_head),
                    nn.Tanh(),
                    nn.Linear(hidden_size // self.num_head, 1, bias=False))
            )

    def forward(self, z, r):
        
        z = z.view(z.shape[0], z.shape[1], self.num_head, z.shape[2] // self.num_head)
        
        if r > 1:
            w = self.att_layers[0](z)
        else:
            w = self.att_layers[0](z).mean(0)

        beta = torch.softmax(w, dim=0)
        output = torch.mul(beta, z)
        output = output[0] + output[1]

        for i in range(1, self.num_head):
            w = self.att_layers[i](z)
            beta = torch.softmax(w, dim=0)
            temp = torch.mul(beta, z)
            temp_e = temp[0] + temp[1]
            output += temp_e

        return output.view(output.shape[0], output.shape[1] * output.shape[2])


#### graph transformer

class MyGraphTransformer(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, att_heads: int, beta: bool = True,
                 dropout: float = None, concat=True):
        super(MyGraphTransformer, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.heads = att_heads
        self.dropout = dropout
        self.concat = concat
        self.beta = beta

        # Q K V W layer_norm
        self.Q = nn.Linear(self.in_channel, self.out_channel)
        self.K = nn.Linear(self.in_channel, self.out_channel)
        self.v = nn.Linear(self.in_channel, self.out_channel)
        self.W = nn.Linear(self.in_channel, self.out_channel)
        self.Wr = nn.Linear(256, self.out_channel)
        # self.beta_w = nn.Linear(self.in_channel * 3, self.out_channel)
        self.beta_w = nn.Linear(self.out_channel * 2, self.out_channel)
        self.layer_norm = nn.LayerNorm(self.out_channel)
        # softmax、sigmoid、tanh、dropout
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
                    nn.Linear(self.out_channel, self.out_channel),
                    nn.LayerNorm(self.out_channel, eps=1e-6),
                    nn.Tanh(),
                    nn.Linear(self.out_channel, self.out_channel, bias=False))
            
        # self.dropout = nn.Dropout(self.dropout)
        # reset parameters
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.v.weight)
        nn.init.xavier_normal_(self.W.weight)
        nn.init.xavier_normal_(self.beta_w.weight)

    def forward(self, feature, adj_mat, r, flag=1):
        
        H = self.W(feature)
        r = self.Wr(r)
        
        hQ = self.Q(feature)

        hQ12 = hQ[:708] * r[2]
        hQ21 = hQ[708:4900] * r[2]
        hQ22 = hQ[708:4900] * r[3]
        hK = self.K(feature)

        hK12 = hK[:708] * r[2]
        hK21 = hK[708:4900] * r[2]
        hK22 = hK[708:4900] * r[3]

        if flag == 0:
            hQ11 = hQ[:708] * r[0]
            hK11 = hK[:708] * r[0]
        else:
            hQ11 = hQ[:708] * r[1]
            hK11 = hK[:708] * r[1]

        hQ11 = hQ11.view(self.heads, hQ11.shape[0], self.out_channel // self.heads)
        hQ12 = hQ12.view(self.heads, hQ12.shape[0], self.out_channel // self.heads)
        hQ21 = hQ21.view(self.heads, hQ21.shape[0], self.out_channel // self.heads)
        hQ22 = hQ22.view(self.heads, hQ22.shape[0], self.out_channel // self.heads)

        hK11 = hK11.view(self.heads, hK11.shape[0], self.out_channel // self.heads)
        hK12 = hK12.view(self.heads, hK12.shape[0], self.out_channel // self.heads)
        hK21 = hK21.view(self.heads, hK21.shape[0], self.out_channel // self.heads)
        hK22 = hK22.view(self.heads, hK22.shape[0], self.out_channel // self.heads)
        
        hV = self.v(feature).view(self.heads, feature.shape[0], self.out_channel // self.heads)
        alpha11 = hQ11 @ hK11.transpose(1, 2)
        alpha12 = hQ12 @ hK21.transpose(1, 2)
        alpha21 = hQ21 @ hK12.transpose(1, 2)
        alpha22 = hQ22 @ hK22.transpose(1, 2)
        alpha = torch.cat((torch.cat((alpha11, alpha12), dim=2), torch.cat((alpha21, alpha22), dim=2)), dim=1)
        alpha = alpha / math.sqrt(self.heads)
        alpha = torch.matmul(alpha, adj_mat)
        alpha = self.softmax(alpha)

        m = torch.matmul(alpha, hV)
        m = m.transpose(0, 1) 
        m = m.contiguous()
        m = self.relu(m.view(feature.shape[0], -1))
        m = self.layer_norm(m)
        beta = self.sigmoid(self.beta_w(torch.cat((H, m), dim=1)))

        out = beta * m + (1 - beta) * H

        return self.mlp(out)
    
### MLP

class mixer_layer(nn.Module):

    def __init__(self, in_channel, out_channel, in_dim, out_dim, feat_in_dim, feat_out_dim):
        super().__init__()
        
        self.mlp11 = nn.Linear(in_channel, out_channel)
        self.mlp_channel_up = nn.Linear(in_channel, out_channel)
        
        self.mlp21 = nn.Linear(in_dim, out_dim)
        self.mlp_hight_up = nn.Linear(in_dim, out_dim)
        
        self.mlp31 = nn.Linear(feat_in_dim, feat_out_dim)
        self.mlp_feat_up = nn.Linear(feat_in_dim, feat_out_dim)
        
        self.act, self.layer_norm1 = nn.ReLU(), nn.LayerNorm(feat_in_dim)
        self.reset_parameters()

    def reset_parameters(self):
        
        nn.init.xavier_normal_(self.mlp11.weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.mlp21.weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.mlp31.weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.mlp_channel_up.weight)
        nn.init.xavier_normal_(self.mlp_hight_up.weight)
        nn.init.xavier_normal_(self.mlp_feat_up.weight)

    def forward(self, emb):
        emb1 = self.act(self.mlp11(self.layer_norm1(emb).transpose(1, 3))).transpose(1, 3)
        emb1 = emb1 + self.mlp_channel_up(emb.transpose(1, 3)).transpose(1, 3)
        emb2 = self.act(self.mlp21(self.layer_norm1(emb1).transpose(2, 3))).transpose(2, 3)
        emb2 = emb2 + self.mlp_hight_up(emb1.transpose(2, 3)).transpose(2, 3)
        emb3 = self.act(self.mlp31(self.layer_norm1(emb2))) + self.mlp_feat_up(emb2)

        return emb3
    
### Mymodel

class Mymodel(nn.Module):
    def __init__(self, dim_in, MT_dim_in=256, MT_dim_hid=128, MT_dim_out=64, MT_att_heads=4):
        super(Mymodel, self).__init__()
        self.w_channle = nn.Linear(2, 1)
        self.w_feat = nn.Linear(4900, 256)
        self.node_trans = nn.Embedding(4900, 256)
        self.node_gate = nn.Embedding(4900, 256)
        self.rela_feat = nn.Embedding(4, 256)
        self.rela_gate = nn.Embedding(4, 128)
        self.rela_trans = nn.Embedding(4, 128)
        self.w_share = nn.Linear(128, 256)

        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.gamma = 0.0001

        # RT
        self.GT1 = MyGraphTransformer(in_channel=MT_dim_in, out_channel=MT_dim_hid, att_heads=MT_att_heads)
        self.GT2 = MyGraphTransformer(in_channel=MT_dim_hid, out_channel=MT_dim_out, att_heads=MT_att_heads)
        self.GT3 = MyGraphTransformer(in_channel=MT_dim_in, out_channel=MT_dim_hid, att_heads=MT_att_heads)
        self.GT4 = MyGraphTransformer(in_channel=MT_dim_hid, out_channel=MT_dim_out, att_heads=MT_att_heads)
        self.semantic = SemanticAttention(in_size=MT_dim_out, num_head=MT_att_heads, hidden_size=64)

        # MLP
        self.mixer_layers = nn.ModuleList(
            [
                mixer_layer(2, 16, 2, 8, 4900, 512),
                mixer_layer(16, 8, 8, 2, 512, 128)
            ]
        )

        # end
        self.fc = nn.Sequential(
            nn.Linear(64*2+8*2*128,1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        
        nn.init.xavier_normal_(self.fc[0].weight, nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_normal_(self.fc[3].weight, nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_normal_(self.fc[6].weight)

    def forward(self, left, right, Feat_fun, Feat_str):
        
        Feat = torch.cat((Feat_fun.unsqueeze(2), Feat_str.unsqueeze(2)), dim=2)
        Feat = self.w_channle(Feat).squeeze()
        Feat = self.w_feat(Feat)
        drug = Feat[:708]
        se = Feat[708:]

        d_fun = self.h_gate(drug, self.node_trans.weight[:708], self.node_gate.weight[:708],
                            self.rela_gate.weight[0], self.rela_trans.weight[0])
        d_str = self.h_gate(drug, self.node_trans.weight[:708], self.node_gate.weight[:708],
                            self.rela_gate.weight[1], self.rela_trans.weight[1])
        d_ass = self.h_gate(drug, self.node_trans.weight[:708], self.node_gate.weight[:708],
                            self.rela_gate.weight[2], self.rela_trans.weight[2])
        s_ass = self.h_gate(se, self.node_trans.weight[708:], self.node_gate.weight[708:],
                            self.rela_gate.weight[2], self.rela_trans.weight[2])
        s_sim = self.h_gate(se, self.node_trans.weight[708:], self.node_gate.weight[708:],
                            self.rela_gate.weight[3], self.rela_trans.weight[3])

        d_fun = self.w_share(d_fun)
        d_str = self.w_share(d_str)
        d_ass = self.w_share(d_ass)
        s_ass = self.w_share(s_ass)
        s_sim = self.w_share(s_sim)

        num_fun = torch.count_nonzero(Feat_fun[:708, :708], dim=1)
        num_str = torch.count_nonzero(Feat_str[:708, :708], dim=1)
        num_ass_d = torch.count_nonzero(Feat_fun[:708, 708:], dim=1)
        num_ass_s = torch.count_nonzero(Feat_fun[708:, :708], dim=1)
        num_se = torch.count_nonzero(Feat_fun[708:, 708:], dim=1)

        f_drug = drug + (d_fun + d_str + d_ass) / (num_fun + num_str + num_ass_d).view(708, 1)
        f_se = se + (s_ass + s_sim) / (num_ass_s + num_se).view(4192, 1)

        loss_d_fun = self.KG_loss(f_drug, f_drug, self.rela_feat.weight[0], Feat_fun[:708, :708], 0)
        loss_d_str = self.KG_loss(f_drug, f_drug, self.rela_feat.weight[1], Feat_str[:708, :708], 0)
        loss_d_ass = self.KG_loss(f_drug, f_se, self.rela_feat.weight[2], Feat_fun[:708, 708:], 1)
        loss_s_ass = self.KG_loss(f_se, f_drug, self.rela_feat.weight[2], Feat_fun[708:, :708], 1)
        loss_s_sim = self.KG_loss(f_se, f_se, self.rela_feat.weight[3], Feat_fun[708:, 708:], 0)

        feature1 = torch.cat((f_drug, f_se), dim=0)
        
        feat1 = self.GT1(feature1, Feat_fun, self.rela_feat.weight.data, 0)
        feat1 = self.GT2(feat1, Feat_fun, self.rela_feat.weight.data, 0)
        feat2 = self.GT3(feature1, Feat_str, self.rela_feat.weight.data, 1)
        feat2 = self.GT4(feat2, Feat_str, self.rela_feat.weight.data, 1)

        out_in = torch.stack([feat1, feat2], dim=0)
        out_RT = self.semantic(out_in, 2)

        emb = torch.stack((Feat_fun[left], Feat_fun[right + 708]), dim=1)
        emb2 = torch.stack((Feat_str[left], Feat_str[right + 708]), dim=1)
        emb3 = torch.cat((emb.unsqueeze(1), emb2.unsqueeze(1)), dim=1)
        for mix_layer in self.mixer_layers:
            emb3 = mix_layer(emb3)
        emb3 = emb3.view(emb3.shape[0], -1)

        out_rt = torch.cat((out_RT[left], out_RT[right+708]), dim=1)
        out = torch.cat((emb3, out_rt), dim=1)
        out = self.fc(out)

        loss = (loss_d_fun + loss_d_str + loss_d_ass + loss_s_ass + loss_s_sim) * 1e-5

        return out, loss

    def h_gate(self, h, h_m, h_g, r_gate, r_trans):
        h_trans = torch.mul(h_m, h).sum(dim=1).unsqueeze(1)
        h_g = torch.mul(h_g, h).sum(dim=1).unsqueeze(1)
        h_gate = self.sigmoid(torch.mul(h_g, r_gate))
        h_feat = self.tanh(torch.mul(h_trans, r_trans))
        h_feat = torch.mul((1 - h_gate), h_feat)

        return h_feat

    def KG_loss(self, h, t, r, true_score, flag):
        eps = 0.7
        h_ = r * h
        score = torch.matmul(h_, t.T)
        if flag:
            score[score < eps] = 0
            score[score >= eps] = 1
        loss_cross = nn.CrossEntropyLoss()
        KG_loss = loss_cross(score, true_score)
        return KG_loss
    
###

one_positive, zero_positive = data_loader.find_one_zero(RS)
data_0, data_1 = data_loader.index_dp(np.array(RS), 5)

mymodel = Mymodel(4900)
mymodel = mymodel.cuda(device)
mymodel.train()

TPR_ALL = []
FPR_ALL = []
P_ALL = []

for test in range(5):
    torch.cuda.empty_cache()
    print("The {} fold".format(test + 1))
    train_data = []
    test_data = []
    train_data, test_data, t_data = data_loader.Split_data(data_1, data_0, test, 5, RS)
    train_data = np.array(train_data)
    np.savetxt(data_file+"train_data2_" + str(test) + ".txt", np.array(train_data))
    test_data = np.array(test_data)
    t_data = np.array(t_data)
    np.savetxt(data_file+"test_data_"+ str(test) + ".txt", np.array(test_data))
    np.savetxt(data_file+"t_data_" + str(test) + ".txt", np.array(t_data))
    new_RS = data_loader.Preproces_Data(RS, test_data)
    train_x, train_y, train_loader = data_loader.load_data(train_data, 128, np.array(RS))
    test_x, test_y, test_loader = data_loader.load_data(test_data, 100, np.array(RS))
    t_x, t_y, t_loader = data_loader.load_data(t_data, 100, np.array(RS))
    print("train:{},test:{},t:{}".format(train_y.sum(), test_y.sum(), t_y.sum()))

    drug_fun = drug_fun.cpu()
    drug_str = drug_str.cpu()
    new_RS = new_RS.cpu()
    S = S.cpu()
    feat_str = torch.cat((drug_str, new_RS), dim=1)
    feat_fun = torch.cat((drug_fun, new_RS), dim=1)
    feat_s = torch.cat((new_RS.transpose(0, 1), S), dim=1)
    Feat_str = torch.cat((feat_str, feat_s), dim=0)
    Feat_fun = torch.cat((feat_fun, feat_s), dim=0)

    feat_str = feat_str.cuda(device)
    feat_fun = feat_fun.cuda(device)
    feat_s = feat_s.cuda(device)

    Feat_str = Feat_str.cuda(device)
    Feat_fun = Feat_fun.cuda(device)

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()
    file = early_stop_file+"checkpoint_" + str(test) + ".pt"
    early_stopping = EarlyStopping(patience=100, verbose=True, save_path=file)


    print("----begin Training----")
    max_acc = 0
    gamma = 0.00001
    for epoch in range(60):
        since = time.time()
        train_loss = 0  
        train_acc = 0
        t_test_acc = 0
        print("----The {} epoch begin----".format(epoch + 1))
        for step, (x, train_label) in enumerate(train_loader):  
            mymodel.train()
            x = torch.squeeze(x)
            xy_batch = x.type(torch.long).cuda(device)
            t_start = time.time()
            try:
                out, loss_kg = mymodel(xy_batch[:, 0], xy_batch[:, 1], Feat_fun, Feat_str)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            t_end = time.time()

            train_label1 = []
            train_label1.extend(train_label) 

            y = torch.LongTensor(np.array(train_label1).astype(int64))  
            y = Variable(y).cuda(device)  

            loss = loss_func(out, y) + loss_kg  
            optimizer.zero_grad()  
            try:
                loss.backward()  
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            optimizer.step()  
            train_loss += loss.item()
            
            _, pred = out.max(1)  
            num_correct = (pred == y).sum().item()  
            acc = num_correct / x.shape[0]  
            train_acc += acc  
            if step % 500 == 0:
                print('Test:{}.Epoch: {},step:{} Train Loss: {:.8f}, Train Acc: {:.6f}'
                      .format(test + 1, epoch + 1, step, train_loss / len(train_loader), acc ))
           
            if step % 20 == 0 and step != 0:
                for step, (t_x, t_y) in enumerate(t_loader):
                    mymodel.eval()
                    t_x = torch.squeeze(t_x).type(torch.long).cuda(device)
                    t_y = torch.Tensor(t_y).type(torch.int64).cuda(device)
                    with torch.no_grad():
                        logp, t_loss_kg = mymodel(t_x[:, 0], t_x[:, 1], Feat_fun, Feat_str)
                        t_correct_num = (torch.max(logp, dim=1)[1] == t_y).float().sum()
                        t_acc = t_correct_num / x.shape[0]  
                        t_test_acc += t_acc  
                        val_loss = loss_func(logp, t_y) + t_loss_kg
                    t_end = time.time()
                    if step % 500 == 0:
                        print(
                            f'test:{test + 1},epoch: {epoch + 1}, val loss: {val_loss.item()},  time: {t_end - t_start}, acc: {t_acc }')
                early_stopping(val_loss, mymodel)
            if early_stopping.early_stop:
                print(f'early_stoppin!')
                break
        if early_stopping.early_stop:
            print(f'early_stopping!')
            break
        if train_acc / len(train_loader) > max_acc:
            max_acc = train_acc / len(train_loader)
        
        print('Test:{}, Epoch: {}, Train Loss: {:.8f}, Train Acc: {:.6f}'
              .format(test + 1, epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader)))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        

    print("Training finish")
    

#####################test###############################
    mymodel.load_state_dict(torch.load(file))
    mymodel.eval()
    test_acc = 0
    num_cor = 0
    o = np.zeros((0, 2))
    z = 0
    s_time = time.time()
    print("begin test")
    for test_x, test_label in test_loader:
        z = z + test_x.shape[0]
        if z % 10000 == 0:
            print(z)
        test_label1 = []
        test_x = torch.squeeze(test_x)
        test_x_batch = test_x.type(torch.long).cuda(device)
        test_label1.extend(test_label)
        y = torch.LongTensor(np.array(test_label1).astype(int))
        y = Variable(y).cuda(device)
        right_test_out, _, = mymodel(test_x_batch[:, 0], test_x_batch[:, 1], Feat_fun, Feat_str)
        right_test_out = F.softmax(right_test_out, dim=1)  

        _, pred_y = right_test_out.max(1)
        num_correct = (pred_y == y).sum().item()
        t_acc = num_correct / test_x.shape[0]
        test_acc += t_acc
        num_cor += num_correct
        o = np.vstack((o, right_test_out.detach().cpu().numpy()))
    e_time = time.time() - s_time
    print('Testing complete in {:.0f}m {:.0f}s'.format(e_time // 60, e_time % 60))
    print('cor_num:{}'.format(num_cor))
    np.savetxt(result_file+"test_out_" + str(test) + ".txt", o)
    print("Test finish！")

    # ###################B##########################
   
    B = np.array(torch.Tensor.cpu(RS)) / 1
    for i in range(train_data.shape[0]):
        B[int(train_data[i][0])][int(train_data[i][1])] = -1
   
    test_out = np.loadtxt(result_file+"test_out_" + str(test) + ".txt")
    np.savetxt(result_file+"B" + str(test) + ".txt", B)
    
    ##########################R#############################################
    
    R = np.zeros(shape=(RS.shape[0], RS.shape[1]))  

    for i in range(train_data.shape[0]):
        R[int(train_data[i][0])][int(train_data[i][1])] = -1

    for i in range(test_data.shape[0]):
        R[int(test_data[i][0])][int(test_data[i][1])] = test_out[i][1]
    np.savetxt(result_file+"R" + str(test) + ".txt", R)
    
    ###########################f##############################
    
    f = np.zeros(shape=(R.shape[0], 1))
    for i in range(R.shape[0]):
        f[i] = np.sum(R[i] > (-1))
    np.savetxt(result_file+"f" + str(test) + ".txt", f)
    t_endtime = time.time() - starttime
    print('complete in {:.0f}m {:.0f}s'.format(t_endtime // 60, t_endtime % 60))

    TPR, FPR, P = caculate_TPR_FPR_my(R, f, B)
    MIN = min(MIN, int(len(TPR)))
    TPR_ALL.append(TPR)
    FPR_ALL.append(FPR)
    P_ALL.append(P)
    np.savetxt(result_file+"TPR" + str(test) + ".txt", TPR)
    np.savetxt(result_file+"FPR" + str(test) + ".txt", FPR)
    np.savetxt(result_file+"P" + str(test) + ".txt", P)
    
    print("#######result#######")
    
    strin = result_file+"Plt_in_" + str(test) + ".png"
    curve_my(TPR, FPR, P, strin)
    
Rh, labelh = [], []
path = result_file+"Plt.png"
for i in range(5):
    tlh = np.loadtxt(result_file+"B%d.txt" % i)
    trh = np.loadtxt(result_file+"R%d.txt" % i)
    Rh.append(trh)
    labelh.append(tlh)
end_time = time.time() - starttime
print('complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
calculate_AUC_AUPR(Rh, labelh, path)