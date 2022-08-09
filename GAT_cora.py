import argparse
import os
import random

import scipy.sparse as sp
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable


def encode_onehot(labals):  #对输入的特征进行onehot编码
    classes=set(labals)     #创建无序不重复的序列
    class_dict={c: np.identity(len(classes))[i,:] for i,c in enumerate(classes)}#[i,:]相当于将第i行的所有列赋给i位置所在的元素
    #形如上面的写法，将参数定义好之后先运行for循环中的内容，再将所遍历出的内容赋给相应参数

    labal_onehot=np.array(list(map(class_dict.get,labals)),dtype=np.int32)

    return labal_onehot     #得到每个labals对应的onehot向量

def nomalize(mx):#传入一个稀疏矩阵，（进行归一化的原因）采用加法规则时，对于度大的节点特征越来越大，而对于度小的节点却相反，这可能导致网络训练过程中梯度爆炸或者消失的问题。
    #这种聚合方式实际上就是在对邻接求和取平均，属于非对称的归一化方式
    rownum=np.array(mx.sum(1))#计算传入矩阵的行和,并且变成1维的array
    r_inv=np.power(rownum,-1).flatten()
    r_inv[np.isinf(r_inv)]=0#由于上边进行了求倒数，故可能出现无穷大的情况，所以这里将无穷大变成了0
    r_mat_inv=sp.diags(r_inv)#随后将其变成对角线的形式，这里就是图卷积中的度矩阵D^-1
    mx=r_mat_inv.dot(mx)

    return mx

def accuracy(output,labels):
    pred=output.max(1)[1].type_as(labels)
    correct=pred.eq(labels).double()
    correct=correct.sum()

    return correct/len(labels)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def load_data(path="data/cora/",dataset="cora"):
    print("Loading data...")

    idx_features_labels=np.genfromtxt("{}{}.content".format(path,dataset),dtype=np.dtype(str))#从文本文件加载数据，并按指定处理缺失值
    features=sp.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)
    labels=encode_onehot(idx_features_labels[:,-1])
    #build graph
    idx=np.array(idx_features_labels[:,0],dtype=np.int32)
    idx_map={j:i for i,j in enumerate(idx)}
    edges_unordered=np.genfromtxt("{}{}.cites".format(path,dataset),dtype=np.int32)#cites中分别是（被引用论文的编号）+（引用论文的编号）
    # a=edges_unordered.flatten()
    # c=edges_unordered.shape
    edges=np.array(list(map(idx_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)#edges_unordered.shape=(5429,2)
    # b=idx_map[1033]
    shapeee=np.ones(edges.shape[0])
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0],labels.shape[0]),
                        dtype=np.float32)


    adj=adj+adj.T.multiply(adj.T > adj)-adj.multiply(adj.T > adj)

    features=nomalize(features)
    # a=sp.eye(adj.shape[0])
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))#sp.eye生成对角线全是1的矩阵

    idx_train=range(140)
    idx_val=range(200,500)
    idx_test=range(500,1500)

    features=torch.FloatTensor(np.array(features.todense()))
    labels=torch.LongTensor(np.where(labels)[1])#cora只有7类，变成onehot编码之后就可以用编号0~6来代表类别了
    adj = torch.FloatTensor(np.array(adj.todense()))

    idx_train=torch.LongTensor(idx_train)
    idx_val=torch.LongTensor(idx_val)
    idx_test=torch.LongTensor(idx_test)
    print("数据加载成功...")

    return adj,features,labels,idx_train,idx_val,idx_test

def set_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    print('using gpu {}'.format(gpu))

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # a_input= self._prepare_attentional_mechanism_input(Wh)
        # e=self.leakyrelu(torch.matmul(a_input,self.a).squeeze(2))
        e=self._prepare_attentional_mechanism_input(Wh)

        # N=h.size()[0]
        # a1=Wh.repeat(1,N).view(N*N,-1)
        # a2=Wh.repeat(N,1)
        # a_input=torch.cat([Wh.repeat(1,N).view(N*N,-1),Wh.repeat(N,1)],dim=1).view(N,-1,2*self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))


        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # N=Wh.size()[0]
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        a1=self.a[:self.out_features, :]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        b1=self.a[self.out_features:, :]

        # # broadcast add
        e = Wh1 + Wh2.T#相当于将每个节点和自己还有其他所有节点的注意力系数进行了相加（交互），相当于计算了该节点与其他图中的所有的点的相关性
        return self.leakyrelu(e)
        # N=Wh.size()[0]
        # Wh_repreated_in_chunks=Wh.repeat_interleave(N,dim=0)
        # Wh_repreated_alternating=Wh.repreat(N,1)
        # all_combination_matrix=torch.cat([Wh_repreated_in_chunks,Wh_repreated_alternating],dim=1)

        # return all_combination_matrix.view(N,N,2*self.out_features)

        # all_combinations_matrix=torch.cat([Wh1,Wh2],dim=1)
        # return all_combinations_matrix.view(N,N,2*self.out_features)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    args = parser.parse_args()
    # set_gpu(args.gpu)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # Model and optimizer
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val=F.nll_loss(output[idx_val],labels[idx_val])
        acc_val=accuracy(output[idx_val],labels[idx_val])

        if (epoch % 10 == 0):
            print("Epoch:{}".format(epoch),
                  "loss_train:{:.4f}".format(loss_train.item()),
                  "acc_train:{:.4f}".format(acc_train.item()),
                  "loss_val:{:.4f}".format(loss_val.item()),
                  "acc_val:{:.4f}".format(acc_val.item()))
