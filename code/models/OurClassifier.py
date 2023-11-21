import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,confusion_matrix
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
from dgl.dataloading import GraphDataLoader
import numpy as np
import torch
import torch as t
import os
from utils.config import *
from models.MLP import *
import random
import math

params = config()


def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True


array = [983, 523, 697, 689, 523]

def PositionalEncoding(position, d_model, maxlen, device):
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device)
    pe = torch.zeros(maxlen, 1, d_model).to(device)
    pe[:, 0, 0::2] = torch.sin(position * div_term).to(device)
    pe[:, 0, 1::2] = torch.cos(position * div_term).to(device)
    return pe

class BaseClassifier:
    def __init__(self):
        pass

    def cv_train(self, dataset, batchSize=32, num_epochs=10, lr=0.001, kFold=5, savePath='checkpoints/', earlyStop=30,
                 seed=10, device=t.device('cpu')):
        splits = StratifiedKFold(n_splits=kFold, shuffle=True,
                                 random_state=10)  # StratifiedKFold的好处在于他会使得每次抽样的样本中的数据都符合总数据集的数据分布
        fold_best = []
        if not os.path.exists(savePath):  # 如果savePath不存在就生成一个
            os.mkdir(savePath)
        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset[:][0], dataset[:][2])):  # 这里dataset[:][0]存的是图信息，[1]存的是label信息， 接下来要遍历所有生成的fold,以及里面的train_index和val_index
            savePath2 = savePath + f"fold{fold + 1}"  # 保存路径
            setup_seed(array[fold])  # 给每一个fold设定随机种子
            self.reset_parameters()  # 重置参数
            best_f = 0.0
            print('>>>>>>Fold{}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_idx)  # Samples elements randomly from a given list of indices, without replacement.
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = GraphDataLoader(dataset, batch_size=batchSize, sampler=train_sampler, num_workers=0)
            test_loader = GraphDataLoader(dataset, batch_size=batchSize, sampler=test_sampler, num_workers=0)
            optimizer = t.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=0.00001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)

            lossfn = nn.CrossEntropyLoss()
            #lossfn = FocalLoss(gamma=2)
            #lossfn = nn.BCELoss()
            best_record = {'train_loss': 0, 'test_loss': 0, 'train_acc': 0, 'test_acc': 0, 'train_f': 0,
                           'test_f': 0, 'train_pre': 0, 'test_pre': 0, 'train_rec': 0, 'test_rec': 0,
                           'train_roc': 0, 'test_roc': 0}
            nobetter = 0

            for epoch in range(num_epochs):
                train_loss, train_acc, train_f, train_pre, train_rec, train_roc = self.train_epoch(train_loader,
                                                                                                   lossfn,
                                                                                                   optimizer,
                                                                                                   device)

                test_loss, test_acc, test_f, test_pre, test_rec, test_roc = self.valid_epoch(test_loader, lossfn,device)

            #是否使用scheduler
            #    scheduler.step(test_loss)

                print(
                    ">>>Epoch:{} of Fold{} AVG Train Loss:{:.3f}, AVG Test Loss:{:.3f}\n"
                    "Train Acc:{:.3f} %, Train F1-score:{:.3f}, Train Precision:{:.3f}, Train Recall:{:.3f}, Train ROC:{:.3f};\n"
                    "Test Acc:{:.3f} %, Test F1-score:{:.3f}, Test Precision:{:.3f}, Test Recall:{:.3f}, Test ROC:{:.3f}!!!\n".format(
                        epoch + 1, fold + 1, train_loss, test_loss,
                        train_acc * 100, train_f, train_pre, train_rec, train_roc,
                        test_acc * 100, test_f, test_pre, test_rec, test_roc))

                if best_f < test_f:
                    nobetter = 0
                    best_f = test_f
                    best_record['train_loss'] = train_loss
                    best_record['test_loss'] = test_loss
                    best_record['train_acc'] = train_acc
                    best_record['test_acc'] = test_acc
                    best_record['train_f'] = train_f
                    best_record['test_f'] = test_f
                    best_record['train_pre'] = train_pre
                    best_record['test_pre'] = test_pre
                    best_record['train_rec'] = train_rec
                    best_record['test_rec'] = test_rec
                    best_record['train_roc'] = train_roc
                    best_record['test_roc'] = test_roc
                    print(f'>Bingo!!! Get a better Model with valid F1-score: {best_f:.3f}!!!')

                    self.save("%s.pkl" % savePath2, epoch + 1, best_f)
                else:
                    nobetter += 1
                    if nobetter >= earlyStop:
                        print(
                            f'Test F1-score has not improved for more than {earlyStop} steps in epoch {epoch + 1}, stop training.')
                        break
            fold_best.append(best_record)
            print(f'*****The Fold{fold + 1} is done!')
            print(
                "Finally,the best model's Test Acc:{:.3f} %, Test F1-score:{:.3f}, Test Precision:{:.3f}, Test Recall:{:.3f}, Test ROC:{:.3f}!!!\n\n\n".format(
                    best_record['test_acc'] * 100, best_record['test_f'], best_record['test_pre'],
                    best_record['test_rec'], best_record['test_roc']))
            os.rename("%s.pkl" % savePath2,
                      "%s_%s.pkl" % (savePath2, ("%.3f" % best_f)))
        # Print the result of 5 folds
        print('*****All folds are done!')
        print("=" * 20 + "FINAL RESULT" + "=" * 20)
        # Print table header
        row_first = ["Fold", "ACC", "F1-score", "Precision", "Recall", "ROC"]
        print("".join(f"{item:<12}" for item in row_first))
        # Print table content
        metrics = ['test_f', 'test_pre', 'test_rec', 'test_roc']
        for idx, fold in enumerate(fold_best):
            print(f"{idx + 1:<12}" + "%-.3f" % (fold['test_acc'] * 100) + "%-6s" % "%" + "".join(
                f"{fold[key]:<12.3f}" for key in metrics))
        # Print average
        avg, metrics2 = {}, ['test_acc', 'test_f', 'test_pre', 'test_rec', 'test_roc']
        for item in metrics2:
            avg[item] = 0
            for fold in fold_best:
                avg[item] += fold[item]
            avg[item] /= len(fold_best)
        print(f"%-12s" % "Average" + "%-.3f" % (avg['test_acc'] * 100) + "%-6s" % "%" + "".join(
            f"{avg[key]:<12.3f}" for key in metrics))
        print("=" * 52)

    def train_epoch(self, dataloader, loss_fn, optimizer, device):
        train_loss, train_acc, train_f, train_pre, train_rec, train_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.to_train_mode()
        pred_list = []
        label_list = []
        for _, (batched_graph_miRNA, batched_graph_MRNA, labels) in enumerate(dataloader):
            # print(len(labels))
            batched_graph_miRNA, batched_graph_MRNA, labels = batched_graph_miRNA.to(device), batched_graph_MRNA.to(device), labels.to(device)

            feats1 = batched_graph_miRNA.ndata['attr']
            feats2 = batched_graph_MRNA.ndata['attr']
            # print(feats.shape)

            optimizer.zero_grad()
            output = self.calculate_y(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2)
            # print(output.shape)
            output, labels = output.to(t.float32), labels.to(t.int64)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            # pred_list loads the predicted values of the training set samples
            pred_list.extend(output.detach().cpu().numpy())

            # label_list loads the true values of the training set samples (one-hot form)
            label_list.extend(labels.cpu().numpy())

        with t.no_grad():
            pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
            train_loss = loss_fn(pred_tensor, label_tensor)
        pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()
        #print(np.argmax(pred_array, axis=1)) #后面改一波min, 4个
        train_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
        train_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
        train_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro',
                                    zero_division=0)
        train_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
        train_roc = roc_auc_score(F.one_hot(t.tensor(label_list), num_classes=params.n_classes).cpu().numpy(),
                                  pred_array, average='micro')
        return train_loss, train_acc, train_f, train_pre, train_rec, train_roc

    def valid_epoch(self, dataloader, loss_fn, device):
        # self.to_eval_mode()
        with t.no_grad():
            valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            self.to_eval_mode()
            pred_list = []
            label_list = []
            for _, (batched_graph_miRNA, batched_graph_MRNA, labels) in enumerate(dataloader):
                # print(batched_graph)
                batched_graph_miRNA, batched_graph_MRNA, labels = batched_graph_miRNA.to(device), batched_graph_MRNA.to(device), labels.to(device)
                feats1 = batched_graph_miRNA.ndata['attr']
                feats2 = batched_graph_MRNA.ndata['attr']

                output = self.calculate_y(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2)
                output, labels = output.to(t.float32), labels.to(t.int64)

                # pred_list loads the predicted values of the training set samples
                pred_list.extend(output.detach().cpu().numpy())

                # label_list loads the true values of the training set samples (one-hot form)
                label_list.extend(labels.cpu().numpy())
            with t.no_grad():
                pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
                valid_loss = loss_fn(pred_tensor, label_tensor)
            pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()
            #argmax to argmin
            valid_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
            valid_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
            valid_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro',
                                        zero_division=0)
            valid_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
            valid_roc = roc_auc_score(F.one_hot(t.tensor(label_list), num_classes=params.n_classes).cpu().numpy(),
                                      pred_array, average='micro')
        return valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc

    def to_train_mode(self):
        for module in self.moduleList:
            module.train()

    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()

    def save(self, path, epochs, bestMtc=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc}
        for idx, module in enumerate(self.moduleList):
            stateDict[idx] = module.state_dict()
        t.save(stateDict, path)
        print('Model saved in "%s".\n' % path)

    def load(self, path, map_location=None):
        parameters = t.load(path, map_location=map_location)
        for idx, module in enumerate(self.moduleList):
            module.load_state_dict(parameters[idx])
        print("%d epochs and %.3lf val Score 's model load finished." % (parameters['epochs'], parameters['bestMtc']))

    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()

class MLP_layer(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GraphClassifier(BaseClassifier):
    def __init__(self, in_dim, hidden_dim, n_classes, device=t.device("cpu")):
        super(GraphClassifier, self).__init__()
        #only gcn
        #self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        #using batch-norm
        #self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
        #self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        #self.bnorm2 = nn.BatchNorm1d(hidden_dim).to(device)
        #self.conv3 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        #using batch-norm
        #self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
        #self.conv4 = dglnn.GraphConv(hidden_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        #self.bnorm4 = nn.BatchNorm1d(hidden_dim).to(device)

        #only GAT

        #self.conv1 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=4,allow_zero_in_degree=True).to(device)
        #self.conv1 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1, allow_zero_in_degree=True).to(device)
        #self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
        #self.bnorm1 = nn.BatchNorm1d(hidden_dim*4).to(device)
        #self.conv3 = dglnn.GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=1,
        #                           allow_zero_in_degree=True).to(device)
        #self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
        #self.conv3 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=4,
        #                           allow_zero_in_degree=True).to(device)
        #self.conv3 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1,
        #                           allow_zero_in_degree=True).to(device)

        #self.bnorm3 = nn.BatchNorm1d(hidden_dim*4).to(device)
        #self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
        #self.conv4 = dglnn.GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=1,
        #                           allow_zero_in_degree=True).to(device)
        #self.bnorm4 = nn.BatchNorm1d(hidden_dim).to(device)

        #GCN + GAT
        #self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        #self.conv2 = dglnn.GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=1, feat_drop = 0.2, attn_drop=0.2,allow_zero_in_degree=True).to(device)
        #self.conv3 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        #self.conv4 = dglnn.GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=1, feat_drop=0.2, attn_drop=0.2,allow_zero_in_degree=True).to(device)

        #双层GAT
        #self.conv1 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=4,allow_zero_in_degree=True).to(device)
        # self.conv2 = dglnn.GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=4, feat_drop=0.2, attn_drop=0.2,
        #                           allow_zero_in_degree=True).to(device)
        #self.conv3 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=4,
        #                           allow_zero_in_degree=True).to(device)
        # self.conv4 = dglnn.GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=4, feat_drop=0.2, attn_drop=0.2,
        #                           allow_zero_in_degree=True).to(device)

        #GIN
        #using 2-layer mlp
        mlp1 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)

        self.conv1 = dglnn.GINConv(mlp1, 'sum').to(device)

        #using simple linear
        #self.lin1 = nn.Linear(in_dim,hidden_dim,bias=True).to(device)
        #self.lin3 = nn.Linear(hidden_dim,hidden_dim,bias=True).to(device)
        #self.lin5 = nn.Linear(hidden_dim,int(hidden_dim*0.5),bias=True).to(device)

        #self.conv1 = dglnn.GINConv(self.lin1,'sum').to(device)
        #self.conv2 = dglnn.GINConv(self.lin3,'sum').to(device)
        #self.conv5 = dglnn.GINConv(self.lin5, 'sum').to(device)
        #using batch-norm
        #self.bnorm1 = nn.BatchNorm1d(int(hidden_dim)).to(device)
        #self.bnorm2 = nn.BatchNorm1d(hidden_dim).to(device)

        #using 2-layer mlp
        mlp2 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)

        self.conv3 = dglnn.GINConv(mlp2,'sum').to(device)

        #using simple linear
        #self.lin2 = nn.Linear(in_dim, hidden_dim, bias=True).to(device)
        #self.lin4 = nn.Linear(hidden_dim, hidden_dim, bias=True).to(device)
        #self.lin6 = nn.Linear(hidden_dim, int(hidden_dim*0.5), bias=True).to(device)
        #self.conv3 = dglnn.GINConv(self.lin2,'sum').to(device)
        #self.conv4 = dglnn.GINConv(self.lin4, 'sum').to(device)
        #self.conv6 = dglnn.GINConv(self.lin6, 'sum').to(device)
        # using batch-norm
        #self.bnorm3 = nn.BatchNorm1d(int(hidden_dim)).to(device)
        #self.bnorm4 = nn.BatchNorm1d(hidden_dim).to(device)


        #GraphSage
        #self.conv1 = dglnn.SAGEConv(in_dim, hidden_dim, norm=None, aggregator_type='pool').to(device)
        #self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
        #self.conv3 = dglnn.SAGEConv(in_dim, hidden_dim, norm=None, aggregator_type='pool').to(device)
        #self.bnorm2 = nn.BatchNorm1d(hidden_dim).to(device)

        self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
        #self.classify = MLP(inSize=hidden_dim*2*4, outSize=n_classes).to(device)
        #self.moduleList = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])
        self.moduleList = nn.ModuleList([self.conv1, self.conv3, self.classify])

        #for GIN
        #self.moduleList = nn.ModuleList([self.conv1,self.bnorm1 ,self.bnorm3, self.conv3, self.classify])
        #self.moduleList = nn.ModuleList([self.lin1, self.lin2, self.lin3, self.lin4, self.conv1, self.conv2, self.conv3, self.conv4, self.bnorm1,
        #                                    self.bnorm3, self.classify])

        #for mlp GIN
        #self.moduleList = nn.ModuleList([self.conv1, self.conv3, self.bnorm1, self.bnorm3, self.classify])
        #For simple linear GIN
        #self.moduleList = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])


        self.device = device

    def calculate_y(self, g1, h1, g2, h2):  # g是batched graph data. h是node_features
        # Apply graph convolution networks and activation functions
        #original
        #h1 = F.relu(self.conv1(g1, h1, edge_weight=g1.edata['weight']))  # e.g.torch.Size([142, 64])
        #print(h1.shape) #1127,128
        #adding batchnorm
        h1 = self.conv1(g1, h1, edge_weight=g1.edata['weight'])
        #print(h1.shape)

        #add one more layer
        #h1 = self.conv2(g1, h1, edge_weight=g1.edata['weight'])

        #h1 = self.conv5(g1, h1, edge_weight=g1.edata['weight'])

        #h1 = self.bnorm1(h1)
        #for multi-head
        #h1 = self.bnorm1(t.reshape(h1,(h1.shape[0], h1.shape[1]*h1.shape[2])))

        #h1 = F.relu(h1)


        #去掉relu
        #h1 = self.conv1(g1, h1, edge_weight=g1.edata['weight'])
        #print(h1.shape)
        #for gcn part
        '''
        h1 = self.conv2(g1, h1, edge_weight=g1.edata['weight'])
        h1 = self.bnorm2(h1)
        h1 = F.relu(h1)# e.g.torch.Size([142, 64])
        '''

        #original
        #h2 = F.relu(self.conv3(g2, h2, edge_weight=g2.edata['weight']))

        #adding batchnorm
        h2 = self.conv3(g2, h2, edge_weight=g2.edata['weight'])

        #add one more layer
        #h2 = self.conv4(g2, h2, edge_weight=g2.edata['weight'])

        #h2 = self.conv6(g2, h2, edge_weight=g2.edata['weight'])
        #h2 = self.bnorm3(h2)
        #h2 = self.bnorm3(t.reshape(h2,(h2.shape[0], h2.shape[1]*h2.shape[2])))
        #h2 = F.relu(h2)

        '''
        h2 = self.conv4(g2, h2, edge_weight=g2.edata['weight'])
        h2 = self.bnorm4(h2)
        h2 = F.relu(h2)
        '''
        #去掉relu
        #h2 = self.conv3(g2, h2, edge_weight=g2.edata['weight'])

        #for gcn part
        #h2 = F.relu(self.conv4(g2, h2))


        # print(h.shape)
        with g1.local_scope():
            with g2.local_scope():
                # 这行式子的意思是后来对图的变化不会对original的图产生影响
                #print(h1.shape)
                g1.ndata['h'] = h1

                g2.ndata['h'] = h2
                # Use the average readout to get the graph representation
                #hg1 = dgl.mean_nodes(g1, 'h')
                #print(h1)
                hg1 = dgl.mean_nodes(g1, 'h')


                #hg1_num = g1.batch_num_nodes().unsqueeze(1)
                #hg1_pe = PositionalEncoding(hg1_num,16, hg1_num.shape[0], self.device).squeeze(1)
                #hg1 = torch.cat((hg1, hg1_pe), 1)


                hg2 = dgl.mean_nodes(g2, 'h')

                #hg2_num = g2.batch_num_nodes().unsqueeze(1)
                #hg2_pe = PositionalEncoding(hg2_num,16, hg2_num.shape[0], self.device).squeeze(1)

                #hg2 = torch.cat((hg2, hg2_pe), 1)



                # Using topk average readout
                #hg1, _= dgl.topk_nodes(g1, 'h', 5)
                #hg1 = torch.reshape(hg1, (hg1.shape[0], hg1.shape[1] * hg1.shape[2]))
                #hg2, _ = dgl.topk_nodes(g2, 'h', 5)
                #hg2 = torch.reshape(hg2, (hg2.shape[0], hg2.shape[1] * hg2.shape[2]))



                hg = t.cat((hg1,hg2), -1)
                #print(hg.shape)

                #for GAT
                #for multihead
                #using mean
                #hg = t.mean(hg,1)

                #using cat
                #hg = t.reshape(hg,(hg.shape[0],hg.shape[1]*hg.shape[2]))
                #print(hg.shape)


                # print(hg.shape) #e.g. torch.Size([8, 64])，8这里对应batch_size
            return self.classify(hg)
