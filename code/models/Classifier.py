import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
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


# def PositionalEncoding(position, d_model, maxlen, device):
#     div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device)
#     pe = torch.zeros(maxlen, 1, d_model).to(device)
#     pe[:, 0, 0::2] = torch.sin(position * div_term).to(device)
#     pe[:, 0, 1::2] = torch.cos(position * div_term).to(device)
#     return pe

class BaseClassifier:
    def __init__(self):
        pass

    def cv_train(self, dataset, batchSize=32, num_epochs=10, lr=0.001, kFold=5, savePath='checkpoints/', earlyStop=30,
                 seed=10, device=t.device('cpu'), GNN_type='gcn'):
        splits = StratifiedKFold(n_splits=kFold, shuffle=True,
                                 random_state=10)
        fold_best = []
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset[:][0], dataset[:][2])):
            savePath2 = savePath + f"fold{fold + 1}"
            setup_seed(array[fold])
            self.reset_parameters()
            best_f = 0.0
            print('>>>>>>Fold{}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(
                train_idx)  # Samples elements randomly from a given list of indices, without replacement.
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = GraphDataLoader(dataset, batch_size=batchSize, sampler=train_sampler, num_workers=0)
            test_loader = GraphDataLoader(dataset, batch_size=batchSize, sampler=test_sampler, num_workers=0)
            optimizer = t.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=0.00001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                                   # verbose=False,
                                                                   threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                                   min_lr=0,
                                                                   eps=1e-08)

            lossfn = nn.CrossEntropyLoss()
            best_record = {'train_loss': 0, 'test_loss': 0, 'train_acc': 0, 'test_acc': 0, 'train_f': 0,
                           'test_f': 0, 'train_pre': 0, 'test_pre': 0, 'train_rec': 0, 'test_rec': 0,
                           'train_roc': 0, 'test_roc': 0}
            nobetter = 0

            for epoch in range(num_epochs):
                train_loss, train_acc, train_f, train_pre, train_rec, train_roc = self.train_epoch(train_loader,
                                                                                                   lossfn,
                                                                                                   optimizer,
                                                                                                   device,
                                                                                                   GNN_type)

                test_loss, test_acc, test_f, test_pre, test_rec, test_roc = self.valid_epoch(test_loader, lossfn,
                                                                                             device, GNN_type)

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
                    # best_loss = test_loss
                    # best_roc = test_roc
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
                "Finally,the best model's Test loss:{:.5f}, Test Acc:{:.3f} %, Test F1-score:{:.3f}, Test Precision:{:.3f}, Test Recall:{:.3f}, Test ROC:{:.3f}!!!\n\n\n".format(
                    best_record['test_loss'], best_record['test_acc'] * 100, best_record['test_f'],
                    best_record['test_pre'],
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

    def train_epoch(self, dataloader, loss_fn, optimizer, device, GNN_type):
        train_loss, train_acc, train_f, train_pre, train_rec, train_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.to_train_mode()
        pred_list = []
        label_list = []
        for _, (batched_graph_miRNA, batched_graph_MRNA, labels) in enumerate(dataloader):
            batched_graph_miRNA, batched_graph_MRNA, labels = batched_graph_miRNA.to(device), batched_graph_MRNA.to(
                device), labels.to(device)

            feats1 = batched_graph_miRNA.ndata['attr']
            feats2 = batched_graph_MRNA.ndata['attr']

            optimizer.zero_grad()
            output = self.calculate_y(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2, GNN_type)
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
        train_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
        train_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)
        train_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)
        train_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)
        train_roc = roc_auc_score(np.array(label_list), pred_array[:, 1])

        return train_loss, train_acc, train_f, train_pre, train_rec, train_roc

    def valid_epoch(self, dataloader, loss_fn, device, GNN_type):
        with t.no_grad():
            valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            self.to_eval_mode()
            pred_list = []
            label_list = []
            for _, (batched_graph_miRNA, batched_graph_MRNA, labels) in enumerate(dataloader):
                # print(batched_graph)
                batched_graph_miRNA, batched_graph_MRNA, labels = batched_graph_miRNA.to(device), batched_graph_MRNA.to(
                    device), labels.to(device)
                feats1 = batched_graph_miRNA.ndata['attr']
                feats2 = batched_graph_MRNA.ndata['attr']

                output = self.calculate_y(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2, GNN_type)
                output, labels = output.to(t.float32), labels.to(t.int64)

                # pred_list loads the predicted values of the training set samples
                pred_list.extend(output.detach().cpu().numpy())

                # label_list loads the true values of the training set samples (one-hot form)
                label_list.extend(labels.cpu().numpy())
            with t.no_grad():
                pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
                valid_loss = loss_fn(pred_tensor, label_tensor)
            pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()
            # argmax to argmin
            valid_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
            valid_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)
            valid_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)
            valid_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)
            valid_roc = roc_auc_score(np.array(label_list), pred_array[:, 1])

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
    def __init__(self, in_dim, hidden_dim, n_classes, GNN_type, device=t.device("cpu")):
        super(GraphClassifier, self).__init__()
        if GNN_type == "gcn":
            self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
            self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
            self.conv3 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
            self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
            self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
            self.moduleList = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])

        elif GNN_type == "gat":
            self.conv1 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1,
                                       allow_zero_in_degree=True).to(
                device)
            self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
            self.conv3 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1,
                                       allow_zero_in_degree=True).to(device)
            self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
            self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
            self.moduleList = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])

        else:
            mlp1 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)
            self.conv1 = dglnn.GINConv(mlp1, 'sum').to(device)
            # using 2-layer mlp
            mlp2 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)
            self.conv3 = dglnn.GINConv(mlp2, 'sum').to(device)
            self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
            self.moduleList = nn.ModuleList([self.conv1, self.conv3, self.classify])

        self.device = device

    def calculate_y(self, g1, h1, g2, h2, GNN_type):
        # Apply graph neural networks and activation functions

        if GNN_type == "gcn":
            h1 = h1.to(torch.float32)
            h1 = self.conv1(g1, h1, edge_weight=g1.edata['weight'])
            h1 = self.bnorm1(h1)
            h1 = F.relu(h1)
            h2 = h2.to(torch.float32)
            h2 = self.conv3(g2, h2, edge_weight=g2.edata['weight'])
            h2 = self.bnorm3(h2)
            h2 = F.relu(h2)

        elif GNN_type == "gat":
            h1 = h1.to(torch.float32)
            h1 = self.conv1(g1, h1)
            h1 = self.bnorm1(t.reshape(h1, (h1.shape[0], h1.shape[1] * h1.shape[2])))
            h1 = F.relu(h1)
            h2 = h2.to(torch.float32)
            h2 = self.conv3(g2, h2)
            h2 = self.bnorm3(t.reshape(h2, (h2.shape[0], h2.shape[1] * h2.shape[2])))
            h2 = F.relu(h2)

        else:
            h1 = h1.to(torch.float32)
            h1 = self.conv1(g1, h1, edge_weight=g1.edata['weight'])
            h2 = h2.to(torch.float32)
            h2 = self.conv3(g2, h2, edge_weight=g2.edata['weight'])

        with g1.local_scope():
            with g2.local_scope():
                g1.ndata['h'] = h1

                g2.ndata['h'] = h2
                # Use the average readout to get the graph representation
                hg1 = dgl.mean_nodes(g1, 'h')

                hg2 = dgl.mean_nodes(g2, 'h')

                hg = t.cat((hg1, hg2), -1)

            return self.classify(hg)
