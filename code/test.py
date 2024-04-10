import torch

from data.Dataset import *
from models.Classifier import *
from utils.config import *
from data.Dataset_test import *
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score

device = t.device("cuda:0")


def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True

array=[150, 250, 350, 450, 550]

starttime = datetime.datetime.now()

params = config()

testset = testDataset(raw_dir='data/test-DG11_re1_CRC.txt', save_dir=f'data/test_data/rna2vec/k{params.k}_d{params.d}')
batchSize = 64
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)
after_graphs = torch.zeros([1,256])

class TestClassifier:
    def __init__(self):
        pass


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


class GraphTestClassifier(TestClassifier):
    def __init__(self, in_dim, hidden_dim, n_classes, GNN_type, device=t.device("cpu"), stateDict = {}):
        super(GraphTestClassifier, self).__init__()

        if GNN_type == "gcn":
            self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
            self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
            self.conv3 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
            self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
            self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
            self.moduleList_test = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])

        elif GNN_type == "gat":
            self.conv1 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1, allow_zero_in_degree=True).to(
                device)
            self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
            self.conv3 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1,
                                      allow_zero_in_degree=True).to(device)
            self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
            self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
            self.moduleList_test = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])

        else:
            mlp1 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)
            self.conv1 = dglnn.GINConv(mlp1, 'sum').to(device)
            # using 2-layer mlp
            mlp2 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)
            self.conv3 = dglnn.GINConv(mlp2, 'sum').to(device)
            self.classify = MLP(inSize=hidden_dim*2, outSize=n_classes).to(device)
            self.moduleList_test = nn.ModuleList([self.conv1, self.conv3, self.classify])

        for idx, module in enumerate(self.moduleList_test):
            module.load_state_dict(stateDict[idx])
            module.eval()

        self.device = device

    def calculate_y_test(self, g1, h1, g2, h2, GNN_type):
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

                hg = t.cat((hg1,hg2), -1)

            return self.classify(hg)

    def calculate_y_test_hg(self, g1, h1, g2, h2, GNN_type):
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

            return hg

    def test_train(self, dataset, batchSize=64 ,device=t.device('cpu'), GNN_type = 'gcn'):

        test_loader = GraphDataLoader(dataset, batch_size=batchSize,shuffle=False, num_workers=0)

        lossfn = nn.CrossEntropyLoss()

        test_loss, test_acc, test_f, test_pre, test_rec, test_sp, test_roc, test_aupr, y_true, y_pred = self.valid_epoch(test_loader, lossfn, device, GNN_type)

        return test_loss, test_acc, test_f, test_pre, test_rec, test_sp, test_roc, test_aupr, y_true, y_pred


    def valid_epoch(self, dataloader, loss_fn, device, GNN_type):
        with t.no_grad():
            valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            pred_list = []
            label_list = []
            graph_list = torch.zeros([1,256]).to(device)
            for _, (batched_graph_miRNA, batched_graph_MRNA, labels) in enumerate(dataloader):
                # print(batched_graph)
                batched_graph_miRNA, batched_graph_MRNA, labels = batched_graph_miRNA.to(device), batched_graph_MRNA.to(device), labels.to(device)
                feats1 = batched_graph_miRNA.ndata['attr']
                feats2 = batched_graph_MRNA.ndata['attr']

                output = self.calculate_y_test(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2, GNN_type)
                graphs = self.calculate_y_test_hg(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2, GNN_type)
                graph_list = torch.vstack((graph_list,graphs))
                output, labels = output.to(t.float32), labels.to(t.int64)

                # pred_list loads the predicted values of the training set samples
                pred_list.extend(output.detach().cpu().numpy())

                # label_list loads the true values of the training set samples (one-hot form)
                label_list.extend(labels.cpu().numpy())
            real_graphs = graph_list[1:, :].cpu().numpy()
            print(real_graphs.shape)
            #np.save("./visualization/embeddings/before_GNN_embedding_3mer_fold2", real_graphs)

            with t.no_grad():
                pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
                valid_loss = loss_fn(pred_tensor, label_tensor)
            pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()

            # print(np.array(label_list))
            # print(np.argmax(pred_array, axis=1))
            tp, fn, fp, tn = confusion_matrix(np.array(label_list), np.argmax(pred_array, axis=1)).ravel()

            valid_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
            valid_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)
            valid_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1),  pos_label=0)
            valid_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)
            valid_sp = tn / (tn + fp)

            valid_roc = roc_auc_score(np.array(label_list), pred_array[:,1])
            valid_aupr = average_precision_score(np.array(label_list), pred_array[:,1])

            y_true = list(np.array(np.array(label_list)))
            y_pred = list(pred_array[:,1])
        return valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_sp, valid_roc, valid_aupr, y_true, y_pred


Acc_all = []
F_all = []
Pre_all = []
Rec_all = []
Sp_all = []
Roc_all = []
Aupr_all = []
y_true_all = []
y_pred_all = []


for i in range(1,6):
    mapLocation = {'cuda:0': 'cpu', 'cuda:1': 'cpu'}
    gnn = params.GNN_type
    if gnn == "gcn":
        statedic = torch.load(f"checkpoints/Best_model/gcn/{params.k}mer/fold{i}.pkl", map_location=mapLocation)
    elif gnn == 'gat':
        statedic = torch.load(f"checkpoints/Best_model/gat/{params.k}mer/fold{i}.pkl", map_location=mapLocation)
    else:
        statedic = torch.load(f"checkpoints/Best_model/gin/{params.k}mer/fold{i}.pkl", map_location=mapLocation)

    #statedic = torch.load(f"checkpoints/Best_model/fold{i}.pkl", map_location=mapLocation)

    print('>>>>>>Fold{}'.format(i))

    model = GraphTestClassifier(in_dim=params.d, hidden_dim=params.hidden_dim, n_classes=params.n_classes, GNN_type= params.GNN_type, device=params.device, stateDict=statedic)

    test_loss, test_acc, test_f, test_pre, test_rec, test_sp, test_roc, test_aupr, y_true, y_pred = model.test_train(testset, batchSize=params.batchSize, GNN_type=params.GNN_type, device=params.device)

    print(
        ">>>AVG Test Loss:{:.3f}\n"
        "Test Acc:{:.4f} %, Test F1-score:{:.4f}, Test Precision:{:.4f}, Test Recall:{:.4f}, Test Specificity:{:.4f}, Test ROC:{:.4f}, Test AUPR:{:.4f}!!!\n".format(
            test_loss,
            test_acc * 100, test_f, test_pre, test_rec, test_sp, test_roc, test_aupr))

    Acc_all.append(test_acc)
    F_all.append(test_f)
    Pre_all.append(test_pre)
    Rec_all.append(test_rec)
    Sp_all.append(test_sp)
    Roc_all.append(test_roc)
    Aupr_all.append(test_aupr)
    y_true_all=y_true_all+y_true
    y_pred_all=y_pred_all+y_pred

Acc_mean = np.mean(np.array(Acc_all))
F_mean = np.mean(np.array(F_all))
Pre_mean = np.mean(np.array(Pre_all))
Rec_mean = np.mean(np.array(Rec_all))
Sp_mean = np.mean(np.array(Sp_all))
Roc_mean = np.mean(np.array(Roc_all))
Aupr_mean = np.mean(np.array(Aupr_all))

print('*****All folds are done!')
print("For 5 fold results:\n Test Acc:{:.4f} %\n Test F1-score:{:.4f} \n Test Precision:{:.4f} \n Test Recall:{:.4f}\n Test Specificity:{:.4f}\n Test ROC:{:.4f}\n Test AUPR:{:.10f}\n".format(
                Acc_mean * 100, F_mean, Pre_mean, Rec_mean, Sp_mean, Roc_mean, Aupr_mean))

print("*****Drawing the ROC curve")
y_test = np.array(y_true_all)
y_score = np.array(y_pred_all)
print(y_test)
print(y_test.shape)
print(y_score)
print(y_score.shape)


# np.save('./GCN-test.npy', y_test)
# np.save('./GCN-pred.npy', y_score)
fpr,tpr,threshold = roc_curve(y_test, y_score)
roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

endtime = datetime.datetime.now()
print(f'Total running time of all codes is {(endtime - starttime).seconds}s. ')