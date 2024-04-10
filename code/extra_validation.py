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


array = [150, 250, 350, 450, 550]

starttime = datetime.datetime.now()

params = config()

testset = testDataset(raw_dir='data/test-CRC_validation_mirtarbase_201.txt',
                      save_dir=f'data/extra_test_data_201/k{params.k}_d{params.d}')
batchSize = 64
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)


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


class GraphValidClassifier(TestClassifier):
    def __init__(self, in_dim, hidden_dim, n_classes, GNN_type, device=t.device("cpu"), stateDict={}):
        super(GraphValidClassifier, self).__init__()

        if GNN_type == "gcn":
            self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
            self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
            self.conv3 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
            self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
            self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
            self.moduleList_valid = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])

        elif GNN_type == "gat":
            self.conv1 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1,
                                       allow_zero_in_degree=True).to(
                device)
            self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
            self.conv3 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1,
                                       allow_zero_in_degree=True).to(device)
            self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)
            self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
            self.moduleList_valid = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])

        else:
            mlp1 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)
            self.conv1 = dglnn.GINConv(mlp1, 'sum').to(device)
            # using 2-layer mlp
            mlp2 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)
            self.conv3 = dglnn.GINConv(mlp2, 'sum').to(device)
            self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
            self.moduleList_valid = nn.ModuleList([self.conv1, self.conv3, self.classify])

        for idx, module in enumerate(self.moduleList_valid):
            module.load_state_dict(stateDict[idx])
            module.eval()

        self.device = device

    def calculate_y_valid(self, g1, h1, g2, h2, GNN_type):
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

    def valid_train(self, dataset, batchSize=64, device=t.device('cpu'), GNN_type='gcn'):

        test_loader = GraphDataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)

        lossfn = nn.CrossEntropyLoss()

        test_loss, test_rec = self.valid_epoch(test_loader, lossfn, device, GNN_type)

        return test_loss, test_rec

    def valid_epoch(self, dataloader, loss_fn, device, GNN_type):
        with t.no_grad():
            valid_loss, valid_rec = 0.0, 0.0
            pred_list = []
            label_list = []
            for _, (batched_graph_miRNA, batched_graph_MRNA, labels) in enumerate(dataloader):
                # print(batched_graph)
                batched_graph_miRNA, batched_graph_MRNA, labels = batched_graph_miRNA.to(device), batched_graph_MRNA.to(
                    device), labels.to(device)
                feats1 = batched_graph_miRNA.ndata['attr']
                feats2 = batched_graph_MRNA.ndata['attr']

                output = self.calculate_y_valid(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2, GNN_type)

                output, labels = output.to(t.float32), labels.to(t.int64)

                # pred_list loads the predicted values of the training set samples
                pred_list.extend(output.detach().cpu().numpy())

                # label_list loads the true values of the training set samples (one-hot form)
                label_list.extend(labels.cpu().numpy())

            with t.no_grad():
                pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
                valid_loss = loss_fn(pred_tensor, label_tensor)

            pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()
            print(np.array(label_list))
            print(np.argmax(pred_array, axis=1))
            valid_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), pos_label=0)

        return valid_loss, valid_rec


Rec_all = []
mapLocation = {'cuda:0': 'cpu', 'cuda:1': 'cpu'}

GNN_type = params.GNN_type

if GNN_type == "gcn":
    statedic = torch.load(f"checkpoints/Valid_model/fold_GCN.pkl", map_location=mapLocation)
    print('>>>>>>Fold_{}'.format(str("GCN")))
elif GNN_type == "gat":
    statedic = torch.load(f"checkpoints/Valid_model/fold_GAT.pkl", map_location=mapLocation)
    print('>>>>>>Fold_{}'.format(str("GAT")))
else:
    statedic = torch.load(f"checkpoints/Valid_model/fold_GIN.pkl", map_location=mapLocation)
    print('>>>>>>Fold_{}'.format(str("GIN")))

model = GraphValidClassifier(in_dim=params.d, hidden_dim=params.hidden_dim, n_classes=params.n_classes,
                             GNN_type=params.GNN_type, device=params.device, stateDict=statedic)

test_loss, test_rec = model.valid_train(testset, batchSize=params.batchSize, device=params.device,
                                        GNN_type=params.GNN_type)

print(
    ">>>AVG Test Loss:{:.3f}\n"
    "Test Recall:{:.4f}!!!\n".format(
        test_loss, test_rec))

Rec_all.append(test_rec)
Rec_mean = np.mean(np.array(Rec_all))

print('*****All folds are done!')
print("The results of the fold:\n Test Recall:{:.4f} \n".format(Rec_mean))

