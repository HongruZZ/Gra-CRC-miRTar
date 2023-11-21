import torch

from data.OurDataset_v2 import *
from models.OurClassifier import *
from utils.config import *
from data.OurDataset_test import *
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

testset = testDataset(raw_dir='data/test-DG11_re1_CRC.txt', save_dir=f'data/test_data/k{params.k}_d{params.d}')
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
    def __init__(self, in_dim, hidden_dim, n_classes, device=t.device("cpu"), stateDict = {}):
        super(GraphTestClassifier, self).__init__()


        #####GIN
        '''
        self.lin1 = nn.Linear(in_dim,hidden_dim,bias=True).to(device)
        self.lin3 = nn.Linear(hidden_dim,hidden_dim,bias=True).to(device)

        self.conv1 = dglnn.GINConv(self.lin1,'sum').to(device)
        self.conv2 = dglnn.GINConv(self.lin3,'sum').to(device)

        #using batch-norm
        self.bnorm1 = nn.BatchNorm1d(int(hidden_dim)).to(device)

        #using simple linear
        self.lin2 = nn.Linear(in_dim, hidden_dim, bias=True).to(device)
        self.lin4 = nn.Linear(hidden_dim, hidden_dim, bias=True).to(device)

        self.conv3 = dglnn.GINConv(self.lin2,'sum').to(device)
        self.conv4 = dglnn.GINConv(self.lin4,'sum').to(device)

        # using batch-norm
        self.bnorm3 = nn.BatchNorm1d(int(hidden_dim)).to(device)
        '''
        #####GAT

        #self.conv1 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1, allow_zero_in_degree=True).to(
        #    device)
        #self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
        #
        #self.conv3 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=1,
        #                          allow_zero_in_degree=True).to(device)
        #self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)



        #####GCN
        #self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        # using batch-norm
        #self.bnorm1 = nn.BatchNorm1d(hidden_dim).to(device)
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        # self.bnorm2 = nn.BatchNorm1d(hidden_dim).to(device)
        #self.conv3 = dglnn.GraphConv(in_dim, hidden_dim, norm='none', allow_zero_in_degree=True).to(device)
        # using batch-norm
        #self.bnorm3 = nn.BatchNorm1d(hidden_dim).to(device)


        ###GIN MLP2
        mlp1 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)

        self.conv1 = dglnn.GINConv(mlp1, 'sum').to(device)

        # using 2-layer mlp
        mlp2 = MLP_layer(in_dim, hidden_dim, hidden_dim).to(device)

        self.conv3 = dglnn.GINConv(mlp2, 'sum').to(device)

        self.classify = MLP(inSize=hidden_dim*2, outSize=n_classes).to(device)
        #For GAT
        #self.classify = MLP(inSize=hidden_dim * 2, outSize=n_classes).to(device)
        #for GIN
        #self.moduleList = nn.ModuleList([self.conv1,self.bnorm1 ,self.bnorm3, self.conv3, self.classify])

        #For simple linear GIN
        #self.moduleList_test = nn.ModuleList(
        #    [self.lin1, self.lin2, self.lin3, self.lin4, self.conv1, self.conv2, self.conv3, self.conv4, self.bnorm1,
        #     self.bnorm3, self.classify])

        #For GAT
        #self.moduleList_test = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])

        #For GCN
        #self.moduleList_test = nn.ModuleList([self.conv1, self.bnorm1, self.conv3, self.bnorm3, self.classify])

        #For Gin MLP
        self.moduleList_test = nn.ModuleList([self.conv1, self.conv3, self.classify])

        #self.moduleList_test = nn.ModuleList([self.lin1, self.lin2,self.conv1, self.conv3, self.bnorm1, self.bnorm3, self.classify])

        for idx, module in enumerate(self.moduleList_test):
            module.load_state_dict(stateDict[idx])
            module.eval()

        self.device = device

    def calculate_y_test(self, g1, h1, g2, h2):  # g是batched graph data. h是node_features
        # Apply graph convolution networks and activation functions
        #original
        #h1 = F.relu(self.conv1(g1, h1, edge_weight=g1.edata['weight']))  # e.g.torch.Size([142, 64])

        h1 = self.conv1(g1, h1, edge_weight=g1.edata['weight'])

        #print(h1.shape)

        #add one more layer
        #h1 = self.conv2(g1, h1, edge_weight=g1.edata['weight'])

        #h1 = self.conv5(g1, h1, edge_weight=g1.edata['weight'])

        #h1 = self.bnorm1(h1)
        #for multi-head
        #h1 = self.bnorm1(t.reshape(h1, (h1.shape[0], h1.shape[1] * h1.shape[2])))

        #h1 = F.relu(h1)

        #original
        #h2 = F.relu(self.conv3(g2, h2, edge_weight=g2.edata['weight']))

        #adding batchnorm
        h2 = self.conv3(g2, h2, edge_weight=g2.edata['weight'])

        #add one more layer
        #h2 = self.conv4(g2, h2, edge_weight=g2.edata['weight'])


        #h2 = self.bnorm3(h2)
        #h2 = self.bnorm3(t.reshape(h2, (h2.shape[0], h2.shape[1] * h2.shape[2])))
        #h2 = F.relu(h2)


        with g1.local_scope():
            with g2.local_scope():
                # 这行式子的意思是后来对图的变化不会对original的图产生影响
                #print(h1.shape)
                g1.ndata['h'] = h1

                g2.ndata['h'] = h2
                # Use the average readout to get the graph representation

                hg1 = dgl.mean_nodes(g1, 'h')


                #hg1_num = g1.batch_num_nodes().unsqueeze(1)
                #hg1_pe = PositionalEncoding(hg1_num,16, hg1_num.shape[0], self.device).squeeze(1)
                #hg1 = torch.cat((hg1, hg1_pe), 1)


                hg2 = dgl.mean_nodes(g2, 'h')

                #hg2_num = g2.batch_num_nodes().unsqueeze(1)
                #hg2_pe = PositionalEncoding(hg2_num,16, hg2_num.shape[0], self.device).squeeze(1)

                #hg2 = torch.cat((hg2, hg2_pe), 1)


                hg = t.cat((hg1,hg2), -1)


                # print(hg.shape) #e.g. torch.Size([8, 64])，8这里对应batch_size
            return self.classify(hg)

    def calculate_y_test_hg(self, g1, h1, g2, h2):  # g是batched graph data. h是node_features
        # Apply graph convolution networks and activation functions
        # original

        h1 = self.conv1(g1, h1, edge_weight=g1.edata['weight'])

        h2 = self.conv3(g2, h2, edge_weight=g2.edata['weight'])


        with g1.local_scope():
            with g2.local_scope():
                # 这行式子的意思是后来对图的变化不会对original的图产生影响
                # print(h1.shape)
                g1.ndata['h'] = h1

                g2.ndata['h'] = h2
                # Use the average readout to get the graph representation

                hg1 = dgl.mean_nodes(g1, 'h')

                hg2 = dgl.mean_nodes(g2, 'h')

                hg = t.cat((hg1, hg2), -1)

                # print(hg.shape) #e.g. torch.Size([8, 64])，8这里对应batch_size
            return hg

    def test_train(self, dataset, batchSize=64, savePath='checkpoints/Final_model_1',device=t.device('cpu')):
        if not os.path.exists(savePath):  # 如果savePath不存在就生成一个
            os.mkdir(savePath)

        test_loader = GraphDataLoader(dataset, batch_size=batchSize,shuffle=False, num_workers=0)

        lossfn = nn.CrossEntropyLoss()

        test_loss, test_acc, test_f, test_pre, test_rec, test_sp, test_roc, test_aupr, y_true, y_pred = self.valid_epoch(test_loader, lossfn, device)

        return test_loss, test_acc, test_f, test_pre, test_rec, test_sp, test_roc, test_aupr, y_true, y_pred


    def valid_epoch(self, dataloader, loss_fn, device):
        # self.to_eval_mode()
        with t.no_grad():
            valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            #self.to_eval_mode()
            pred_list = []
            label_list = []
            graph_list = torch.zeros([1,256]).to(device)
            for _, (batched_graph_miRNA, batched_graph_MRNA, labels) in enumerate(dataloader):
                # print(batched_graph)
                batched_graph_miRNA, batched_graph_MRNA, labels = batched_graph_miRNA.to(device), batched_graph_MRNA.to(device), labels.to(device)
                feats1 = batched_graph_miRNA.ndata['attr']
                feats2 = batched_graph_MRNA.ndata['attr']

                output = self.calculate_y_test(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2)
                graphs = self.calculate_y_test_hg(batched_graph_miRNA, feats1, batched_graph_MRNA, feats2)
                graph_list = torch.vstack((graph_list,graphs))

                output, labels = output.to(t.float32), labels.to(t.int64)

                # pred_list loads the predicted values of the training set samples
                pred_list.extend(output.detach().cpu().numpy())

                # label_list loads the true values of the training set samples (one-hot form)
                label_list.extend(labels.cpu().numpy())
            real_graphs = graph_list[1:, :].cpu().numpy()
            print(real_graphs.shape)
            np.save("after_GNN_embedding", real_graphs)

            with t.no_grad():
                pred_tensor, label_tensor = t.tensor(np.array(pred_list)), t.tensor(np.array(label_list))
                valid_loss = loss_fn(pred_tensor, label_tensor)
            pred_array = F.softmax(t.tensor(np.array(pred_list)), dim=1).cpu().numpy()

            #argmax to argmin
            print(label_list)
            print(pred_array)
            valid_acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
            #valid_acc = accuracy_score(F.one_hot(t.tensor(label_list), num_classes=params.n_classes).cpu().numpy(),
            #                          pred_array)
            valid_f = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), average='micro')
            valid_pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), average='micro',
                                        zero_division=0)
            valid_rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), average='micro')
            tn, fp, fn, tp = confusion_matrix(np.array(label_list), np.argmax(pred_array, axis=1)).ravel()
            valid_sp = tn / (tn + fp)
            valid_roc = roc_auc_score(F.one_hot(t.tensor(label_list), num_classes=params.n_classes).cpu().numpy(),
                                      pred_array, average='micro')
            #valid_aupr = average_precision_score(np.array(label_list), np.argmax(pred_array, axis=1), average='micro')
            valid_aupr = average_precision_score(F.one_hot(t.tensor(label_list), num_classes=params.n_classes).cpu().numpy(),
                                      pred_array, average='micro')

            y_true = list(np.array(np.array(label_list)))

            #y_pred = list(np.array(np.argmax(pred_array, axis=1)))
            y_pred = list(pred_array[:,1])
        return valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_sp, valid_roc, valid_aupr, y_true, y_pred

    def to_train_mode(self):
        for module in self.moduleList:
            module.train()

    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()


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

Acc_all = []
F_all = []
Pre_all = []
Rec_all = []
Sp_all = []
Roc_all = []
Aupr_all = []
y_true_all = []
y_pred_all = []


for i in range(1,2):
    mapLocation = {'cuda:0': 'cpu', 'cuda:1': 'cpu'}
    #statedic = torch.load(f"checkpoints/Final_model_1/fold{i}.pkl", map_location=mapLocation)
    statedic = torch.load(f"checkpoints/Best_model/fold{i}.pkl", map_location=mapLocation)

    print('>>>>>>Fold{}'.format(i))

    model = GraphTestClassifier(in_dim=params.d, hidden_dim=params.hidden_dim, n_classes=params.n_classes, device=params.device, stateDict=statedic)

    test_loss, test_acc, test_f, test_pre, test_rec, test_sp, test_roc, test_aupr, y_true, y_pred = model.test_train(testset, batchSize=params.batchSize,savePath=params.savePath,device=params.device)

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
print("For 5 fold results:\n Test Acc:{:.4f} %\n Test F1-score:{:.4f} \n Test Precision:{:.4f} \n Test Recall:{:.4f}\n Test Specificity:{:.4f}\n Test ROC:{:.4f}\n Test AUPR:{:.4f}\n".format(
                Acc_mean * 100, F_mean, Pre_mean, Rec_mean, Sp_mean, Roc_mean, Aupr_mean))

print("*****Drawing the ROC curve")
y_test = np.array(y_true_all)
y_score = np.array(y_pred_all)
print(y_test)
print(y_test.shape)
print(y_score)
print(y_score.shape)

'''
np.save('./GAT-test.npy', y_test)
np.save('./GAT-pred.npy', y_score)'''
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
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