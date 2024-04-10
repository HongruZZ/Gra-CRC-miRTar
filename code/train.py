import torch
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from data.Dataset import *
from models.Classifier import *
from utils.config import *
import datetime

device = t.device("cuda:0")
print(device)


def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True

array=[180, 280, 380, 480, 580]

starttime = datetime.datetime.now()


print("load config")
params = config()
print("load dataset")
dataset = RNAPairDataset(raw_dir='data/train-DG11_re1_CRC.txt', save_dir=f'checkpoints/mmgraph/k{params.k}_d{params.d}')
print("load model")
model = GraphClassifier(in_dim=params.d, hidden_dim=params.hidden_dim, n_classes=params.n_classes, GNN_type=params.GNN_type, device=params.device)
print("load finished")


model.cv_train(dataset, batchSize=params.batchSize,
               num_epochs=params.num_epochs,
               lr=params.lr,
               kFold=params.kFold,
               savePath=params.savePath,
               device=params.device,
               GNN_type=params.GNN_type
               )

endtime = datetime.datetime.now()
print(f'Total running time of all codes is {(endtime - starttime).seconds}s. ')