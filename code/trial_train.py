import torch
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from data.OurDataset_v2 import *
from models.OurClassifier import *
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

params = config()




dataset = LncRNADataset(raw_dir='data/train-DG11_re1_CRC.txt', save_dir=f'checkpoints/mmgraph/k{params.k}_d{params.d}') #

model = GraphClassifier(in_dim=params.d, hidden_dim=params.hidden_dim, n_classes=params.n_classes, device=params.device)

model.cv_train(dataset, batchSize=params.batchSize,
               num_epochs=params.num_epochs,
               lr=params.lr,
               kFold=params.kFold,
               savePath=params.savePath,
               device=params.device
               )

endtime = datetime.datetime.now()
print(f'Total running time of all codes is {(endtime - starttime).seconds}s. ')