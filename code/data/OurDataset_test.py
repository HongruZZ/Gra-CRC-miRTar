import os
import pickle
from collections import Counter
import dgl
from dgl.data import DGLDataset, save_graphs, load_graphs
import numpy as np
from dgl.data.utils import save_info, load_info
from dgl.nn.pytorch import EdgeWeightNorm
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from utils.config import *
import itertools
from data.OurDataset_v2 import *
from utils.config import *
import datetime

params = config()
print("start loading training dataset")
dataset = LncRNADataset(raw_dir='data/train-DG11_re1_CRC.txt', save_dir=f'checkpoints/mmgraph/k{params.k}_d{params.d}')
print("finish loading training dataset")

class testDataset(DGLDataset):
    """
        url : str
            The url to download the original dataset.
        raw_dir : str
            Specifies the directory where the downloaded data is stored or where the downloaded data is stored. Default: ~/.dgl/
        save_dir : str
            The directory where the finished dataset will be saved. Default: the value specified by raw_dir
        force_reload : bool
            If or not to re-import the dataset. Default: False
        verbose : bool
            Whether to print progress information.
        """

    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(testDataset, self).__init__(name='lncrna',
                                            url=url,
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose
                                            )
        print('***Executing init function***')
        print('Dataset initialization is completed!\n')

    def process(self):
        # Processing of raw data into plots, labels
        print('***Executing process function***')
        self.kmers = params.k  # 这里k是4
        # Open files and load data
        print('Loading the raw data...')
        with open(self.raw_dir, 'r') as f:
            data = []
            for i in tqdm(f):  # tqdm可以实时反映iteration到哪一步了
                data.append(i.strip('\n').split(','))

        # Get labels and k-mer sentences
        k_miRNA, k_MRNA, rawLab = [[i[2][j:j + self.kmers] for j in range(len(i[2]) - self.kmers + 1)] for i in data], [[i[3][j:j + self.kmers] for j in range(len(i[3]) - self.kmers + 1)] for i in data], \
                                  [i[4] for i in data]
        # K_RNA生成K-mer个RNA, rawlab表示每个LncRNA的label

        # Get the mapping variables for label and label_id
        print('Getting all the dictionary from training dataset')
        self.lab2id = dataset.lab2id
        self.id2lab = dataset.id2lab
        self.kmers2id_mi = dataset.kmers2id_mi
        self.id2kmers_mi = dataset.id2kmers_mi
        self.kmers2id_m = dataset.kmers2id_m
        self.id2kmers_m = dataset.id2kmers_m

        # Get the ids of RNAsequence and label
        self.k_miRNA = k_miRNA
        self.k_MRNA = k_MRNA
        self.labels = t.tensor([self.lab2id[i] for i in rawLab])

        #规定只有这里面的才能有作为kmer看待，only for rna2vec, 更改下面以应对不同的k-mer
        f = ['A', 'C', 'G', 'T']
        c = itertools.product(f, f, f, f, f)
        res = ['<EOS>']
        for i in c:
            temp = i[0] + i[1] + i[2] + i[3] + i[4]
            res.append(temp)
        res = np.array(res)

        self.res = res


        self.idSeq_mi = list()
        for s in self.k_miRNA:
            temp_seq = list()
            for i in s:
                if i not in res:
                    i = '<EOS>'
                temp_seq.append(self.kmers2id_mi[i])
            self.idSeq_mi.append(temp_seq)

        self.idSeq_mi = np.array(self.idSeq_mi, dtype=object)

        self.idSeq_m = list()
        for s in self.k_MRNA:
            temp_seq = list()
            for i in s:
                if i not in res:
                    i = '<EOS>'
                temp_seq.append(self.kmers2id_m[i])
            self.idSeq_m.append(temp_seq)

        self.idSeq_m = np.array(self.idSeq_m, dtype=object)

        #self.vectorize()

        # Construct and save the graph
        self.graph1s = []
        self.graph2s = []
        self.vector = {}


        #这里注意修改
        with open(f'checkpoints/Node_feature/rna2vec_k5_d128_miRNA-MRNA.pkl', 'rb') as f:
            print('Loading rna2vec.....')
            tmp = pickle.load(f)
            self.vector['embedding_mi'], self.vector['embedding_m'] = tmp['embedding_mi'], tmp['embedding_m']

        for i in range(len(self.idSeq_mi)):  # 对于index列中的每一个sequence
            print(i)
            newidSeq_mi = []  # 对于每一个idseq,这个list从0一直往上增加，如果是之前出现过的则会返回到之前kmer对应的顺序index
            newidSeq_m = []
            old2new_mi = {}  # 生成一个字典，里面的元素都是各不相同的
            old2new_m = {}
            count_mi = 0
            count_m = 0

            for eachid in self.idSeq_mi[i]:
                if eachid not in old2new_mi:
                    old2new_mi[eachid] = count_mi
                    count_mi += 1
                newidSeq_mi.append(old2new_mi[eachid])
            counter_uv = Counter(list(zip(newidSeq_mi[:-1], newidSeq_mi[
                                                            1:])))  # 除去最后一个，除去第一个, 这步就是在构造De Brujin Graph了， Counter生成一个后面edge的频数字典
            graph1 = dgl.graph(list(counter_uv.keys()))
            weight = t.FloatTensor(list(counter_uv.values()))
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph1, weight)
            graph1.edata['weight'] = norm_weight
            node_features = self.vector['embedding_mi'][list(old2new_mi.keys())]
            graph1.ndata['attr'] = t.tensor(node_features)

            self.graph1s.append(graph1)

            for eachid in self.idSeq_m[i]:
                if eachid not in old2new_m:
                    old2new_m[eachid] = count_m
                    count_m += 1
                newidSeq_m.append(old2new_m[eachid])
            counter_uv = Counter(list(
                zip(newidSeq_m[:-1], newidSeq_m[1:])))  # 除去最后一个，除去第一个, 这步就是在构造De Brujin Graph了， Counter生成一个后面edge的频数字典
            graph2 = dgl.graph(list(counter_uv.keys()))
            weight = t.FloatTensor(list(counter_uv.values()))
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph2, weight)
            graph2.edata['weight'] = norm_weight
            node_features = self.vector['embedding_m'][list(old2new_m.keys())]
            graph2.ndata['attr'] = t.tensor(node_features)

            self.graph2s.append(graph2)





    def __getitem__(self, idx):
        # Get a sample corresponding to it by idx
        return self.graph1s[idx], self.graph2s[idx], self.labels[idx]

    def __len__(self):
        # Number of data samples
        return len(self.graph1s)

    def save(self):
        # Save the processed data to `self.save_path`
        print('***Executing save function***')
        save_graphs(self.save_dir + "miRNA.bin", self.graph1s, {'labels': self.labels})
        save_graphs(self.save_dir + "MRNA.bin", self.graph2s, {'labels': self.labels})
        # Save additional information in the Python dictionary
        info_path = self.save_dir + "_info.pkl"
        info = {'kmers': self.kmers, 'kmers2id_mi': self.kmers2id_mi, 'id2kmers_mi': self.id2kmers_mi, 'kmers2id_m': self.kmers2id_m, 'id2kmers_m': self.id2kmers_m, 'lab2id': self.lab2id,
                'id2lab': self.id2lab}
        save_info(info_path, info)

    def load(self):
        # Import processed data from `self.save_path`
        print('***Executing load function***')
        self.graph1s, label_dict = load_graphs(self.save_dir + "miRNA.bin")
        self.graph2s, label_dict = load_graphs(self.save_dir + "MRNA.bin")
        self.labels = label_dict['labels']
        info_path = self.save_dir + "_info.pkl"
        info = load_info(info_path)
        self.kmers, self.kmers2id_mi, self.id2kmers_mi, self.kmers2id_m, self.id2kmers_m, self.lab2id, self.id2lab = info['kmers'], info['kmers2id_mi'], info[
            'id2kmers_mi'], info['kmers2id_m'], info['id2kmers_m'], info['lab2id'], info['id2lab']

    def has_cache(self):
        # Check if there is processed data in `self.save_path`
        print('***Executing has_cache function***')
        graph_path_mi = self.save_dir + "miRNA.bin"
        graph_path_m = self.save_dir + "MRNA.bin"
        info_path = self.save_dir + "_info.pkl"
        return os.path.exists(graph_path_mi) and os.path.exists(graph_path_m) and os.path.exists(info_path)


