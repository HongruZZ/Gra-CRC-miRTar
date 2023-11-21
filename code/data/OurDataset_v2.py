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

params = config()


class LncRNADataset(DGLDataset):
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
        super(LncRNADataset, self).__init__(name='lncrna',
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
        print('Getting the mapping variables for label and label id...')
        self.lab2id, self.id2lab = {}, []
        cnt = 0
        for lab in tqdm(rawLab):  # 这一步生成了lab2id的字典，label的集合以及class的个数
            if lab not in self.lab2id:
                self.lab2id[lab] = cnt
                self.id2lab.append(lab)
                cnt += 1
        self.classNum = cnt

        # Get the mapping variables for kmers and kmers_id (to both mirna and mrna)
        print('Getting the mapping variables for kmers and kmers id...')
        self.kmers2id_mi, self.id2kmers_mi = {"<EOS>": 0}, ["<EOS>"]

        f = ['A', 'C', 'G', 'T']
        c = itertools.product(f, f, f, f, f)
        res = []
        for i in c:
            temp = i[0] + i[1] + i[2]+ i[3] + i[4]
            res.append(temp)
        res = np.array(res)

        kmersCnt_mi = 1
        for i in res:
            self.kmers2id_mi[i] = kmersCnt_mi
            self.id2kmers_mi.append(i)
            kmersCnt_mi += 1
        self.kmersNum_mi = kmersCnt_mi

        self.kmers2id_m, self.id2kmers_m = self.kmers2id_mi, self.id2kmers_mi
        self.kmersNum_m = kmersCnt_mi

        # Get the ids of RNAsequence and label
        self.k_miRNA = k_miRNA
        self.k_MRNA = k_MRNA
        self.labels = t.tensor([self.lab2id[i] for i in rawLab])

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

        #original codes
        #self.idSeq_mi = np.array([[self.kmers2id_mi[i] for i in s] for s in self.k_miRNA],
        #                      dtype=object)  # 用index来表示Labels和idSeq
        #self.idSeq_m = np.array([[self.kmers2id_m[i] for i in s] for s in self.k_MRNA],
        #                         dtype=object)  # 用index来表示Labels和idSeq

        self.vectorize()

        # Construct and save the graph
        self.graph1s = []
        self.graph2s = []


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
            print(list(old2new_mi.keys()))
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
            print(list(old2new_m.keys()))
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

    def vectorize(self, method="rna2vec", feaSize=params.d, window=5, sg=1,
                  workers=8, loadCache=True):
        self.vector = {}
        print('\n***Executing vectorize function***')
        if os.path.exists(
                f'checkpoints/Pair_feature/{method}_k{self.kmers}_d{feaSize}_miRNA-MRNA.pkl') and loadCache:  # 这一步是看是否存在pkl文件，有的话就直接读取
            with open(f'checkpoints/Pair_feature/{method}_k{self.kmers}_d{feaSize}_miRNA-MRNA.pkl', 'rb') as f:
                if method == 'kmers':
                    tmp = pickle.load(f)
                    self.vector['encoder'], self.kmersFea = tmp['encoder'], tmp['kmersFea']
                elif method == 'word2vec':
                    print('Loading word2vec....')
                    tmp = pickle.load(f)
                    self.vector['embedding_mi'], self.vector['embedding_m'] = tmp['embedding_mi'], tmp['embedding_m']
                else:
                    print('Loading rna2vec.....')
                    tmp = pickle.load(f)
                    self.vector['embedding_mi'], self.vector['embedding_m'] = tmp['embedding_mi'], tmp['embedding_m']

            print(f'Load cache from checkpoints/Pair_feature/{method}_k{self.kmers}_d{feaSize}_miRNA-MRNA.pkl!')
            return
        if method == 'word2vec':
            print('Using Word2vec.....')
            doc_mi = [i + ['<EOS>'] for i in self.k_miRNA]
            model_mi = Word2Vec(doc_mi, min_count=0, window=window, vector_size=feaSize, workers=workers, sg=sg, seed=10)
            word2vec_mi = np.zeros((self.kmersNum_mi, feaSize), dtype=np.float32)
            for i in range(self.kmersNum_mi):
                word2vec_mi[i] = model_mi.wv[self.id2kmers_mi[i]]  # 生成Word2vec向量矩阵
            self.vector['embedding_mi'] = word2vec_mi

            doc_m = [i + ['<EOS>'] for i in self.k_MRNA]
            model_m = Word2Vec(doc_m, min_count=0, window=window, vector_size=feaSize, workers=workers, sg=sg, seed=10)
            word2vec_m = np.zeros((self.kmersNum_m, feaSize), dtype=np.float32)
            for i in range(self.kmersNum_m):
                word2vec_m[i] = model_m.wv[self.id2kmers_m[i]]  # 生成Word2vec向量矩阵
            self.vector['embedding_m'] = word2vec_m

        elif method == 'kmers':
            enc = OneHotEncoder(categories='auto')
            enc.fit([[i] for i in self.kmers2id.values()])
            feaSize = len(self.kmers2id)
            kmers = np.zeros((len(self.labels), feaSize))
            bs = 50000
            print('Getting the kmers vector...')
            for i, t in enumerate(self.idSeq):
                for j in range((len(t) + bs - 1) // bs):
                    kmers[i] += enc.transform(np.array(t[j * bs:(j + 1) * bs]).reshape(-1, 1)).toarray().sum(
                        axis=0)
            kmers = kmers[:, 1:]
            feaSize -= 1
            # Normalized
            kmers = (kmers - kmers.mean(axis=0)) / kmers.std(axis=0)
            self.vector['encoder'] = enc
            self.kmersFea = kmers

        else:
            print('Using RNA2vec......')
            with open(f'checkpoints/Pair_feature/RNA2VEC_k{self.kmers}_miRNA_mammal.pkl', 'rb') as f:
                tmp = pickle.load(f)
                word_dict_mi = tmp['word_embedding_dic']
                rna2vec_mi = np.zeros((self.kmersNum_mi, 128), dtype=np.float32)
                #print(word_dict_mi)
                for i in range(len(self.id2kmers_mi)):
                    #print(self.kmersNum_mi)
                    if self.id2kmers_mi[i] not in word_dict_mi:
                        rna2vec_mi[i] = word_dict_mi[list(word_dict_mi.keys())[0]]
                    else:
                        rna2vec_mi[i] = word_dict_mi[self.id2kmers_mi[i]]

            with open(f'checkpoints/Pair_feature/RNA2VEC_k{self.kmers}_MRNA_mammal.pkl', 'rb') as f:
                tmp = pickle.load(f)
                word_dict_m = tmp['word_embedding_dic']
                rna2vec_m = np.zeros((self.kmersNum_m, 128), dtype=np.float32) #embedding dimension
                for i in range(len(self.id2kmers_mi)):
                    #print(self.id2kmers_m[i])
                    #print(i)
                    #print(self.id2kmers_m)
                    rna2vec_m[i] = word_dict_m[self.id2kmers_m[i]]

            self.vector['embedding_mi'] = rna2vec_mi
            self.vector['embedding_m'] = rna2vec_m


        # Save k-mer vectors
        with open(f'checkpoints/Pair_feature/{method}_k{self.kmers}_d{feaSize}_miRNA-MRNA.pkl', 'wb') as f:
            if method == 'kmers':
                pickle.dump({'encoder': self.vector['encoder'], 'kmersFea': self.kmersFea}, f, protocol=4)
            elif method == 'word2vec':
                pickle.dump({'embedding_mi': self.vector['embedding_mi'],'embedding_m': self.vector['embedding_m']}, f, protocol=4)
            else:
                pickle.dump({'embedding_mi': self.vector['embedding_mi'], 'embedding_m': self.vector['embedding_m']}, f,
                            protocol=4)

