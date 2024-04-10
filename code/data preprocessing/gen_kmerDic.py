import numpy as np
import pandas as pd
import pickle
from gensim.models import KeyedVectors
from gensim import models
from gensim.models import word2vec

file = './results/5-mer-mir-mammal-RNA2vec-20230720-0423-k3to3-128d-4c-1527758Mbp-sliding-Piu.w2v'
df = pd.read_csv(file,header=None)

new_df = pd.DataFrame()
embedding_list = []

new_df['name'] = df[0].apply(lambda x:x.split(' ')[0])
new_df['list'] = df[0].apply(lambda x:x.split(' ')[1:])
new_df.index = list(new_df['name'])

key_list = new_df['name'].values



for i in new_df['list']:
    li = [float(j) for j in i]
    embedding_list.append(np.array(li))

embedding_list = np.array(embedding_list)

#embedding dimension
word_embedding_dic = {'<EOS>': np.zeros(128)}
word_embedding_list = []

for i in range(1,len(key_list)):
    if key_list[i] not in word_embedding_list:
        word_embedding_dic[key_list[i]] = embedding_list[i]
        word_embedding_list.append(key_list[i])


with open(f'Datasets/dictionary/RNA2VEC_k5_miRNA_mammal.pkl', 'wb') as f:

    pickle.dump({'word_embedding_dic': word_embedding_dic}, f,
                    protocol=4)


with open(f'Datasets/dictionary/RNA2VEC_k5_miRNA_mammal.pkl', 'rb') as f:
    tmp = pickle.load(f)
    aa = tmp['word_embedding_dic']
    print(len(aa))




