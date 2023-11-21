# CRCGraph
The pre-training nucleotide-to-graph neural networks to identify potential miRNA-mRNA interactions in colorectal cancer patients

# Requirements
    python==3.8
    torch==1.12.0+cu113
    dgl==1.1.1+cu113
    scikit-learn==1.0.1
    numpy==1.24.4
    gensim==4.1.2
    tqdm==4.62.3
    pandas==2.0.2
    scipy==1.10.1

# Usage

## Before running the codes
#### Please download the [supplement materials](https://drive.google.com/drive/folders/1caGodK_1220YXQfKSLjHIvi87VBRau97?usp=drive_link) for the codes.
Place all of the files in `pairfeatures` folder to `/checkpoints/Pair_feature`.

Place all of the files in `pre-train-dataset/mirna_mammal` to `data preprocessing/New_data/mirna_mammal`.

Place all of the files in `pre-train-dataset/mrna_mammal` to `data preprocessing/New_data/mrna_mammal`.

Place all of the files in `train-dataset` to `checkpoints\mmgraph`.

Place all of the files in `test-dataset` to `data\test_data`.

## Simple usage
You can regenerate our model's results using the commands below:
```
python trial_train.py
```


