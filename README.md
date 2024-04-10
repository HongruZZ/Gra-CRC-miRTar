# Gra-CRC-miRTar
The pre-training nucleotide-to-graph neural networks to identify potential miRNA-mRNA interactions in colorectal cancer patients

# Requirements
The codes are tested in Python 3.8.16 and you can install all of the required packages by running the following codes:
```
pip install -r requirements.txt
```

# Usage

## Before running the codes
#### Please download the [supplement materials](https://drive.google.com/drive/folders/1caGodK_1220YXQfKSLjHIvi87VBRau97?usp=drive_link) for the codes.
Place all of the files in `pairfeatures` folder to `/checkpoints/Pair_feature`.

Place all of the files in `pre-train-dataset/mirna_mammal` to `data preprocessing/New_data/mirna_mammal`.

Place all of the files in `pre-train-dataset/mrna_mammal` to `data preprocessing/New_data/mrna_mammal`.

Place all of the files in `train-dataset` to `checkpoints\mmgraph`.

Place all of the files in `test-dataset` to `data\test_data`.

## Simple usage
#### You can regenerate our model's results using the commands below.

For training:
```
python trial_train.py
```
For testing:
```
python trial_data_3.py
```


