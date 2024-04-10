# Gra-CRC-miRTar
The pre-training nucleotide-to-graph neural networks to identify potential miRNA-mRNA interactions in colorectal cancer patients

[Diagram](code/visualization/Workflow Diagram.jpg)

# Requirements
The codes are tested in Python 3.8.16 and you can install all of the required packages by running the following commands:
```
pip install -r requirements.txt
```

# Usage

## Before running the codes
#### Please download the [supplemental materials](https://drive.google.com/drive/folders/1caGodK_1220YXQfKSLjHIvi87VBRau97?usp=drive_link) for the codes.
Place all of the files in `pairfeatures` folder to `/checkpoints/Pair_feature`.

Place all of the files in `/Gra-CRC-miRTar_supplemental_material/pre_train_dataset/mirna_mammal` to `/code/data preprocessing/New_data/mirna_mammal`.

Place all of the files in `/Gra-CRC-miRTar_supplemental_material/pre_train_dataset/mrna_mammal` to `/code/data preprocessing/New_data/mrna_mammal`.

Place all of the files in `/Gra-CRC-miRTar_supplemental_material/train_dataset` to `/code/checkpoints/mmgraph`.

Place all of the files in `/Gra-CRC-miRTar_supplemental_material/test_dataset` to `/code/data/test_data/rna2vec`.

Place all of the files in `/Gra-CRC-miRTar_supplemental_material/extra_validation_dataset` to `/code/data/extra_test_data_201`.


## Simple usage
#### You can regenerate our model's results using the commands below.

For training:
```
python train.py
```
For testing:
```
python test.py
```
For extra validation:
```
python extra_validation.py
```


