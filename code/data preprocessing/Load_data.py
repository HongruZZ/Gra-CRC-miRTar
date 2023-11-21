import pandas as pd
import numpy as np
import re
#Load CSV
#csvpath = ('C:\\UF research\\Code implementation\\Data\\293T_SRR18281057_collapsed_hybrids_dG11.csv')
csvpath = ('C:\\UF research\\Code implementation\\Data\\EXP_314_dG11.csv')

#Output TXT
#txtpath = ('C:\\UF research\\Code implementation\\preMLI-main\\Datasets\\Our Dataset\\MiRNA')
txtpath = ('C:\\UF research\\Code implementation\\preMLI-main\\Datasets\\Our_Dataset')

#Name the file
#txtname = ('Big_DG11_MiRNA')
#txtname = ('Small_DG11_MiRNA')
#txtname = ('Big_DG11_MRNA')
txtname = ('DG11_true')


realcsvpath = csvpath
realtxtpath = txtpath+'\\'+txtname+'.txt'
traintxtpath = txtpath+'\\'+'train-'+txtname+'.txt'
testtxtpath = txtpath+'\\'+'test-'+txtname+'.txt'
print(realtxtpath)
data = pd.read_csv(realcsvpath)
print(data.head(5))

#Delete some incorrect rows

drop_row_index = []
for i, j in data.iterrows():
    if re.search(r"\W",j['element_sequence']) != None:
        drop_row_index.append(i)
data1 = data.drop(drop_row_index)



#Change the column loading
#loadyouwant = ['miRNA_name', 'miRNA_sequence']
loadyouwant = ['miRNA_name', 'Gene_ID', 'miRNA_sequence', 'element_sequence','dG']
loadyoudontwant = ['miRNA_name', 'Gene_ID', 'miRNA_sequence', 'negative_element_1X', 'dG']

print(loadyouwant)
print(loadyoudontwant)
youwant=loadyouwant
youdontwant=loadyoudontwant
print(youwant[0])
print(youdontwant[0])

all_id = np.arange(len(data1[youwant[0]]))

test_1 = np.arange(2500)
test_2 = np.arange(len(data1[youwant[0]])-2500,len(data1[youwant[0]]))
test_id = np.concatenate((test_1, test_2), axis=None)


train_id = np.setdiff1d(all_id,test_id)
print(train_id)
print(all_id.shape)
print(train_id.shape)
print(test_id.shape)

##Generating Training Dataset

txt1 = open(traintxtpath, 'w')

#Here we set the threshold to be -16.5

for i in train_id:
    for name in youwant:
        if name == youwant[len(youwant)-1]:
            txt1.write(str(1) + '\n')

        else:
            txt1.write(str(data1.iloc[i][name]) + ',')
    for name in youdontwant:
        if name == youwant[len(youwant)-1]:
            txt1.write(str(0) + '\n')

        else:
            txt1.write(str(data1.iloc[i][name]) + ',')
txt1.close()

##Generating Testing Dataset

txt2 = open(testtxtpath, 'w')

#Here we set the threshold to be -16.5

for i in test_id:
    for name in youwant:
        if name == youwant[len(youwant)-1]:
            txt2.write(str(1) + '\n')

        else:
            txt2.write(str(data1.iloc[i][name]) + ',')
    for name in youdontwant:
        if name == youwant[len(youwant)-1]:
            txt2.write(str(0) + '\n')

        else:
            txt2.write(str(data1.iloc[i][name]) + ',')
txt2.close()

#To check the maximum length of negative_element 1.5X
max_length_1 = 10000
min_seq = str(0)
min_index = 0
for i in all_id:
    if len(data1.iloc[i]['element_sequence']) < max_length_1:
        max_length_1 = len(data1.iloc[i]['element_sequence'])
        min_seq = data1.iloc[i]['element_sequence']
        min_index = i
print("The minimum length of MRNA 1x is", max_length_1)
print(min_seq)
print(min_index)

max_length_0 = 10000
for i in all_id:
    if len(data1.iloc[i]['miRNA_sequence']) < max_length_0:
        max_length_0 = len(data1.iloc[i]['miRNA_sequence'])
print("The minimum length of MiRNA is", max_length_0)


