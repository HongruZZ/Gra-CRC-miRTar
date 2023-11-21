import numpy as np
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve

from sklearn import metrics
import pylab as plt


def ks(y_predicted1, y_true1, y_predicted2, y_true2, y_predicted3, y_true3, y_predicted4, y_true4 , y_predicted5, y_true5, y_predicted6, y_true6, y_predicted7, y_true7, y_predicted8, y_true8):#
    Font = {'size': 12, 'family': 'Times New Roman'}
    Font_title = {'size': 18, 'family': 'Times New Roman'}

    label1 = y_true1
    label2 = y_true2
    label3 = y_true3
    label4 = y_true4
    label5 = y_true5
    label6 = y_true6
    label7 = y_true7
    label8 = y_true8

    fpr1, tpr1, thres1 = metrics.roc_curve(label1, y_predicted1)
    fpr2, tpr2, thres2 = metrics.roc_curve(label2, y_predicted2)
    fpr3, tpr3, thres3 = metrics.roc_curve(label3, y_predicted3)
    fpr4, tpr4, thres4 = metrics.roc_curve(label4, y_predicted4)
    fpr5, tpr5, thres5 = metrics.roc_curve(label5, y_predicted5)
    fpr6, tpr6, thres6 = metrics.roc_curve(label6, y_predicted6)
    fpr7, tpr7, thres7 = metrics.roc_curve(label7, y_predicted7)
    fpr8, tpr8, thres8 = metrics.roc_curve(label8, y_predicted8)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    roc_auc3 = metrics.auc(fpr3, tpr3)
    roc_auc4 = metrics.auc(fpr4, tpr4)
    roc_auc5 = metrics.auc(fpr5, tpr5)
    roc_auc6 = metrics.auc(fpr6, tpr6)
    roc_auc7 = metrics.auc(fpr7, tpr7)
    roc_auc8 = metrics.auc(fpr8, tpr8)

    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, 'b', label='GIN (AUC = %0.3f) ' % roc_auc1, color='Red', linewidth=1.5)
    plt.plot(fpr7, tpr7, 'b', label='GCN (AUC = %0.3f)' % roc_auc7, color='palevioletred', linewidth=1.5)
    plt.plot(fpr8, tpr8, 'b', label='GAT (AUC = %0.3f)' % roc_auc8, color='peru', linewidth=1.5)
    plt.plot(fpr2, tpr2, 'b', label='preMLI (AUC = %0.3f)' % roc_auc2, color='mediumaquamarine', linewidth=1.5)
    plt.plot(fpr3, tpr3, 'b', label='CIRNN (AUC = %0.3f)' % roc_auc3, color='olive', linewidth=1.5)
    plt.plot(fpr4, tpr4, 'b', label='LncMirNet (AUC = %0.3f)' % roc_auc4, color='blue', linewidth=1.5)
    plt.plot(fpr5, tpr5, 'b', label='PmliPred (AUC = %0.3f)' % roc_auc5, color='orange', linewidth=1.5)
    plt.plot(fpr6, tpr6, 'b', label='PmliHFM (AUC = %0.3f)' % roc_auc6, color='purple', linewidth=1.5)
    plt.legend(loc='lower right', prop=Font)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.title("ROC Curves", Font_title)
    plt.tick_params(labelsize=10)
    plt.show()
    return abs(fpr1 - tpr1).max(), abs(fpr2 - tpr2).max(), abs(fpr3 - tpr3).max(), abs(fpr5 - tpr5).max(), abs(fpr6 - tpr6).max() #, abs(fpr4 - tpr4).max()

def kt(y_predicted1, y_true1, y_predicted2, y_true2, y_predicted3, y_true3, y_predicted4, y_true4 , y_predicted5, y_true5, y_predicted6, y_true6, y_predicted7, y_true7, y_predicted8, y_true8):#
    Font = {'size': 12, 'family': 'Times New Roman'}
    Font_title = {'size': 18, 'family': 'Times New Roman'}

    label1 = y_true1
    label2 = y_true2
    label3 = y_true3
    label4 = y_true4
    label5 = y_true5
    label6 = y_true6
    label7 = y_true7
    label8 = y_true8


    precision_1, recall_1, threshold_1 = precision_recall_curve(label1, y_predicted1)  # 计算Precision和Recall
    aupr_1 = auc(recall_1, precision_1)  # 计算AUPR值
    precision_2, recall_2, threshold_2 = precision_recall_curve(label2, y_predicted2)  # 计算Precision和Recall
    aupr_2 = auc(recall_2, precision_2)  # 计算AUPR值
    precision_3, recall_3, threshold_3 = precision_recall_curve(label3, y_predicted3)  # 计算Precision和Recall
    aupr_3 = auc(recall_3, precision_3)  # 计算AUPR值
    precision_4, recall_4, threshold_4 = precision_recall_curve(label4, y_predicted4)  # 计算Precision和Recall
    aupr_4 = auc(recall_4, precision_4)  # 计算AUPR值
    precision_5, recall_5, threshold_5 = precision_recall_curve(label5, y_predicted5)  # 计算Precision和Recall
    aupr_5 = auc(recall_5, precision_5)  # 计算AUPR值
    precision_6, recall_6, threshold_6 = precision_recall_curve(label6, y_predicted6)  # 计算Precision和Recall
    aupr_6 = auc(recall_6, precision_6)  # 计算AUPR值
    precision_7, recall_7, threshold_7 = precision_recall_curve(label7, y_predicted7)  # 计算Precision和Recall
    aupr_7 = auc(recall_7, precision_7)  # 计算AUPR值
    precision_8, recall_8, threshold_8 = precision_recall_curve(label8, y_predicted8)  # 计算Precision和Recall
    aupr_8 = auc(recall_8, precision_8)  # 计算AUPR值

    plt.figure(figsize=(10, 10))
    plt.plot(precision_1, recall_1, 'b', label='GIN (AUPR = %0.3f)' % aupr_1, color='Red', linewidth=1.5)
    plt.plot(precision_7, recall_7, 'b', label='GCN (AUPR = %0.3f)' % aupr_7, color='palevioletred', linewidth=1.5)
    plt.plot(precision_8, recall_8, 'b', label='GAT (AUPR = %0.3f)' % aupr_8, color='peru', linewidth=1.5)
    plt.plot(precision_2, recall_2, 'b', label='preMLI (AUPR = %0.3f)' % aupr_2, color='mediumaquamarine', linewidth=1.5)
    plt.plot(precision_3, recall_3, 'b', label='CIRNN (AUPR = %0.3f)' % aupr_3, color='olive', linewidth=1.5)
    plt.plot(precision_4, recall_4, 'b', label='LncMirNet (AUPR = %0.3f)' % aupr_4, color='blue', linewidth=1.5)
    plt.plot(precision_5, recall_5, 'b', label='PmliPred (AUPR = %0.3f)' % aupr_5, color='orange', linewidth=1.5)
    plt.plot(precision_6, recall_6, 'b', label='PmliHFM (AUPR = %0.3f)' % aupr_6, color='purple', linewidth=1.5)
    plt.legend(loc='lower left', prop=Font)
    plt.plot([0.5, 1], [1, 0.5], 'k--')
    plt.xlim([0.5, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', Font)
    plt.xlabel('Recall', Font)
    plt.title("Precision-Recall Curves", Font_title)
    plt.tick_params(labelsize=10)
    plt.show()
    return print("finished")

GNN_test = np.load('./predict_results/GIN-test.npy')
GNN_pred = np.load('./predict_results/GIN-pred.npy')
GCN_test = np.load('./predict_results/GCN-test.npy')
GCN_pred = np.load('./predict_results/GCN-pred.npy')
GAT_test = np.load('./predict_results/GAT-test.npy')
GAT_pred = np.load('./predict_results/GAT-pred.npy')
Premli_test = np.load('./predict_results/Premli-test.npy')
Premli_pred = np.load('./predict_results/Premli-pred.npy')
CIRNN_test = np.load('./predict_results/CIRNN-test.npy')
CIRNN_pred = np.load('./predict_results/CIRNN-pred.npy')
LM_test = np.load('./predict_results/Lncmirnet-test.npy')
LM_pred = np.load('./predict_results/Lncmirnet-pred.npy')
PP_test = np.load('./predict_results/PmliPred-test.npy')
PP_pred = np.load('./predict_results/PmliPred-pred.npy')
PH_test = np.load('./predict_results/PmliHFM-test.npy')
PH_pred = np.load('./predict_results/PmliHFM-pred.npy')

#print AUROC
print(ks(GNN_pred, GNN_test, Premli_pred, Premli_test, CIRNN_pred, CIRNN_test,LM_pred, LM_test, PP_pred, PP_test, PH_pred, PH_test, GCN_pred, GCN_test, GAT_pred, GAT_test))#
#print AUPRC
#print(kt(GNN_pred, GNN_test, Premli_pred, Premli_test, CIRNN_pred, CIRNN_test,LM_pred, LM_test, PP_pred, PP_test, PH_pred, PH_test, GCN_pred, GCN_test, GAT_pred, GAT_test))#