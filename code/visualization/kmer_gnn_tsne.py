from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from umap import UMAP


before_3mer = np.load("./embeddings/before_GNN_embedding_3mer.npy")
after_3mer = np.load("./embeddings/after_GNN_embedding_3mer.npy")
before_4mer = np.load("./embeddings/before_GNN_embedding_4mer.npy")
after_4mer = np.load("./embeddings/after_GNN_embedding_4mer.npy")
before_5mer = np.load("./embeddings/before_GNN_embedding_5mer.npy")
after_5mer = np.load("./embeddings/after_GNN_embedding_5mer.npy")
before_6mer = np.load("./embeddings/before_GNN_embedding_6mer.npy")
after_6mer = np.load("./embeddings/after_GNN_embedding_6mer.npy")
labels = np.load("./embeddings/labels.npy")


before_3mer = pd.DataFrame(before_3mer)
after_3mer = pd.DataFrame(after_3mer)
before_4mer = pd.DataFrame(before_4mer)
after_4mer = pd.DataFrame(after_4mer)
before_5mer = pd.DataFrame(before_5mer)
after_5mer = pd.DataFrame(after_5mer)
before_6mer = pd.DataFrame(before_6mer)
after_6mer = pd.DataFrame(after_6mer)

labels = pd.DataFrame({'labels':labels})


labels.loc[labels['labels'] == 0] = 'Interaction'
labels.loc[labels['labels'] == 1] = 'Non-interaction'

#before GNN
df3_1 = pd.concat([before_3mer,labels], axis=1)
df3_2 = pd.concat([after_3mer,labels], axis=1)
df4_1 = pd.concat([before_4mer,labels], axis=1)
df4_2 = pd.concat([after_4mer,labels], axis=1)
df5_1 = pd.concat([before_5mer,labels], axis=1)
df5_2 = pd.concat([after_5mer,labels], axis=1)
df6_1 = pd.concat([before_6mer,labels], axis=1)
df6_2 = pd.concat([after_6mer,labels], axis=1)


features3_1 = df3_1.loc[:, :255]
features3_2 = df3_2.loc[:, :255]
features4_1 = df4_1.loc[:, :255]
features4_2 = df4_2.loc[:, :255]
features5_1 = df5_1.loc[:, :255]
features5_2 = df5_2.loc[:, :255]
features6_1 = df6_1.loc[:, :255]
features6_2 = df6_2.loc[:, :255]

target = df3_1.labels


#for T-SNE
print("begin TSNE")
tsne = TSNE(n_components=2)
print("finished TSNE")
print("begin projection")
projections3_1 = tsne.fit_transform(features3_1)
print("finished 1")
projections3_2 = tsne.fit_transform(features3_2)
print("finished 2")
projections4_1 = tsne.fit_transform(features4_1)
print("finished 3")
projections4_2 = tsne.fit_transform(features4_2)
print("finished 4")
projections5_1 = tsne.fit_transform(features5_1)
print("finished 5")
projections5_2 = tsne.fit_transform(features5_2)
print("finished 6")
projections6_1 = tsne.fit_transform(features6_1)
print("finished 7")
projections6_2 = tsne.fit_transform(features6_2)
print("finished 8")


print("finished projection")

print("begin scatter")


fig, ((ax1,ax3,ax5,ax7),(ax2,ax4,ax6,ax8)) = plt.subplots(2,4)
tsne_result_df_1 = pd.DataFrame({'tsne_1': projections3_1[:,0], 'tsne_2': projections3_1[:,1], 'label': df3_1.labels})
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df_1, ax=ax1,s=5)
lim = (projections3_1.min()-5, projections3_1.max()+5)
ax1.set_xlim(lim)
ax1.set_ylim(lim)
ax1.set_aspect('equal')
ax1.set_title('A', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 22}, loc='left')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel("3-mer", fontdict={'family': 'Times New Roman', 'size': 18})
ax1.set_ylabel("")
ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.0, fontsize = 'x-small')


tsne_result_df_2 = pd.DataFrame({'tsne_1': projections3_2[:,0], 'tsne_2': projections3_2[:,1], 'label': df3_2.labels})
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df_2, ax=ax2,s=5)
lim = (projections3_2.min()-5, projections3_2.max()+5)
ax2.set_xlim(lim)
ax2.set_ylim(lim)
ax2.set_aspect('equal')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel("3-mer", fontdict={'family': 'Times New Roman', 'size': 18})
ax2.set_ylabel("")
ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.0, fontsize = 'x-small')


tsne_result_df_3 = pd.DataFrame({'tsne_1': projections4_1[:,0], 'tsne_2': projections4_1[:,1], 'label': df4_1.labels})
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df_3, ax=ax3,s=5)
lim = (projections4_1.min()-5, projections4_1.max()+5)
ax3.set_xlim(lim)
ax3.set_ylim(lim)
ax3.set_aspect('equal')
ax3.set_title('B', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 22}, loc='left')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel("4-mer", fontdict={'family': 'Times New Roman', 'size': 18})
ax3.set_ylabel("")
ax3.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.0, fontsize = 'x-small')

tsne_result_df_4 = pd.DataFrame({'tsne_1': projections4_2[:,0], 'tsne_2': projections4_2[:,1], 'label': df4_2.labels})
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df_4, ax=ax4,s=5)
lim = (projections4_2.min()-5, projections4_2.max()+5)
ax4.set_xlim(lim)
ax4.set_ylim(lim)
ax4.set_aspect('equal')
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xlabel("4-mer", fontdict={'family': 'Times New Roman', 'size': 18})
ax4.set_ylabel("")
ax4.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.0, fontsize = 'x-small')

tsne_result_df_5 = pd.DataFrame({'tsne_1': projections5_1[:,0], 'tsne_2': projections5_1[:,1], 'label': df5_1.labels})
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df_5, ax=ax5,s=5)
lim = (projections5_1.min()-5, projections5_1.max()+5)
ax5.set_xlim(lim)
ax5.set_ylim(lim)
ax5.set_aspect('equal')
ax5.set_title('C', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 18}, loc='left')
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_xlabel("5-mer", fontdict={'family': 'Times New Roman', 'size': 18})
ax5.set_ylabel("")
ax5.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.0, fontsize = 'x-small')

tsne_result_df_6 = pd.DataFrame({'tsne_1': projections5_2[:,0], 'tsne_2': projections5_2[:,1], 'label': df5_2.labels})
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df_6, ax=ax6,s=5)
lim = (projections5_2.min()-5, projections5_2.max()+5)
ax6.set_xlim(lim)
ax6.set_ylim(lim)
ax6.set_aspect('equal')
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_xlabel("5-mer", fontdict={'family': 'Times New Roman', 'size': 18})
ax6.set_ylabel("")
ax6.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.0, fontsize = 'x-small')

tsne_result_df_7 = pd.DataFrame({'tsne_1': projections6_1[:,0], 'tsne_2': projections6_1[:,1], 'label': df6_1.labels})
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df_7, ax=ax7,s=5)
lim = (projections6_1.min()-5, projections6_1.max()+5)
ax7.set_xlim(lim)
ax7.set_ylim(lim)
ax7.set_aspect('equal')
ax7.set_title('D', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 18}, loc='left')
ax7.set_xticks([])
ax7.set_yticks([])
ax7.set_xlabel("6-mer", fontdict={'family': 'Times New Roman', 'size': 18})
ax7.set_ylabel("")
ax7.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.0, fontsize = 'x-small')

tsne_result_df_8 = pd.DataFrame({'tsne_1': projections6_2[:,0], 'tsne_2': projections6_2[:,1], 'label': df6_2.labels})
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df_8, ax=ax8,s=5)
lim = (projections6_2.min()-5, projections6_2.max()+5)
ax8.set_xlim(lim)
ax8.set_ylim(lim)
ax8.set_aspect('equal')
ax8.set_xticks([])
ax8.set_yticks([])
ax8.set_xlabel("6-mer", fontdict={'family': 'Times New Roman', 'size': 18})
ax8.set_ylabel("")
ax8.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.0, fontsize = 'x-small')



fig.show()
plt.subplots_adjust(wspace=0.15, hspace=0)
# plt.savefig('Figure 4.png')
plt.show()