from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from umap import UMAP


before = np.load("./embeddings/before_GNN_embedding.npy")
after = np.load("./embeddings/after_GNN_embedding.npy")
labels = np.load("./embeddings/labels.npy")
before = pd.DataFrame(before)
after = pd.DataFrame(after)
labels = pd.DataFrame({'labels':labels})


labels.loc[labels['labels'] == 0] = 'interaction'
labels.loc[labels['labels'] == 1] = 'non-interaction'

#before GNN
#df = pd.concat([before,labels], axis=1)

#after GNN
df = pd.concat([after,labels], axis=1)

print(df)



features = df.loc[:, :255]
target = df.labels


#for T-SNE
print("begin TSNE")
tsne = TSNE(n_components=2, random_state=0, perplexity=50)
print("finished TSNE")
print("begin projection")
projections = tsne.fit_transform(features)
print("finished projection")
print(df.labels)


'''
#for UMAP
print("begin UMAP")
umap_2d = UMAP(n_components=3, n_neighbors=500, min_dist=0.6, init='random', random_state=0)
print("finished UMAP")
print("begin projection")
projections = umap_2d.fit_transform(features)
print("finished projection")
print(df.labels)
'''


print("begin scatter")

fig = px.scatter(
    projections, x=0, y=1,
    color=df.labels,
    labels={0: "Sepal Length (cm)",1: "Sepal Width (cm)", 'color': 'labels'}, title='before GNN',
)

fig.update_layout(
    title="",
    xaxis_title="",
    yaxis_title="",
    legend_title="",
    font=dict(
        family="Times New Roman, monospace",
        size=30,
        color="black"
    )
)

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))



fig.update_yaxes(visible=False, showticklabels=False)
fig.update_xaxes(visible=False, showticklabels=False)

print("start writing")

fig.show()