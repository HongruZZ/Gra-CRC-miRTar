import torch as t


# Set the parameter variables and change them uniformly here
class config:
    def __init__(self):
        # k value of k-mer
        self.k = 5
        # Dimension of word2vec word vector, i.e. node feature dimension
        self.d = 128
        # Parameters of the hidden layer of the graph convolutional networks
        self.hidden_dim = 128
        # Number of sample categories
        self.n_classes = 2
        # Set random seeds
        self.seed = 6657
        # Training parameters
        self.batchSize = 64
        self.num_epochs = 150
        self.lr = 0.001
        self.kFold = 5
        self.savePath = f"checkpoints/mmmodel_gin/k{self.k}_d{self.d}_h{self.hidden_dim}_b{self.batchSize}/" #这里修改了
        self.device = t.device("cuda:0")

    def set_seed(self,s):
        self.seed=s
        self.savePath = f"checkpoints/mmmodel_gin/k{self.k}_d{self.d}_h{self.hidden_dim}_b{self.batchSize}_s{self.seed}/"
