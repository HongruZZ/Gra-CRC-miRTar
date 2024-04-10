import torch as t


# Set the parameter variables and change them uniformly here
class config:
    def __init__(self):
        # k value of k-mer
        self.k = 5
        # Dimension of embedding vector, i.e. node feature dimension
        self.d = 128
        # Parameters of the hidden layer of the graph neural networks
        self.hidden_dim = 128
        # Number of sample categories
        self.n_classes = 2
        # Set random seeds
        self.seed = 6657
        # Training parameters
        self.batchSize = 128
        self.num_epochs = 1
        self.lr = 0.001
        self.kFold = 5
        self.GNN_type = "gin"  # "gat", "gin"
        self.savePath = f"checkpoints/mmmodel_{str(self.GNN_type)}/k{self.k}_d{self.d}_h{self.hidden_dim}_b{self.batchSize}/"
        self.device = t.device("cuda:0")

    def set_seed(self, s):
        self.seed = s
        self.savePath = f"checkpoints/mmmodel_{str(self.GNN_type)}/k{self.k}_d{self.d}_h{self.hidden_dim}_b{self.batchSize}_s{self.seed}/"
