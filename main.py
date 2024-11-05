import tarfile
import gzip
import shutil
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import networkx as nx
import numpy
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, summary
from torch_geometric.utils import train_test_split_edges
import torch_sparse

datasetNames = ['facebook','gplus','twitter']
#hyperparameters
batch_size = 256
num_neighbors = [10,10]
lr = 2e-3
hidden_dim = 64
epoch_border = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unzippingFromTar(filename):
    if filename.endswith("tar.gz"):
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path='/unzipped')
            print("File extracted successfully.")

def unzippingFromTxt(filename):
    if filename.endswith("txt.gz"):
        with gzip.open(filename, 'rb') as f_in:
          splitted_name = filename.split('.')
          with open(f'/unzipped/{splitted_name[0]}.txt', 'wb') as f_out:
              shutil.copyfileobj(f_in, f_out)

def gettingDatanames(directory, typeData) -> list:
  filenames = []
  for filename in os.listdir(directory):
    if filename.endswith(typeData):
      filenames.append(f'{directory}/{filename}')
  filenames.sort(key = lambda x: int(x.split('.')[0].split('/')[3]))
  return filenames

def creatinEdgeData(path):
  edges_df = pd.read_csv(path, sep='\s+', header=None, names=['source', 'target'])

  # Get unique nodes from the edge list
  nodes = pd.concat([edges_df['source'], edges_df['target']]).unique()

  # Encode the node labels into integers (ensure all nodes are included)
  label_encoder = LabelEncoder()
  encoded_nodes = label_encoder.fit_transform(nodes)

  # Edge indices for PyTorch Geometric (re-encoded for consistency)
  edges_df['source'] = label_encoder.transform(edges_df['source'])
  edges_df['target'] = label_encoder.transform(edges_df['target'])

  edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

  # Updated number of nodes after encoding
  num_nodes = len(nodes)

  # Create a feature matrix for all nodes (dummy features, identity matrix for now)
  x = torch.eye(num_nodes, dtype=torch.float)
  # Dummy labels for node classification
  y = torch.randint(0, 2, (num_nodes,), dtype=torch.long)

  # Create the PyTorch Geometric Data object
  data = Data(x=x, edge_index=edge_index, y=y)
    # Split dataset into training and test
  data.train_mask = torch.rand(len(nodes)) < 0.8  # 80% training data
  data.test_mask = ~data.train_mask

  return data

class BaseDataset(Dataset):
    """
    TODO, későbbi verzióra fog készülni
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
    @property
    def raw_file_names(self):
        return gettingDatanames(f'/unzipped{self.root}', '.edges')

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(0,len(gettingDatanames(f'/unzipped{self.root}', '.edges')))]

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = creatinEdgeData(raw_path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

class LinkPredictor(torch.nn.Module):
    """
    Élprédikáció: Megjósolja, hogy az adott két csúcs között van-e él
    """
    def __init__(self, input_dim, hidden_dim):
        super(LinkPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_src, z_dst = z[src], z[dst]
        return torch.sigmoid(self.fc(torch.cat([z_src, z_dst], dim=1)))

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, data.edge_index)

def edge_train_model(train_loader, test_loader, optimizer, model, device,criterion):
    def train():
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Kódolás
            z = model.encode(batch.x, batch.edge_index)

            # Pozitív élek dekódolása
            pos_pred = model.decode(z, batch.edge_index)

            # Negatív minták generálása
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index, num_nodes=batch.x.size(0), num_neg_samples=batch.edge_index.size(1)
            )

            # Negatív élek dekódolása
            neg_pred = model.decode(z, neg_edge_index)

            # Loss számítása: Pozitív és negatív minták összehasonlítása
            pos_loss = F.binary_cross_entropy(pos_pred.squeeze(), torch.ones(pos_pred.size(0)).to(device))
            neg_loss = F.binary_cross_entropy(neg_pred.squeeze(), torch.zeros(neg_pred.size(0)).to(device))

            # Teljes veszteség
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_loader)

    def test():
        model.eval()
        correct = 0
        total = 0  # Összes tesztadat száma

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                # Kódolás és predikció
                z = model.encode(batch.x, batch.edge_index)

                # Pozitív élek predikciója
                pos_pred = model.decode(z, batch.edge_index)

                # Negatív élek mintavételezése (azaz BSZ2 szempontjából komplementere)
                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index, num_nodes=batch.x.size(0), num_neg_samples=batch.edge_index.size(1)
                )
                neg_pred = model.decode(z, neg_edge_index)

                # Predikciók összehasonlítása
                pos_correct = (pos_pred > 0.5).sum().item()
                neg_correct = (neg_pred < 0.5).sum().item()

                correct += pos_correct + neg_correct
                #correct += pos_correct
                total += pos_pred.size(0) + neg_pred.size(0)
                #total += pos_pred.size(0)
        return correct / total

    for epoch in range(epoch_border):
        loss = train()
        acc = test()
        # Elemzés
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

def train_edge(data,device, model):
  criterion = CrossEntropyLoss()
  train_loader = NeighborLoader(data, num_neighbors=num_neighbors , batch_size=batch_size, input_nodes=data.train_mask)
  test_loader = NeighborLoader(data, num_neighbors=num_neighbors , batch_size=batch_size, input_nodes=data.test_mask)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
  edge_train_model(train_loader, test_loader, optimizer, model, device,criterion)



if __name__ == '__main__':
    #Későbbi verzióra
    '''
    for datasetName in datasetNames:
        unzippingFromTar(f'{datasetName}.tar.gz')
    facebook_dataset = BaseDataset(root='/facebook')
    gplus_dataset = BaseDataset(root='/gplus')
    twitter_dataset = BaseDataset(root='/twitter')
    '''
    for datasetName in datasetNames:
        print(datasetName)
        if not os.path.exists("/unzipped"):
            os.mkdir("/unzipped")
        unzippingFromTxt(f'{datasetName}_combined.txt.gz')
    data_facebook = creatinEdgeData('/unzipped/facebook_combined.txt')
    data_gplus = creatinEdgeData('/unzipped/facebook_combined.txt')
    data_twitter = creatinEdgeData('/unzipped/facebook_combined.txt')

    model = LinkPredictor(data_facebook.num_features, hidden_dim).to(device)
    # Instantiate the model, optimizer, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    print('Facebook')
    train_edge(data_facebook,device,model)
    print('GPlus')
    train_edge(data_gplus,device,model)
    print('Twitter')
    train_edge(data_twitter,device,model)
