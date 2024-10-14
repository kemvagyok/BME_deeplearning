if __name__ == '__main__':
    files = ['facebook.tar.gz','gplus.tar.gz','twitter.tar.gz']  # Or the actual name of your file

    # Extract the .tar.gz file
    for uploaded_file in files:
    if uploaded_file.endswith("tar.gz"):
        with tarfile.open(uploaded_file, "r:gz") as tar:
            tar.extractall(path='/')
            print("File extracted successfully.")

    facebook_edges_file = '/facebook/0.edges'
    gplus_edges_file = '/gplus/100129275726588145876.edges'
    twitter_edges_file = '/twitter/12831.edges'
    datasets_edges_file = [facebook_edges_file, gplus_edges_file,twitter_edges_file ]
    for edges_file in datasets_edges_file:
    edges_df = pd.read_csv(edges_file, delim_whitespace=True, header=None, names=['source', 'target'])

    # Get unique nodes from the edge list
    nodes = pd.concat([edges_df['source'], edges_df['target']]).unique()

    # Create a graph using NetworkX
    G = nx.from_pandas_edgelist(edges_df, 'source', 'target')

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


    class GCNNet(nn.Module):
        def __init__(self):
            super(GCNNet, self).__init__()
            self.conv1 = GCNConv(data.num_features, 16)  # Input to hidden layer
            self.conv2 = GCNConv(16, 2)  # Hidden to output layer (binary classification)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))  # Apply first GCN layer with ReLU
            x = F.dropout(x, training=self.training)  # Apply dropout for regularization
            x = self.conv2(x, edge_index)  # Apply second GCN layer
            return F.log_softmax(x, dim=1)

    # Instantiate the model, optimizer, and loss function
    model = GCNNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Split dataset into training and test
    train_mask = torch.rand(len(nodes)) < 0.8  # 80% training data
    test_mask = ~train_mask

    # Training loop
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    # Evaluation loop
    def test():
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)
        correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()
        return acc

    # Train for 100 epochs
    for epoch in range(100):
        loss = train()
        if epoch % 10 == 0:
            acc = test()
            print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')