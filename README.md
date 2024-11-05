# Friend recommendation with graph neural networks
## Team:
### REX-DINO-AI 
## Members of the team
### Benedek Sági - ECSGGY
### Árkossy Máte - 
## Details of the project:

The goal of this project is to develop a personalized friend recommendation system by using Graph Neural Networks (GNNs). You have to analyze data from Facebook, Google+, or Twitter to suggest meaningful connections based on user profiles and interactions. This project offers a hands-on opportunity to deepen your deep learning and network analysis skills.

## Functions of files:
### Dockerfile
Initialzing the software environment of the project
### main.py
Including the data acquisition, data preparation.
### requirements.txt
Installing the libries of the project
## Related works:
### Github:
#### https://github.com/aditya-grover/node2vec
#### https://github.com/pyg-team/pytorch_geometric
### Papers:
#### https://arxiv.org/abs/1607.00653
#### https://arxiv.org/abs/1611.07308

## How to run it:
CMD: docker build . -t bme_deeplearning

## how to run the pipeline:
1. Preprocessing the data (FB, Gplus, twitter, only edges) 
2. Training the models with data
3. Evaluate the model

## How to train the model:
Initialize the model in training mode and reset gradients for each batch.
Encode the nodes: For each batch, the model encodes node features and edge connections (batch.x and batch.edge_index) into embeddings (z).
Decode positive edges: Using these embeddings, it predicts the existence of each positive edge (batch.edge_index).
Generate negative edges: Negative edges are sampled to act as non-existent or "false" edges.
Decode negative edges: The model predicts the absence of these negative edges using neg_edge_index.
Compute loss: This combines the error for predicting positive edges (using binary cross-entropy) with that of the negative edges. Backpropagation adjusts model weights to minimize this combined loss.
Repeat for each batch until all data is processed, calculating the average loss per epoch.
## How to evaluate the model:
Set the model in evaluation mode to disable dropout or other regularization.
Predict positive edges and sample negative edges for comparison.
Compute accuracy: For positive predictions, values >0.5 are considered correct predictions of edges, and values <0.5 for negative edges are correct for non-existent edges.
Calculate accuracy: Summing the correctly predicted positive and negative edges and dividing by the total number of predictions gives the accuracy score for that epoch.
