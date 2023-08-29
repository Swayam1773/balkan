# pip install requirements.txt

from flask import Flask, jsonify, request
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import pickle

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

with open('author_to_index.pickle', 'rb') as f:
    author_to_index = pickle.load(f)

def index_to_author(value):
    for key, val in author_to_index.items():
        if val == value:
            return key

class GNNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNModel, self).__init__()
        self.conv = GNNConv(num_features, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

class GNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNConv, self).__init__(aggr="add")  # "add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

app = Flask(__name__)

loaded_model = GNNModel(data.num_node_features , 64, 1)
loaded_model.load_state_dict(torch.load('model.pt'))

def get_closest_coauthors(model, data, author_index, k=5):

    original_tensor = data.x[author_index] 
    original_tensor = original_tensor.unsqueeze(0)  
    new_tensor = torch.zeros((332, original_tensor.shape[1]))
    temp = torch.cat((original_tensor, new_tensor), dim=0)

    model.eval()
    with torch.no_grad():
        similarity_scores = model(temp, data.edge_index)

    closest_coauthor_indices = np.argsort(similarity_scores.squeeze().numpy())[:k]

    closest_coauthor_ids = [index_to_author(index) for index in closest_coauthor_indices]
    closest_coauthor_scores = similarity_scores.squeeze().numpy()[closest_coauthor_indices]

    return closest_coauthor_indices, closest_coauthor_ids, closest_coauthor_scores

@app.route('/closest_coauthors')
def closest_coauthors():

    author_id = request.args.get('id')

    author_index = author_to_index[author_id]

    indices, author_ids, likeliness_scores = get_closest_coauthors(loaded_model, data, author_index)

    results = []
    for i, (author_index, author_id, score) in enumerate(zip(indices, author_ids, likeliness_scores)):
        result = {
            'rank': str(i + 1),
            'author_id': str(author_id),
            'likeliness': str(score)
        }
        results.append(result)
    print(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True,host = "0.0.0.0",port = 5000)
