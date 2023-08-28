# pip install requirements.txt

from flask import Flask, jsonify, request
import torch
import numpy as np
from neo4j import GraphDatabase
import networkx as nx
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


def create_data(user,password):
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=(user, password))
    def run_query(query):
        with driver.session() as session:
            result = session.run(query)
            records = [record for record in result]
            return records
    query = " MATCH (a:Author)-[:CO_AUTHORED]->(b:Author) RETURN a.author_id AS author1, b.author_id AS author2 "
    records = run_query(query)
    G = nx.Graph()
    author_to_index = {}
    index = 0
    for record in records:
        author1 = record["author1"]
        author2 = record["author2"]
        if author1 not in author_to_index:
            author_to_index[author1] = index
            index += 1
        if author2 not in author_to_index:
            author_to_index[author2] = index
            index += 1
        G.add_edge(author_to_index[author1], author_to_index[author2])

    author_features_query = "MATCH (a:Author) RETURN a.author_id AS author_id,a.Feature28 AS Feature28,a.Feature24 AS Feature24,        a.Feature27 AS Feature27,        a.Feature26 AS Feature26,        a.Feature29 AS Feature29,        a.Feature20 AS Feature20,        a.Feature23 AS Feature23,        a.Feature22 AS Feature22,        a.Feature25 AS Feature25,        a.Feature174 AS Feature174,        a.Feature177 AS Feature177,        a.Feature176 AS Feature176,        a.Feature21 AS Feature21,        a.Feature172 AS Feature172,        a.Feature179 AS Feature179,        a.Feature178 AS Feature178,        a.Feature175 AS Feature175,        a.Feature39 AS Feature39,        a.Feature171 AS Feature171,        a.Feature170 AS Feature170,        a.Feature173 AS Feature173,        a.Feature36 AS Feature36,        a.Feature35 AS Feature35,        a.Feature38 AS Feature38,        a.Feature37 AS Feature37,        a.Feature32 AS Feature32,        a.Feature31 AS Feature31,        a.Feature34 AS Feature34,        a.Feature33 AS Feature33,        a.Feature163 AS Feature163,        a.Feature166 AS Feature166,        a.Feature165 AS Feature165,        a.Feature30 AS Feature30,        a.Feature168 AS Feature168,        a.Feature167 AS Feature167,        a.Feature169 AS Feature169,        a.Feature164 AS Feature164,        a.Feature48 AS Feature48,        a.Feature160 AS Feature160,        a.Feature162 AS Feature162,        a.Feature161 AS Feature161,        a.Feature44 AS Feature44,        a.Feature47 AS Feature47,        a.Feature46 AS Feature46,        a.Feature49 AS Feature49,        a.Feature40 AS Feature40,        a.Feature43 AS Feature43,        a.Feature42 AS Feature42,        a.Feature45 AS Feature45,        a.Feature196 AS Feature196,        a.Feature199 AS Feature199,        a.Feature198 AS Feature198,        a.Feature41 AS Feature41,        a.Feature194 AS Feature194,        a.Feature191 AS Feature191,        a.Feature190 AS Feature190,        a.Feature197 AS Feature197,        a.Feature59 AS Feature59,        a.Feature193 AS Feature193,        a.Feature192 AS Feature192,        a.Feature195 AS Feature195,        a.Feature56 AS Feature56,        a.Feature55 AS Feature55,        a.Feature58 AS Feature58,        a.Feature57 AS Feature57,        a.Feature52 AS Feature52,        a.Feature51 AS Feature51,        a.Feature54 AS Feature54,        a.Feature53 AS Feature53,        a.Feature185 AS Feature185,        a.Feature188 AS Feature188,        a.Feature187 AS Feature187,        a.Feature50 AS Feature50,        a.Feature183 AS Feature183,        a.Feature180 AS Feature180,        a.Feature189 AS Feature189,        a.Feature186 AS Feature186,        a.Feature19 AS Feature19,        a.Feature182 AS Feature182,        a.Feature181 AS Feature181,        a.Feature184 AS Feature184,        a.Feature16 AS Feature16,        a.Feature15 AS Feature15,        a.Feature18 AS Feature18,        a.Feature17 AS Feature17,        a.Feature12 AS Feature12,        a.Feature11 AS Feature11,        a.Feature14 AS Feature14,        a.Feature13 AS Feature13,        a.Feature204 AS Feature204,        a.Feature207 AS Feature207,        a.Feature206 AS Feature206,        a.Feature10 AS Feature10,        a.Feature210 AS Feature210,        a.Feature209 AS Feature209,        a.Feature208 AS Feature208,        a.Feature205 AS Feature205,        a.Feature212 AS Feature212,        a.Feature211 AS Feature211,        a.Feature214 AS Feature214,        a.Feature213 AS Feature213,        a.Feature201 AS Feature201,        a.Feature200 AS Feature200,        a.Feature203 AS Feature203,        a.Feature202 AS Feature202,        a.Feature106 AS Feature106,        a.Feature105 AS Feature105,        a.Feature108 AS Feature108,        a.Feature107 AS Feature107,        a.Feature114 AS Feature114,        a.Feature111 AS Feature111,        a.Feature110 AS Feature110,        a.Feature109 AS Feature109,        a.Feature6 AS Feature6,        a.Feature113 AS Feature113,        a.Feature112 AS Feature112,        a.Feature115 AS Feature115,        a.Feature1 AS Feature1,        a.Feature9 AS Feature9,        a.Feature8 AS Feature8,        a.Feature7 AS Feature7,        a.Feature5 AS Feature5,        a.Feature4 AS Feature4,        a.Feature3 AS Feature3,        a.Feature2 AS Feature2,        a.Feature216 AS Feature216,        a.Feature215 AS Feature215,        a.Feature218 AS Feature218,        a.Feature217 AS Feature217,        a.Feature100 AS Feature100,        a.Feature221 AS Feature221,        a.Feature220 AS Feature220,        a.Feature219 AS Feature219,        a.Feature222 AS Feature222,        a.Feature104 AS Feature104,        a.Feature103 AS Feature103,        a.Feature224 AS Feature224,        a.Feature68 AS Feature68,        a.Feature102 AS Feature102,        a.Feature223 AS Feature223,        a.Feature101 AS Feature101,        a.Feature67 AS Feature67,        a.Feature66 AS Feature66,        a.Feature129 AS Feature129,        a.Feature69 AS Feature69,        a.Feature65 AS Feature65,        a.Feature128 AS Feature128,        a.Feature64 AS Feature64,        a.Feature127 AS Feature127,        a.Feature61 AS Feature61,        a.Feature60 AS Feature60,        a.Feature63 AS Feature63,        a.Feature62 AS Feature62,        a.Feature131 AS Feature131,        a.Feature130 AS Feature130,        a.Feature133 AS Feature133,        a.Feature132 AS Feature132,        a.Feature135 AS Feature135,        a.Feature134 AS Feature134,        a.Feature137 AS Feature137,        a.Feature136 AS Feature136,        a.Feature119 AS Feature119,        a.Feature77 AS Feature77,        a.Feature118 AS Feature118,        a.Feature79 AS Feature79,        a.Feature117 AS Feature117,        a.Feature75 AS Feature75,        a.Feature116 AS Feature116,        a.Feature78 AS Feature78,        a.Feature71 AS Feature71,        a.Feature74 AS Feature74,        a.Feature73 AS Feature73,        a.Feature76 AS Feature76,        a.Feature70 AS Feature70,        a.Feature122 AS Feature122,        a.Feature121 AS Feature121,        a.Feature72 AS Feature72,        a.Feature123 AS Feature123,        a.Feature126 AS Feature126,        a.Feature125 AS Feature125,        a.Feature120 AS Feature120,        a.Feature149 AS Feature149,        a.Feature89 AS Feature89,        a.Feature88 AS Feature88,        a.Feature124 AS Feature124,        a.Feature85 AS Feature85,        a.Feature84 AS Feature84,        a.Feature87 AS Feature87,        a.Feature86 AS Feature86,        a.Feature80 AS Feature80,        a.Feature154 AS Feature154,        a.Feature83 AS Feature83,        a.Feature82 AS Feature82,        a.Feature153 AS Feature153,        a.Feature152 AS Feature152,        a.Feature81 AS Feature81,        a.Feature155 AS Feature155,        a.Feature157 AS Feature157,        a.Feature156 AS Feature156,        a.Feature159 AS Feature159,        a.Feature158 AS Feature158,        a.Feature138 AS Feature138,        a.Feature99 AS Feature99,        a.Feature151 AS Feature151,        a.Feature150 AS Feature150,        a.Feature95 AS Feature95,        a.Feature98 AS Feature98,        a.Feature139 AS Feature139,        a.Feature97 AS Feature97,        a.Feature143 AS Feature143,        a.Feature94 AS Feature94,        a.Feature93 AS Feature93,        a.Feature96 AS Feature96,        a.Feature141 AS Feature141,        a.Feature92 AS Feature92,        a.Feature144 AS Feature144,        a.Feature91 AS Feature91,        a.Feature148 AS Feature148,        a.Feature147 AS Feature147,        a.Feature90 AS Feature90,        a.Feature142 AS Feature142,        a.Feature140 AS Feature140,        a.Feature146 AS Feature146,        a.Feature145 AS Feature145  ORDER BY a.author_id"
    author_features_records = run_query(author_features_query)

    author_features_dict = {}
    for record in author_features_records:
        author_id = record["author_id"]
        features = [int(record[f"Feature{i}"]) for i in range(1, 225)]  
        author_features_dict[author_id] = features

    edge_index = torch.tensor(list(G.edges), dtype=torch.int).t().contiguous()

    sorted_author_ids = sorted(author_to_index, key=author_to_index.get)

    x = torch.tensor([author_features_dict[author_id] for author_id in sorted_author_ids], dtype=torch.float)

    num_classes = 2 
    y = torch.randint(0, num_classes, (x.size(0),), dtype=torch.float)

    data = Data(x=x,y=y, edge_index=edge_index)
    driver.close()
    return data,author_to_index

data,author_to_index = create_data("neo4j","Swayam2003")

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
    app.run(host = "0.0.0.0",port = 5000)
