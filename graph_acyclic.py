import torch
import json
import pandas as pd
import sys
import numpy as np
from pykt.models import init_model
from pykt.models.dkt import DKT  # Ensure this import matches your model’s location
# Paths

#2009 Subset
# model_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2009_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_61b18c72-7a08-4f47-abac-517f2de6b3df/qid_model.ckpt'
# config_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2009_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_61b18c72-7a08-4f47-abac-517f2de6b3df/config.json'
# keyid2idx_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/data/assist2009/keyid2idx.json'

#2017 Subset
# config_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2017_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_e7d4fccf-ae92-4a29-b752-8a14fb98d181/config.json'
# model_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2017_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_e7d4fccf-ae92-4a29-b752-8a14fb98d181/qid_model.ckpt'
# keyid2idx_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/data/assist2017/keyid2idx.json'

#2012 Subset
# config_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2012_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_1cddda95-5295-41c2-ba71-1b6257f173be/config.json'
# model_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2012_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_1cddda95-5295-41c2-ba71-1b6257f173be/qid_model.ckpt'
# keyid2idx_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/data/assist2012_subset_problem_bodies/keyid2idx.json'

#2009 Fullset
# config_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2009_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_7eabe162-8d6b-4395-a505-043cfd54cc8d/config.json'
# model_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2009_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_7eabe162-8d6b-4395-a505-043cfd54cc8d/qid_model.ckpt'
# keyid2idx_path = '/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/data/assist2009_full_dataset/keyid2idx.json'

#2012 Fullset
# config_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2012_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_7412ad24-3119-4f67-be8e-9b0573c6a07f/config.json'
# model_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2012_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_7412ad24-3119-4f67-be8e-9b0573c6a07f/qid_model.ckpt'
# keyid2idx_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/data/assist2012_fulldataset/keyid2idx.json'

#2017 Fullset
config_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2017_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_e43dc05e-49bd-4c7e-a3c1-62006facfbf9/config.json'
model_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2017_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_e43dc05e-49bd-4c7e-a3c1-62006facfbf9/qid_model.ckpt'
keyid2idx_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/data/assist2017_full_dataset2/keyid2idx.json'

#2015 Fullset
# config_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2015_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_4a0d3a8d-4e88-42a1-a6de-08b815fe62c9/config.json'
# model_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2015_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_4a0d3a8d-4e88-42a1-a6de-08b815fe62c9/qid_model.ckpt'
# keyid2idx_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/data/assist2015_full_dataset/keyid2idx.json'

import networkx as nx
import matplotlib.pyplot as plt

# Load model configuration
with open(config_path, 'r') as file:
    config = json.load(file)
model_config = config['model_config']
data_config = config['data_config']
# Remove training-specific parameters from model_config
training_params = ['learning_rate', 'optimizer', 'batch_size', 'num_epochs', 'use_wandb','add_uuid']
model_config = {key: val for key, val in model_config.items() if key not in training_params}
# Initialize the DKT model
model = init_model('dkt', model_config, data_config, 'qid')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
# Load keyid2idx mapping
with open(keyid2idx_path, 'r') as file:
    keyid2idx = json.load(file)

is_acyclic = False
threshold_min = 0.0
threshold_step = 0.0001 

while is_acyclic == False:
    
    keys = keyid2idx["concepts"].keys()
    flipped_dict = {value: key for key, value in keyid2idx["concepts"].items()}
    adjacency_list = {}

    for concept_id in keys:
        question_ids = [concept_id] #skill id
        responses = [1]  # -1 as a placeholder for the prediction
        # Convert question IDs to indices using keyid2idx mapping
        question_indices = [keyid2idx['concepts'][qid] for qid in question_ids]
        # Convert to tensors
        questions_tensor = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)
        responses_tensor = torch.tensor(responses, dtype=torch.long).unsqueeze(0)
        # Run the model for prediction
        with torch.no_grad():
            prediction = model(questions_tensor, responses_tensor)
            # Extract the prediction for the last question (56961)
            last_question_pred = prediction[:, -1, :]
            # print(last_question_pred)
            total_sum = last_question_pred.sum().item() #sums all the tensor indeces
            last_question_pred = last_question_pred / total_sum
            # print(last_question_pred)
            last_question_pred = torch.where(last_question_pred < threshold_min, torch.tensor(0), torch.tensor(1)) #if the relation is less than 0.015 set the index to 0 otherwise set it to 1
            last_question_pred_list = last_question_pred[0].tolist() #converts tensor to list
            # print(last_question_pred_list)
            
            indices_of_ones = [flipped_dict[index] for index, value in enumerate(last_question_pred_list) if value == 1] #extracts the 1's from last_question_pred_list and its associated index
            indices_of_ones = [value for value in indices_of_ones if value != concept_id] #removes loops in the graph
            adjacency_list[concept_id] = indices_of_ones 
            # print(adjacency_list)

    G = nx.DiGraph(adjacency_list)
    
    if not nx.is_directed_acyclic_graph(G):
        threshold_min = threshold_min + threshold_step
    # If the graph is acyclic, move the minimum threshold up
    else:
        is_acyclic = True

formatted_threshold = "{:.4f}".format(threshold_min)
print("threshold:", formatted_threshold)

keys = keyid2idx["concepts"].keys()
# print(keys)
flipped_dict = {value: key for key, value in keyid2idx["concepts"].items()}
# print(flipped_dict)
adjacency_list = {}

for idx, concept_id in enumerate(keys):
    question_ids = [concept_id] #skill id
    responses = [1]  # -1 as a placeholder for the prediction
    # Convert question IDs to indices using keyid2idx mapping
    question_indices = [keyid2idx['concepts'][qid] for qid in question_ids]
    # Convert to tensors
    questions_tensor = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)
    responses_tensor = torch.tensor(responses, dtype=torch.long).unsqueeze(0)
    # Run the model for prediction
    with torch.no_grad():
        prediction = model(questions_tensor, responses_tensor)
        # Extract the prediction for the last question (56961)
        last_question_pred = prediction[:, -1, :]
        # if idx == 10 or idx == 13:
        #     print(last_question_pred)
        total_sum = last_question_pred.sum().item() #sums all the tensor indeces
        last_question_pred = last_question_pred / total_sum
        # if idx == 10 or idx == 13:
        #     print(last_question_pred)
        last_question_pred = torch.where(last_question_pred < threshold_min, torch.tensor(0), torch.tensor(1)) #if the relation is less than 0.015 set the index to 0 otherwise set it to 1
        # if idx == 10 or idx == 13:
        #     print(last_question_pred)
        last_question_pred_list = last_question_pred[0].tolist() #converts tensor to list
        indices_of_ones = [flipped_dict[index] for index, value in enumerate(last_question_pred_list) if value == 1] #extracts the 1's from last_question_pred_list and its associated index
        # if idx == 10 or idx == 13:
        #     print(indices_of_ones)
        indices_of_ones = [value for value in indices_of_ones if value != concept_id] #removes loops in the graph
        adjacency_list[concept_id] = indices_of_ones
        # if idx == 10 or idx == 13:
        #     print(adjacency_list)

G = nx.DiGraph(adjacency_list)

print(adjacency_list)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print("number of nodes:", num_nodes)
print("number of edges:", num_edges) 

pos = nx.circular_layout(G)
plt.figure(figsize=(12, 8)) 
nx.draw(G, pos, ax=None, with_labels=True, font_size=8, node_size=500, node_color='lightblue', edge_color='gray', alpha=0.5)
plt.show()

# nodes_with_outgoing_edges = [node for node, out_degree in G.out_degree() if out_degree > 0]
# print(nodes_with_outgoing_edges)

nodes_with_degree_greater_than_one = [node for node, degree in G.degree() if degree >= 1]
print(nodes_with_degree_greater_than_one)