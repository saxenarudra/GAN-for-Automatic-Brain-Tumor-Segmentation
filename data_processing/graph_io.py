import numpy as np
import os
import networkx as nx
import pickle
import json

'''
A collection of helper functions for graph processing
'''


'''
Input:
A supervoxel partitioning(2D or 3D array of integers)
A vector of labels for each node of the same length as the number of unique integers in the supervoxel partitioning (excl -1)
Function:
Assigns each voxel the label of the supervoxel it belongs to. E.g. if a voxel has the value 100, it will be assigned the label
at index 100 in the vector of node_labels.
Voxels labelled with -1 are assigned a label of 0.
'''
def project_nodes_to_img(svs,node_label):
    #the background is -1 in the sv partitioning, so set this to be healthy
    node_label = np.append(node_label,0)
    return node_label[svs]


def save_networkx_graph(g,fi):
    graph_json = nx.readwrite.json_graph.node_link_data(g)
    str_dump = json.dumps(graph_json)
    with open(fi,'w') as f:
        f.write(str_dump)
    #print("Saved ",fp)

def load_networkx_graph(fi):
    with open(fi,'r') as f:
        json_graph = json.loads(f.read())
        return nx.readwrite.json_graph.node_link_graph(json_graph)



