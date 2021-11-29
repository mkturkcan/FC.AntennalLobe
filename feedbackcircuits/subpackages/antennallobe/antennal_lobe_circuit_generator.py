import networkx as nx
import numpy as np
import json 

import sys
sys.path.append("...")

G = nx.read_gexf('hemibrain_AL_ctype_level.gexf')
print('Cell type graph has '+str(len(G.nodes()))+' nodes.')

def get_glom_nodes(G, target_glom):
    """Retrieve nodes for a glomerulus.
    
    # Arguments
    G: NetworkX graph to retrieve nodes from.
    target_glom: Name of the fruit fly olfactory system glomerulus to retrieve the nodes of.
    
    # Returns
    list: List of nodes to use.
    """
    subnodes = []
    for i in G.nodes():
        if (('PNs' in i) or ('ORNs' in i)) and target_glom+' ' not in i:
            pass
        else:
            subnodes.append(i)
    return subnodes

def construct_subcircuit(G, nodes):
    """Construct feedback subcircuit given a set of nodes.
    
    # Arguments
    G: NetworkX graph to retrieve nodes from.
    nodes: Name of the fruit fly olfactory system glomerulus to retrieve the nodes of.
    
    # Returns
    G: The subcircuit consisting of the neurons in a list, their successors, and their predecessors.
    """
    allmininodes = list(set(nodes))
    relnodes = []
    for i in allmininodes:
        relnodes += list(G.predecessors(i)) + list(G.successors(i))
    relnodes = list(set(relnodes))
    relnodes = [i for i in relnodes if i.isnumeric()]
    GwLNs = G.subgraph(relnodes).copy()
    return GwLNs

def filter_G_edgeweights(G, synapse_count):
    """Filter the edges in a graph.
    
    # Arguments
    G: NetworkX graph to filter.
    synapse_count: Synapse number filter to use.
    
    # Returns
    G: The modified graph.
    """
    edges_to_remove = []
    for i in G.edges(data=True):
        if i[2]['weight'] < synapse_count:
            edges_to_remove.append([i[0],i[1]])
    for i in edges_to_remove:
        G.remove_edge(i[0],i[1])
    for i in G.nodes():
        G.nodes()[i]['label'] = i
    return G