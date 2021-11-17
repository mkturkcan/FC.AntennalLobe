import pickle
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances


def interaction_weights_default(x,y):
    eps = 1e-6
    A = pairwise_distances(x, y)
    B = 1. / (A+eps)
    return B

def interaction_weights_simple(x,y):
    eps = 1e-6
    A = pairwise_distances(x, y)
    B = 1. / (A+eps)
    return B * 0. + 1.

def interaction_weights_alt(x,y):
    eps = 1e-6
    A = pairwise_distances(x, y)
    B = 1. / (A+eps)
    return B

def interaction_filter_default(B,x,y):
    return B[x,y]>=0.5

def interaction_activation_default(x):
    return x

def simple_circuit_scaler(synapses_a, interaction_group, x, y):
    return synapses_a[x][y].shape[0]

def spiking_circuit(cell_groups, synapse_groups, neuron_models, synapse_models, interaction_models, 
                    interaction_weights = interaction_weights_default, 
                    interaction_filter = interaction_filter_default, 
                    interaction_activation = interaction_activation_default, 
                    name='default',
                    n_data_dict = {}):
    """Generates a general single-synapse-scale spiking circuit."""
    G = nx.MultiDiGraph()
    visual_neurons = {}
    for cell_group in cell_groups:
        if cell_group in neuron_models:
            for i in cell_groups[cell_group]:
                neuron_models[cell_group](G, i)
                if i in n_data_dict:
                    visual_neurons[i] = n_data_dict[i]

    visual_components = []

    for cell_group_a in cell_groups:
        for cell_group_b in cell_groups:
            synapse_group = cell_group_a + '-' + cell_group_b
            if synapse_group in synapse_groups:
                synapses = synapse_groups[synapse_group]
                if synapse_group in synapse_models:
                    for i in cell_groups[cell_group_a]:
                        for j in cell_groups[cell_group_b]:
                            if i in synapses:
                                if j in synapses[i]:
                                    for k in range(synapses[i][j].shape[0]):
                                        visual_component = synapse_models[synapse_group](G, i, j, k)
                                        if visual_component is not None:
                                            visual_component.append(synapses[i][j][k])
                                            visual_component.append(synapse_group)
                                            visual_components.append(visual_component)

    n_interactions = 0
    interaction_numbers = {}
    affected_units = {}
    for cell_group_a in cell_groups:
        for cell_group_b in cell_groups:
            for cell_group_c in cell_groups:
                interaction_group = cell_group_a + '-' + cell_group_b + '-' + cell_group_c
                interaction_numbers[interaction_group] = {}
                affected_units[interaction_group] = {'0': {}, '1': {}}
                if interaction_group in interaction_models:
                    synapses_a = synapse_groups[cell_group_a + '-' + cell_group_b]
                    synapses_b = synapse_groups[cell_group_b + '-' + cell_group_c]
                    for x in cell_groups[cell_group_a]:
                        interaction_numbers[interaction_group][x] = {}
                        for y in cell_groups[cell_group_b]:
                            interaction_numbers[interaction_group][x][y] = {'kept': 0, 'removed': 0}
                            for z in cell_groups[cell_group_c]:
                                # n_interactions = 0
                                if y in synapses_a[x] and z in synapses_b[y]:
                                    if y not in affected_units[interaction_group]['1']:
                                        affected_units[interaction_group]['1'][y] = {}
                                    if z not in affected_units[interaction_group]['1'][y]:
                                        affected_units[interaction_group]['1'][y][z] = np.zeros((synapses_b[y][z].shape[0],))
                                    if x not in affected_units[interaction_group]['0']:
                                        affected_units[interaction_group]['0'][x] = {}
                                    if y not in affected_units[interaction_group]['0'][x]:
                                        affected_units[interaction_group]['0'][x][y] = np.zeros((synapses_a[x][y].shape[0],))
                                    B = interaction_weights(synapses_a[x][y], synapses_b[y][z])
                                    for u in range(synapses_a[x][y].shape[0]):
                                        for v in range(synapses_b[y][z].shape[0]):
                                            if interaction_filter(B,u,v):
                                                visual_component = interaction_models[interaction_group](G, x, y, z, u, v, interaction_activation(B[u,v]))
                                                if visual_component is not None:
                                                    visual_component.append((synapses_a[x][y][u]+synapses_b[y][z][v])/2)
                                                    visual_component.append(interaction_group)
                                                    visual_components.append(visual_component)
                                                n_interactions += 1
                                                interaction_numbers[interaction_group][x][y]['kept'] += 1
                                                affected_units[interaction_group]['1'][y][z][v] = 1
                                                affected_units[interaction_group]['0'][x][y][u] = 1
                                            else:
                                                interaction_numbers[interaction_group][x][y]['removed'] += 1
                                                
                                            
    print('#Nodes:', len(G.nodes()))
    print('#Interactions:', n_interactions)
    for i,v in G.nodes(data=True):
        if 'class' not in v:
            print('Error:',i)
    nx.write_gexf(G, '{}.gexf'.format(name))
    np.save('visual_components_{}'.format(name), visual_components)
    np.save('visual_neurons_{}'.format(name), visual_neurons)
    return interaction_numbers, affected_units
    
    

def spiking_circuit_simple(cell_groups, synapse_groups, neuron_models, synapse_models, interaction_models, 
                           interaction_weights = interaction_weights_default, 
                           interaction_filter = interaction_filter_default, 
                           interaction_activation = interaction_activation_default, 
                           name='default', circuit_scaler=simple_circuit_scaler, syn_filter = 10, n_data_dict={}):
    """Generates a simple spiking circuit."""
    G = nx.MultiDiGraph()
    visual_neurons = {}
    for cell_group in cell_groups:
        if cell_group in neuron_models:
            for i in cell_groups[cell_group]:
                neuron_models[cell_group](G, i)
                if i in n_data_dict:
                    visual_neurons[i] = n_data_dict[i]

    visual_components = []

    for cell_group_a in cell_groups:
        for cell_group_b in cell_groups:
            synapse_group = cell_group_a + '-' + cell_group_b
            if synapse_group in synapse_groups:
                synapses = synapse_groups[synapse_group]
                if synapse_group in synapse_models:
                    for i in cell_groups[cell_group_a]:
                        for j in cell_groups[cell_group_b]:
                            if i in synapses:
                                if j in synapses[i]:
                                    # for k in range(synapses[i][j].shape[0]):
                                    if synapses[i][j].shape[0]>=syn_filter:
                                        visual_component = synapse_models[synapse_group](G, i, j, 0, gain=synapses[i][j].shape[0])
                                        if visual_component is not None:
                                            visual_component.append(synapses[i][j][0])
                                            visual_component.append(synapse_group)
                                            visual_components.append(visual_component)

    n_interactions = 0
    for cell_group_a in cell_groups:
        for cell_group_b in cell_groups:
            for cell_group_c in cell_groups:
                interaction_group = cell_group_a + '-' + cell_group_b + '-' + cell_group_c
                if interaction_group in interaction_models:
                    synapses_a = synapse_groups[cell_group_a + '-' + cell_group_b]
                    synapses_b = synapse_groups[cell_group_b + '-' + cell_group_c]
                    for x in cell_groups[cell_group_a]:
                        for y in cell_groups[cell_group_b]:
                            for z in cell_groups[cell_group_c]:
                                # n_interactions = 0
                                if y in synapses_a[x] and z in synapses_b[y]:
                                    B = interaction_weights(synapses_a[x][y], synapses_b[y][z])
                                    BB = B.copy()
                                    if synapses_a[x][y].shape[0]>=syn_filter and synapses_b[y][z].shape[0]>=syn_filter:
                                        for u in range(1):
                                            for v in range(1):
                                                visual_component = interaction_models[interaction_group](G, x, y, z, u, v, circuit_scaler(synapses_a, interaction_group, x, y) * synapses_b[y][z].shape[0] )
                                                n_interactions += 1


    print('#Nodes:', len(G.nodes()))
    print('#Interactions:', n_interactions)
    for i,v in G.nodes(data=True):
        if 'class' not in v:
            print('Error:',i)
    nx.write_gexf(G, '{}.gexf'.format(name))
    np.save('visual_components_{}'.format(name), visual_components)
    np.save('visual_neurons_{}'.format(name), visual_neurons)





class SCC:
    """A class that implements the concept of an SCC."""
    def __init__(self, synapse_models, interaction_models, inputs = None, outputs = None):
        self.synapse_models = synapse_models
        self.interaction_models = interaction_models
        self.inputs = inputs
        self.outputs = outputs
        
class SPU:
    """A class that implements the concept of an SPU."""
    def __init__(self):
        self.neuron_models = {}
        self.synapse_models = {}
        self.interaction_models = {}
        self.SCCs = []
    def update_neuron_models(self, neuron_models):
        self.neuron_models.update(neuron_models)
    def add(self, SCC):
        self.synapse_models.update(SCC.synapse_models)
        self.interaction_models.update(SCC.interaction_models)
        self.SCCs.append(SCC)
