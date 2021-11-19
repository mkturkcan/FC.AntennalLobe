import numpy as np
import networkx as nx
import seaborn as sns

G = nx.read_gexf('hemi12_G_normalized.gexf')
neurons = np.load('hemi_neurons.npy', allow_pickle=True).item()
def search(inp):
    return [x for x,v in neurons.items() if inp in v]

def merge_nodes(G, nodes, new_node, node_dict = None, text = 'nan/nan/nan/nan/nan/nan', **attr):
    """ Utility function for merging a number of nodes into one in networkx in place.
    
    # Arguments:
        G (nx graph): A networkx graph.
        nodes (list): List of nodes to combine.
        new_node (str): Name of the new node.
        attr (dict): Optional. Additional attributes for the new node.
    """
    G.add_node(new_node, **attr) # Add node corresponding to the merged nodes
    edge_iterator = list(G.edges(data=True))
    for n1,n2,data in edge_iterator:
        if n1 in nodes:
            if G.has_edge(new_node,n2):
                w = data['weight']
                G[new_node][n2]['weight'] += w
            else:
                G.add_edge(new_node,n2,**data)
        elif n2 in nodes:
            if G.has_edge(n1,new_node):
                w = data['weight']
                G[n1][new_node]['weight'] += w
            else:
                G.add_edge(n1,new_node,**data)
    
    for n in nodes: # remove the merged nodes
        if n in G.nodes():
            G.remove_node(n)
    node_dict[new_node] = text
    return G, node_dict


def get_presynaptic(X,threshold=0.05):
    predecessors = []
    for x in X:
        if x in G.nodes():
            for i in G.predecessors(x):
                if G.edges[i, x]['weight']>=threshold:
                    predecessors.append(i)
    predecessors = list(set(predecessors))
    return predecessors

def get_search(names, verb = 'show'):
    _str = verb + ' /:referenceId:['+', '.join([str(i) for i in names])+']'
    print(_str)
    return _str

def get_set_color(names, color):
    _str = 'color /:referenceId:['+', '.join([str(i) for i in names])+'] '+color
    print(_str)
    return _str

def get_presynaptic_weighted(X,threshold=0.05):
    predecessors = {}
    for x in X:
        if x in G.nodes():
            for i in G.predecessors(x):
                if G.edges[i, x]['weight']>=threshold:
                    if i in predecessors:
                        predecessors[i] += G.edges[i, x]['weight']
                    else:
                        predecessors[i] = G.edges[i, x]['weight']
    predecessors = dict(sorted(predecessors.items(), key=lambda item: item[1], reverse=True))
    return predecessors

def get_field(X, i=0):
    return [neurons[x].split('/')[i] for x in X]

def get_field_with_keys(X, i=0):
    return [(x,neurons[x].split('/')[i]) for x in X]

def get_top(X, K=10):
    i = 0
    returned = []
    for x,v in X.items():
        returned.append(x)
        i += 1
        if i == K:
            return returned
    return returned

def discard_field(X, string, i=0):
    if isinstance(X, list):
        return [x for x in X if string not in neurons[x].split('/')[i]]
    elif isinstance(X, dict):
        return_dict = {}
        for x,v in X.items():
            if string not in neurons[x].split('/')[i]:
                return_dict[x] = v
        return return_dict


def keep_field(X, string, i=0):
    if isinstance(X, list):
        return [x for x in X if string in neurons[x].split('/')[i]]
    elif isinstance(X, dict):
        return_dict = {}
        for x,v in X.items():
            if string in neurons[x].split('/')[i]:
                return_dict[x] = v
        return return_dict
    
def threshold_graph(G, threshold=0.05):
    edge_iterator = list(G.edges(data=True))
    edges_to_remove = []
    for n1,n2,data in edge_iterator:
        # print(data)
        if data['weight']<threshold:
            edges_to_remove.append((n1,n2))
    for i in edges_to_remove:
        G.remove_edge(i[0],i[1])
    return G


Go = nx.read_gexf('hemi12off_light.gexf')
neuronso = np.load('hemi_neurons.npy', allow_pickle=True).item()


def get_paths_order(name, G, order, a = None, b = None):
    """ Utility function for finding paths.
    
    # Arguments:
        name (str): Start node.
        G (nx graph): A networkx graph.
        order (int): Maximum hop number.
        a (str): Start node.
        b (str): End node.
    """
    paths=[]
    if order == 0:
        return [name], []
    desc = list(G.successors(name))
    desc2 = desc.copy()
    for i in desc:
        # print(i)
        if i == b:
            a_first = a.split('->')[0]
            # print(b, i, i.count(b))
            if (a+'->').count(a_first)<2 and (a+'->').count(b)==0:
                aa = a.split('->')
                ba = list(set(aa))
                if len(aa) == len(ba):
                    # print('Path:',a+'->',b)
                    paths.append(a+'->'+b)
        else:
            descs3, paths2 = get_paths_order(i, G, order-1, a = a + '->'+i, b=b)
            
            desc2 += descs3
            paths += paths2
    
    return desc2, paths

"""
# Example:
LC4s = search('LC4/')
all_paths = []
all_desc = []
for i in LC4s:
    for j in ORNs:
        for k in ORNs[j]:
            if k in G.nodes():
                desc, paths = get_paths_order(k, G, 3, a = k, b = i)
                all_desc += desc
                all_paths += paths
    print(len(paths))
"""

glom_names = ['VP1d', 'VM4', 'VA6', 'VL1', 'VA4', 'VA5', 'VM2', 'VM3', 'VM1', 'VA2', 'VA3', 'VM7v', 'VA7l', 'VA7m', 'DC4', 'DC3', 'DC2', 'VM7d', 'DC1', 'DL3', 'VP4', 'VM5v', 'VP5', 'DM6', 'D', 'DP1m', 'DM4', 'VL2p', 'DL1', 'DM5', 'DP1l', 'VP2', 'DM1', 'DL5', 'DA2', 'DA3', 'DL4', 'VP3', 'DM2', 'DA1', 'VL2a', 'VM5d', 'DM3', 'V', 'VP1m', 'VC3l', 'VC3m', 'VP1l', 'VC5', 'DL2d', 'VA1d', 'VC4', 'DL2v', 'DA4l', 'VA1v', 'DA4m', 'VC1', 'VC2']
glom_names.sort()
glom_names = glom_names[::-1]

def get_color_set(N):                                             
    t = np.linspace(-510, 510, N)                                              
    colors = np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)
    color_array = []
    for i in range(colors.shape[0]):
        color_array.append('#%02x%02x%02x' % (colors[i,0], colors[i,1], colors[i,2]))
    return color_array

all_ns = []
ORNs = {}
for glom_name in glom_names:
    RNs = search('ORN_'+glom_name+'_')+search('HRN_'+glom_name+'_')+search('TRN_'+glom_name+'_')
    ORNs[glom_name] = [i for i in RNs if i not in all_ns]
    all_ns += ORNs[glom_name]
    
PNs = {}
for glom_name in glom_names:
    PNs[glom_name] = search(glom_name+'_lPN')+search(glom_name+'_vPN')+search(glom_name+'_adPN')+search(glom_name+'_ilPN')+search(glom_name+'_l2PN')+search(glom_name+'_ivPN')#+search(glom_name+'+')
    
    
    
def list_diff(x,y):
    return [i for i in x if i not in y]
LNs = search('LN')
LNs_b = search('LN2')
LNs_c = search('lLN1')
LNs_d = search('v2LN')
LNs_not_a = search('-')
LNs_not_b = search('FB')
LNs_not_c = search('LAL')
LNs_not_d = search('LNd')
LNs_not_L = search('_L')
LNs_irrelevant = ['1699029767','1699707102','5813069064','1698347801','5813015889','5813054969','1671085318','1729382486','5813049168','1791797463','1729736878','1792492536']
LNs = list_diff(LNs, LNs_not_a)
LNs = list_diff(LNs, LNs_not_b)
LNs = list_diff(LNs, LNs_not_c)
LNs = list_diff(LNs, LNs_not_d)
LNs = LNs + LNs_b + LNs_c + LNs_d
LNs = list(set(LNs))
LNs = list_diff(LNs, LNs_not_L)
LNs = list_diff(LNs, LNs_irrelevant)
print(len(LNs))
print(get_field(LNs, 3))

all_neurons = LNs.copy()
for i in PNs.keys():
    all_neurons += PNs[i]
for i in ORNs.keys():
    all_neurons += ORNs[i]
    
LN_types_hemi = list(set([i for i in (get_field(LNs, 3)) if '?' not in i]))
LN_types_hemi.sort()
len(LN_types_hemi) # 77
mPNs = search('M_')
mPN_types_hemi = list(set([i for i in (get_field(mPNs, 3)) if '_L' not in i and i[:2]=='M_']))
mPN_types_hemi.sort()
len(mPN_types_hemi) # 93

all_neurons = LNs.copy()
for i in PNs.keys():
    all_neurons += PNs[i]
for i in ORNs.keys():
    all_neurons += ORNs[i]
    
CG = Go.subgraph(all_neurons).copy()
Cneurons = np.load('hemi_neurons.npy', allow_pickle=True).item()
for unique_type in ORNs.keys():
    vals =  ORNs[unique_type]# + PNs[unique_type]
    CG, Cneurons = merge_nodes(CG, vals, unique_type+' ORNs', Cneurons, text = 'type/type/type/type/type/type')
    vals =  PNs[unique_type]
    CG, Cneurons = merge_nodes(CG, vals, unique_type+' PNs', Cneurons, text = 'type/type/type/type/type/type')
    
    
"""
#Glom-level:

all_neurons = LNs.copy()
for i in PNs.keys():
    all_neurons += PNs[i]
for i in ORNs.keys():
    all_neurons += ORNs[i]
    
CG = Go.subgraph(all_neurons).copy()
Cneurons = np.load('hemi_neurons.npy', allow_pickle=True).item()
for unique_type in ORNs.keys():
    print(unique_type)
    vals =  ORNs[unique_type]# + PNs[unique_type]
    CG, Cneurons = merge_nodes(CG, vals, unique_type+' ORNs', Cneurons, text = 'type/type/type/type/type/type')
    vals =  PNs[unique_type]
    CG, Cneurons = merge_nodes(CG, vals, unique_type+' PNs', Cneurons, text = 'type/type/type/type/type/type')
    CG, Cneurons = merge_nodes(CG, [unique_type+' PNs', unique_type+' ORNs'], unique_type+' Glom', Cneurons, text = 'type/type/type/type/type/type')
"""



def extract_ln_features(gloms_to_keep = ['DC1','DL5'], fig_name = 'DC1_DL5', title = "Features of Local Neuron Cell Types for DM4 and DL5"):
    """Extract features for local neurons."""

    nodes_to_keep = []
    CG_mini = CG.copy()

    edges_to_remove = []
    for u,v,data in CG_mini.edges(data=True):
        if data['weight']<1: # used to be 25
            edges_to_remove.append((u,v))
    for i in edges_to_remove:
        CG_mini.remove_edge(i[0],i[1])

    # gloms_to_keep = ['DC1','DL5']
    for _type in gloms_to_keep:
        nodes_to_keep += list(CG_mini.predecessors(_type+' ORNs')) + list(CG_mini.successors(_type+' ORNs')) + list(CG_mini.successors(_type+' PNs')) + list(CG_mini.predecessors(_type+' PNs'))

    for _type in gloms_to_keep:
        nodes_to_keep += [_type+' ORNs'] + [_type+' PNs']
    nodes_to_keep = list(set(nodes_to_keep))

    nodes_to_keep_copy = nodes_to_keep.copy()
    for i in nodes_to_keep_copy:
        if 'ORNs' in i or 'PNs' in i:
            glom = i.split(' ')[0]
            print(glom)
            if glom not in gloms_to_keep:
                print('Removed', i)
                nodes_to_keep.remove(i)
            else:
                print('Kept', i)

    print('Final:')
    print(nodes_to_keep)
    #for _type in ['DC1','DL5']:
    #    nodes_to_keep += [_type+' ORNs'] + [_type+' PNs']
    CG_mini = CG_mini.subgraph(nodes_to_keep)
    for i in CG_mini.nodes():
        CG_mini.nodes[i]['label'] = i
    for i,j in CG_mini.edges():
        CG_mini.edges[i,j]['label'] = CG_mini.nodes[i]['label'] + '-' + CG_mini.nodes[j]['label']

    nx.write_graphml(CG_mini, '2gloms_{}.graphml'.format(fig_name))
    
    CG_for_connectivity = CG_mini.copy()
    
    A = nx.adjacency_matrix(CG_for_connectivity)

    CGnodes = list(CG_for_connectivity.nodes())
    is_ORN = [i for i in range(len(CGnodes)) if ' ORNs' in CGnodes[i]]
    is_PN = [i for i in range(len(CGnodes)) if ' PNs' in CGnodes[i]]
    is_LN  = [i for i in range(len(CGnodes)) if CGnodes[i] in LNs]
    ORN_names = [CGnodes[i] for i in range(len(CGnodes)) if ' ORNs' in CGnodes[i]]
    PN_names = [CGnodes[i] for i in range(len(CGnodes)) if ' PNs' in CGnodes[i]]
    LN_names = [CGnodes[i] for i in range(len(CGnodes)) if CGnodes[i] in LNs]
    is_PN = []
    PN_names = []
    for ORN_name in ORN_names:
        glom_name = ORN_name.split(' ')[0]
        try:
            is_PN.append(CGnodes.index(glom_name+' PNs'))
            PN_names.append(glom_name+' PNs')
        except:
            pass

    B = A[is_ORN,:][:,is_LN].todense()
    BR = A[is_LN,:][:,is_ORN].T.todense()
    print(B.shape)

    C = A[is_LN,:][:,is_PN].T.todense()
    CR = A[is_PN,:][:,is_LN].todense()
    D = A[is_LN,:][:,is_LN].T.todense()

    diff_M = np.abs((B>0)*1.-(C>0)*1.)

    diff_M = ((B>0)*1.-(C>0)*1.)

    # 7-8, 9-10
    group_features = ['Diff(ORN>LN,LN>PN)', 
                      'Diff(ORN>LN,LN>ORN)', 
                      'AND(ORN>LN,LN>ORN)', 
                      'AND(ORN>LN,LN>PN)',
                      '#ORN>LN Gloms',
                      '#LN>ORN Gloms',
                      '#LN>PN Gloms',
                      '#ORN>LN Syns','#LN>ORN Syns','#LN>PN Syns','#PN>LN Syns', '#LN>LN Syns', '#LN>LN Syns (I-O)', '#LN>LN Partners']
    len(group_features)
    groups = [np.sum(np.abs((B>0)*1.-(C>0)*1.), axis=0), # Diffs for OR>LN and LN>PN
    np.sum(np.abs((B>0)*1.-(BR>0)*1.), axis=0), # Diffs for OR>LN and LN>OR
    np.sum(np.multiply((B>0)*1.,(BR>0)*1.), axis=0), # Diffs for OR>LN and LN>OR
    np.sum(np.multiply((B>0)*1.,(C>0)*1.), axis=0),
    np.sum((B>0)*1., axis=0),
    np.sum((BR>0)*1., axis=0),
    np.sum((C>0)*1., axis=0),
    np.sum(C, axis=0),
    np.sum(CR, axis=0),
    np.sum(B, axis=0),
    np.sum(BR, axis=0), 
    np.sum(D, axis=0)+np.sum(D.T, axis=0), 
    np.sum(D, axis=0)-np.sum(D.T, axis=0), 
    np.sum((D>0)*1., axis=0),]

    from scipy.stats import zscore
    J = np.asarray(np.vstack(groups).T)
    O = zscore(J)

    filtered_LN_names = []
    for i in LN_names:
        for j in all_real_LN_names:
            if i in j:
                filtered_LN_names.append(j)
                break

    sns.clustermap(O, yticklabels = filtered_LN_names, xticklabels = group_features, figsize=(10,40), method='ward', cbar_pos = None)
    plt.title(title)
    plt.savefig('LN_feature_clusters_general_{}.pdf'.format(fig_name))
    return J, LN_names