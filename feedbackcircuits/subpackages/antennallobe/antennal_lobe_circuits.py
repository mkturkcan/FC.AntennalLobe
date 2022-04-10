import pickle
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
import sys
sys.path.append("...")
from models.circuit_generators import *
def read_pickle(x):
    with (open(x, "rb")) as f:
        data = pickle.load(f)
    return data

X = np.load('al_synapses2.npy')
# al_res = np.load('res1_al.npy', allow_pickle=True)
n_data = np.load('al_references.npy')
n_data_dict = {}
for i in range(n_data.shape[0]):
    n_data_dict[n_data[i,1]] = n_data[i,0]

ORN_PN_pos = read_pickle('ORN_PN_pos.pickle')
ORN_LN_pos = read_pickle('ORN_LN_pos.pickle')
LN_PN_pos = read_pickle('LN_PN_pos.pickle')
PN_LN_pos = read_pickle('PN_LN_pos.pickle')
LN_ORN_pos = read_pickle('LN_ORN_pos.pickle')
LN_LN_pos = read_pickle('LN_LN_pos.pickle')

uniques = list(np.unique(X[:,3]))
print('uniques',len(uniques))
ORN_uniques = [i for i in uniques if 'ORN' in i]
PN_uniques = [i for i in uniques if 'PN' in i]
LN_uniques = [i for i in uniques if 'LN' in i]
print('ORN uniques',len(ORN_uniques))
print('PN uniques',len(PN_uniques))
print('LN uniques',len(LN_uniques))

def get_LNs(glom = 'DL5'):
    found_LNs = []
    for i in [i for i in ORN_uniques if glom in i]:
        for j in LN_uniques:
            if j in ORN_LN_pos[i]:
                found_LNs.append(j)
    for i in [i for i in PN_uniques if glom in i]:
        for j in LN_uniques:
            if j in PN_LN_pos[i]:
                found_LNs.append(j)
    for j in LN_uniques:
        for i in [i for i in ORN_uniques if glom in i]:
            if i in LN_ORN_pos[j]:
                found_LNs.append(j)
    for j in LN_uniques:
        for i in [i for i in PN_uniques if glom in i]:
            if i in LN_PN_pos[j]:
                found_LNs.append(j)
    found_LNs = list(set(found_LNs))
    return found_LNs
                
DL5_LNs = get_LNs('DL5')
DC1_LNs = get_LNs('DC1')

only_DL5_LNs = [i for i in DL5_LNs if i not in DC1_LNs]
only_DC1_LNs = [i for i in DC1_LNs if i not in DL5_LNs]
remnant_LNs = [i for i in LN_uniques if i not in only_DL5_LNs and i not in only_DC1_LNs]



def setup_spiking_SPU(ORN_PN_gain = 1., ORN_LN_gain = 1., LN_PN_gain = 1.):
    """Holds computational models and parameters for an antennal lobe model, and generates it using SPUs."""
    def gen_ORN(G, x):
        params = dict(
            br=1.0,
            dr=10.0,
            gamma=0.138,
            a1=45.0,
            b1=0.8,
            a2=199.574,
            b2=51.887,
            a3=2.539,
            b3=0.9096,
            kappa=9593.9,
            p=1.0,
            c=0.06546,
            Imax=150.159,
        )
        G.add_node(x+'_OTP', **{"class": "OTP"}, **params)

        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x+'_RN', **{"class": "NoisyConnorStevens"}, **params)
        G.add_edge(x+'_OTP', x+'_RN')

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=0.6/10000.*ORN_PN_gain,
        )
        G.add_node(x+'_ARN', **{"class": "Alpha"}, **params)
        G.add_edge(x+'_RN', x+'_ARN')

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=0.6/1000.*ORN_LN_gain,
        )
        G.add_node(x+'_ARN_LN', **{"class": "Alpha"}, **params)
        G.add_edge(x+'_RN', x+'_ARN_LN')

    def gen_PN(G, x):
        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x+'_PN', **{"class": "NoisyConnorStevens"}, **params)

    def gen_LN(G, x):
        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x, **{"class": "NoisyConnorStevens"}, **params)

    def ORN_PN_LN_ORN_interaction(G,x,y,z,i,j, dist_gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./100. * dist_gain,
        )
        G.add_node(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "Alpha"}, **params)
        G.add_edge(x, x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j))

        params = dict(
            dummy=0.0,
        )
        G.add_node(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "PreLN"}, **params) # LN>(LN>ORN)
        G.add_edge(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j))
        #G.add_node(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        #G.add_edge(y+'_ARN', x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j))
        #G.add_edge(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), z)
        G.add_edge(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), y+'_'+z+'_AT'+'_'+str(j)) # (LN>ORN) to (ORN>PN)
        return [x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), 'g']

    def ORN_ORN_LN_interaction(G,x,y,z,i,j, dist_gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./100. * dist_gain,
        )
        G.add_node(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "Alpha"}, **params)
        G.add_edge(x, x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j))

        params = dict(
            dummy=0.0,
        )
        G.add_node(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "PreLN"}, **params) # LN>(LN>ORN)
        G.add_edge(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j))
        #G.add_node(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        #G.add_edge(y+'_ARN', x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j))
        #G.add_edge(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), z)
        G.add_edge(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), y+'_'+z+'_AT'+'_'+str(j)) # (LN>ORN) to (ORN>PN)
        return [y+'_'+z+'_AT'+'_'+str(j), 'g']

    def gen_ORNPN_syn(G, x, y, i, gain=1.):
        params = dict(
            bias=1.0,
            gain=1.0,
        )
        G.add_node(x+'_'+y+'_AT'+'_'+str(i), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        G.add_edge(x+'_ARN', x+'_'+y+'_AT'+'_'+str(i))
        G.add_edge(x+'_'+y+'_AT'+'_'+str(i), y+'_PN')
        return [x+'_'+y+'_AT'+'_'+str(i), 'I']

    def gen_ORNLN_syn(G, x, y, i, gain=1.):
        params = dict(
            bias=1.0,
            gain=1.0,
        )
        G.add_node(x+'_'+y+'_AT'+'_'+str(i), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        G.add_edge(x+'_ARN_LN', x+'_'+y+'_AT'+'_'+str(i))
        G.add_edge(x+'_'+y+'_AT'+'_'+str(i), y)
        return [x+'_'+y+'_AT'+'_'+str(i), 'I']

    def gen_regsyn(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./10000.*gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x+'_PN', x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    def gen_regsyn_PN(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./10000.*gain*LN_PN_gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x, x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y+'_PN')
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    def gen_regsyn_LN(G, x, y, i, gain=1.):

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./1000.*gain,
        )

        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)

        params = dict(
            bias=1.0,
            gain=1.,
        )

        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x+'_RN', x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    neuron_models = {'ORNs': gen_ORN, 'PNs': gen_PN, 'LNs': gen_LN}
    synapse_models = {'ORNs-LNs': gen_ORNLN_syn, 'LNs-PNs': gen_regsyn_PN, 'PNs-LNs': gen_regsyn, 'ORNs-PNs': gen_ORNPN_syn}
    interaction_models = {'LNs-ORNs-PNs': ORN_PN_LN_ORN_interaction, 'LNs-ORNs-LNs': ORN_PN_LN_ORN_interaction}
    
    AL_SPU = SPU()
    glomerulus_SCC = SCC({'ORNs-LNs': gen_ORNLN_syn, 'ORNs-PNs': gen_ORNPN_syn}, {'LNs-ORNs-PNs': ORN_PN_LN_ORN_interaction, 'LNs-ORNs-LNs': ORN_PN_LN_ORN_interaction})
    LN_PN_SCC = SCC({'LNs-PNs': gen_regsyn_PN, 'PNs-LNs': gen_regsyn}, {})
    AL_SPU.add(glomerulus_SCC)
    AL_SPU.add(LN_PN_SCC)
    AL_SPU.add_neuron_models({'ORNs': gen_ORN, 'PNs': gen_PN, 'LNs': gen_LN})
    
    return AL_SPU


def setup_spiking_beta(ORN_PN_gain = 1., ORN_LN_gain = 1., LN_PN_gain = 1., PN_LN_gain = 1., interaction_gain = 1., LN_LN_gain=1.):
    """Holds computational models and parameters for an antennal lobe model 
    (for the beta release of the package)."""
    def gen_ORN(G, x):
        params = dict(
            br=1.0,
            dr=10.0,
            gamma=0.138,
            a1=45.0,
            b1=0.8,
            a2=199.574,
            b2=51.887,
            a3=2.539,
            b3=0.9096,
            kappa=9593.9,
            p=1.0,
            c=0.06546,
            Imax=150.159,
        )
        G.add_node(x+'_OTP', **{"class": "OTP"}, **params)

        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x+'_RN', **{"class": "NoisyConnorStevens"}, **params)
        G.add_edge(x+'_OTP', x+'_RN')

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=0.6/10000.*ORN_PN_gain,
        )
        G.add_node(x+'_ARN', **{"class": "Alpha"}, **params)
        G.add_edge(x+'_RN', x+'_ARN')

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=0.6/1000.*ORN_LN_gain,
        )
        G.add_node(x+'_ARN_LN', **{"class": "Alpha"}, **params)
        G.add_edge(x+'_RN', x+'_ARN_LN')

    def gen_PN(G, x):
        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x+'_PN', **{"class": "NoisyConnorStevens"}, **params)

    def gen_LN(G, x):
        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x, **{"class": "NoisyConnorStevens"}, **params)

    def ORN_PN_LN_ORN_interaction(G,x,y,z,i,j, dist_gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./100. * dist_gain * interaction_gain,
        )
        G.add_node(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "Alpha"}, **params)
        G.add_edge(x, x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j))

        params = dict(
            dummy=0.0,
        )
        G.add_node(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "PreLN"}, **params) # LN>(LN>ORN)
        G.add_edge(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j))
        #G.add_node(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        #G.add_edge(y+'_ARN', x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j))
        #G.add_edge(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), z)
        G.add_edge(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), y+'_'+z+'_AT'+'_'+str(j)) # (LN>ORN) to (ORN>PN)
        return [x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), 'g']

    def ORN_ORN_LN_interaction(G,x,y,z,i,j, dist_gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./100. * dist_gain * interaction_gain,
        )
        G.add_node(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "Alpha"}, **params)
        G.add_edge(x, x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j))

        params = dict(
            dummy=0.0,
        )
        G.add_node(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "PreLN"}, **params) # LN>(LN>ORN)
        G.add_edge(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j))
        #G.add_node(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        #G.add_edge(y+'_ARN', x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j))
        #G.add_edge(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), z)
        G.add_edge(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), y+'_'+z+'_AT'+'_'+str(j)) # (LN>ORN) to (ORN>PN)
        return [y+'_'+z+'_AT'+'_'+str(j), 'g']

    def gen_ORNPN_syn(G, x, y, i, gain=1.):
        params = dict(
            bias=1.0,
            gain=1. * gain,
        )
        G.add_node(x+'_'+y+'_AT'+'_'+str(i), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        G.add_edge(x+'_ARN', x+'_'+y+'_AT'+'_'+str(i))
        G.add_edge(x+'_'+y+'_AT'+'_'+str(i), y+'_PN')
        return [x+'_'+y+'_AT'+'_'+str(i), 'I']

    def gen_ORNLN_syn(G, x, y, i, gain=1.):
        params = dict(
            bias=1.0,
            gain=1. * gain,
        )
        G.add_node(x+'_'+y+'_AT'+'_'+str(i), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        G.add_edge(x+'_ARN_LN', x+'_'+y+'_AT'+'_'+str(i))
        G.add_edge(x+'_'+y+'_AT'+'_'+str(i), y)
        return [x+'_'+y+'_AT'+'_'+str(i), 'I']

    def gen_regsyn(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./10000.*gain*PN_LN_gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x+'_PN', x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    def gen_regsyn_PN(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./10000.*gain*LN_PN_gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x, x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y+'_PN')
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    def gen_regsyn_LN(G, x, y, i, gain=1.):

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./1000.*gain,
        )

        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)

        params = dict(
            bias=1.0,
            gain=1.,
        )

        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x+'_RN', x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']
    
    def gen_regsyn_LN2(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./10000.*gain*LN_LN_gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=-1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x, x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    neuron_models = {'ORNs': gen_ORN, 'PNs': gen_PN, 'LNs': gen_LN}
    synapse_models = {'ORNs-LNs': gen_ORNLN_syn, 
                      'LNs-PNs': gen_regsyn_PN, 'PNs-LNs': gen_regsyn, # Synaptic Feedback Loop
                      'ORNs-PNs': gen_ORNPN_syn, 
                      'LNs-LNs': gen_regsyn_LN2}
    interaction_models = {'LNs-ORNs-PNs': ORN_PN_LN_ORN_interaction, 
                          'LNs-ORNs-LNs': ORN_PN_LN_ORN_interaction} # Feedback Loop with Interactions
    return neuron_models, synapse_models, interaction_models

def setup_spiking_default(ORN_PN_gain = 1., ORN_LN_gain = 1., LN_PN_gain = 1., PN_LN_gain = 1., interaction_gain = 1., LN_LN_gain=1., exLN_LN_gain=1., LN_exLN_gain=1.):
    # Default setup for the ORN-PN-LN-ORN network
    def gen_ORN(G, x):
        params = dict(
            br=1.0,
            dr=10.0,
            gamma=0.138,
            a1=45.0,
            b1=0.8,
            a2=199.574,
            b2=51.887,
            a3=2.539,
            b3=0.9096,
            kappa=9593.9,
            p=1.0,
            c=0.06546,
            Imax=150.159,
        )
        G.add_node(x+'_OTP', **{"class": "OTP"}, **params)

        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x+'_RN', **{"class": "NoisyConnorStevens"}, **params)
        G.add_edge(x+'_OTP', x+'_RN')

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=0.6/10000.*ORN_PN_gain,
        )
        G.add_node(x+'_ARN', **{"class": "Alpha"}, **params)
        G.add_edge(x+'_RN', x+'_ARN')

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=0.6/1000.*ORN_LN_gain,
        )
        G.add_node(x+'_ARN_LN', **{"class": "Alpha"}, **params)
        G.add_edge(x+'_RN', x+'_ARN_LN')

    def gen_PN(G, x):
        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x+'_PN', **{"class": "NoisyConnorStevens"}, **params)

    def gen_LN(G, x):
        params = dict(
            ms=-5.3,
            ns=-4.3,
            hs=-12.0,
            gNa=120.0,
            gK=20.0,
            gL=0.3,
            ga=47.7,
            ENa=55.0,
            EK=-72.0,
            EL=-17.0,
            Ea=-75.0,
            sigma=0.00,
            refperiod=0.0,
        )
        G.add_node(x, **{"class": "NoisyConnorStevens"}, **params)

    def ORN_PN_LN_ORN_interaction(G,x,y,z,i,j, dist_gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./100. * dist_gain * interaction_gain,
        )
        G.add_node(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "Alpha"}, **params)
        G.add_edge(x, x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j))

        params = dict(
            dummy=0.0,
        )
        G.add_node(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "PreLN"}, **params) # LN>(LN>ORN)
        G.add_edge(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j))
        G.add_edge(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), y+'_'+z+'_AT'+'_'+str(j)) # (LN>ORN) to (ORN>PN)
        return [x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), 'g']

    def ORN_ORN_LN_interaction(G,x,y,z,i,j, dist_gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./100. * dist_gain * interaction_gain,
        )
        G.add_node(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "Alpha"}, **params)
        G.add_edge(x, x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j))

        params = dict(
            dummy=0.0,
        )
        G.add_node(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), **{"class": "PreLN"}, **params) # LN>(LN>ORN)
        G.add_edge(x+'_to_'+y+'_PreLNAlpha'+'_'+str(z)+'_'+str(i)+'_'+str(j), x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j))
        #G.add_node(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        #G.add_edge(y+'_ARN', x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j))
        #G.add_edge(x+'_'+y+'_to_'+z+'_AT'+'_'+str(i)+'_'+str(j), z)
        G.add_edge(x+'_to_'+y+'_PreLN'+'_'+str(z)+'_'+str(i)+'_'+str(j), y+'_'+z+'_AT'+'_'+str(j)) # (LN>ORN) to (ORN>PN)
        return [y+'_'+z+'_AT'+'_'+str(j), 'g']

    def gen_ORNPN_syn(G, x, y, i, gain=1.):
        params = dict(
            bias=1.0,
            gain=1. * gain,
        )
        G.add_node(x+'_'+y+'_AT'+'_'+str(i), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        G.add_edge(x+'_ARN', x+'_'+y+'_AT'+'_'+str(i))
        G.add_edge(x+'_'+y+'_AT'+'_'+str(i), y+'_PN')
        return [x+'_'+y+'_AT'+'_'+str(i), 'I']

    def gen_ORNLN_syn(G, x, y, i, gain=1.):
        params = dict(
            bias=1.0,
            gain=1. * gain,
        )
        G.add_node(x+'_'+y+'_AT'+'_'+str(i), **{"class": "OSNAxt2"}, **params) # ORN>(ORN>PN)
        G.add_edge(x+'_ARN_LN', x+'_'+y+'_AT'+'_'+str(i))
        G.add_edge(x+'_'+y+'_AT'+'_'+str(i), y)
        return [x+'_'+y+'_AT'+'_'+str(i), 'I']

    def gen_regsyn(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./10000.*gain*PN_LN_gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x+'_PN', x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    def gen_regsyn_PN(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./10000.*gain*np.abs(LN_PN_gain),
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=1.*np.sign(LN_PN_gain),
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x, x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y+'_PN')
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    def gen_regsyn_LN(G, x, y, i, gain=1.):

        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./1000.*gain,
        )

        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)

        params = dict(
            bias=1.0,
            gain=1.,
        )

        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x+'_RN', x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']
    
    def gen_regsyn_LN2(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./10000.*gain*LN_LN_gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=-1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x, x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']
    
    def gen_regsyn_LNex(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./1000000.*gain*exLN_LN_gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x, x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']
    
    def gen_regsyn_exLN(G, x, y, i, gain=1.):
        params = dict(
            ar=12.5,
            ad=12.19,
            gmax=1./1000000.*gain*LN_exLN_gain,
        )
        G.add_node(x+'_to_'+y+'_Alpha_'+str(i), **{"class": "Alpha"}, **params)
        params = dict(
            bias=1.0,
            gain=-1.,
        )
        G.add_node(x+'_to_'+y+'_Converter_'+str(i), **{"class": "OSNAxt2"}, **params)
        G.add_edge(x, x+'_to_'+y+'_Alpha_'+str(i))
        G.add_edge(x+'_to_'+y+'_Alpha_'+str(i), x+'_to_'+y+'_Converter_'+str(i))
        G.add_edge(x+'_to_'+y+'_Converter_'+str(i), y)
        return [x+'_to_'+y+'_Alpha_'+str(i), 'g']

    neuron_models = {'ORNs': gen_ORN, 'PNs': gen_PN, 'LNs': gen_LN, 'exLNs': gen_LN}
    synapse_models = {'ORNs-LNs': gen_ORNLN_syn, 'LNs-PNs': gen_regsyn_PN, 'PNs-LNs': gen_regsyn, 'ORNs-PNs': gen_ORNPN_syn, 'LNs-LNs': gen_regsyn_LN2, 'LNs-exLNs': gen_regsyn_LNex, 'exLNs-LNs': gen_regsyn_exLN}
    interaction_models = {'LNs-ORNs-PNs': ORN_PN_LN_ORN_interaction, 'LNs-ORNs-LNs': ORN_PN_LN_ORN_interaction}
    return neuron_models, synapse_models, interaction_models


def generate_simple_al():
    """Generates a simple antennal lobe circuit with two glomeruli (DM4 and DL5)."""
    cell_groups = {'ORNs': [i for i in ORN_uniques if 'DM4' in i]+[i for i in ORN_uniques if 'DL5' in i], 
                'PNs': [i for i in PN_uniques if 'DM4' in i and 'ad' in i] + [i for i in PN_uniques if 'DL5' in i], 
                "LNs": LN_uniques}
    synapse_groups = {'ORNs-PNs': ORN_PN_pos, 
                      'ORNs-LNs': ORN_LN_pos, 'LNs-ORNs': LN_ORN_pos, # Feedback Loop 1
                      'LNs-PNs': LN_PN_pos, 'PNs-LNs': PN_LN_pos} # Feedback Loop 2
    neuron_models, synapse_models, interaction_models = setup_spiking_default(ORN_LN_gain=10., ORN_PN_gain=50., LN_PN_gain=1e-2, interaction_gain=2e-3)

    spiking_circuit_simple(cell_groups, synapse_groups, neuron_models, synapse_models, interaction_models, name='DM4_DL5', syn_filter=5)


class AbstractCircuit:
    def __init__(self):
        """Initializes a circuit."""
        self.cell_types = {}
        self.synapse_types = {}
    def add_cell_type(self, cell_type_name, cells):
        """Adds a cell type to the circuit.

        # Arguments
            cell_type_name (str):
                Cell type name.
            cells (list of str):
                Names of the neurons in the connectomics/synaptomics dataset to model these neurons by.
        """
        self.cell_types[cell_type_name] = cells
    def add_feedforward_connection(self, presynaptic_cells, postsynaptic_cells, model):
        """Adds a model to model the feedforward connection with.

        # Arguments
            presynaptic_cells (str):
                Cell type name for the presynaptic neurons.
            postsynaptic_cells (str):
                Cell type name for the postsynaptic neurons.
            model (function):
                Model generating connection for the computational model to model the feedforward connection with.
        """
        self.synapse_types[presynaptic_cells+'-'+postsynaptic_cells] = model
    def add_feedback_connection(self, presynaptic_cells, postsynaptic_cells, forward_model, backward_model): 
        """Adds a model to model the feedforward connection with.

        # Arguments
            presynaptic_cells (str):
                Cell type name for the presynaptic neurons.
            postsynaptic_cells (str):
                Cell type name for the postsynaptic neurons.
            model (function):
                Model generating connection for the computational model to model the feedforward connection with.
        """
        self.synapse_types[presynaptic_cells+'-'+postsynaptic_cells] = forward_model
        self.synapse_types[postsynaptic_cells+'-'+presynaptic_cells] = backward_model
    

class Subregion:
    """Class defining a subregion/component in the brain."""
    def __init__(self, C, name):
        """Initializes a circuit.
        
        # Arguments
            C (AbstractCircuit class):
                Cell type name for the presynaptic neurons.
        """
        self.name = name
        self.cell_types = {}
        self.synapse_types = {}
        self.cell_definitions = {}
        self.connectivity_definitions = {}
        self.C = C
    def add_cell_type(self, cell_type_name, cells, local_type):
        self.C.add_cell_type(cell_type_name, cells)
        if local_type not in self.cell_definitions:
            self.cell_definitions[local_type] = cells
        if local_type in self.cell_definitions:
            self.cell_definitions[local_type] += cells
    def add_feedforward_connection(self, presynaptic_cells, postsynaptic_cells, model):
        """Adds a model to model the feedforward connection with.

        # Arguments
            presynaptic_cells (str):
                Cell type name for the presynaptic neurons.
            postsynaptic_cells (str):
                Cell type name for the postsynaptic neurons.
            model (function):
                Model generating connection for the computational model to model the feedforward connection with.
        """
        self.C.add_feedforward_connection(presynaptic_cells, postsynaptic_cells, model)
    def add_feedback_connection(self, presynaptic_cells, postsynaptic_cells, forward_model, backward_model):
        """Adds a model to model the feedforward connection with.

        # Arguments
            presynaptic_cells (str):
                Cell type name for the presynaptic neurons.
            postsynaptic_cells (str):
                Cell type name for the postsynaptic neurons.
            model (function):
                Model generating connection for the computational model to model the feedforward connection with.
        """
        self.C.add_feedforward_connection(presynaptic_cells, postsynaptic_cells, forward_model, backward_model)

class Glomerulus(Subregion):
    """Class defining a glomerulus component in the antennal lobe.
    """
    def __init__(self, *kwargs):
        super().__init__(*kwargs)

class Compartment(Subregion):
    """Class defining a compartment component in the mushroom body.
    """
    def __init__(self, *kwargs):
        super().__init__(*kwargs)

class Tract(Subregion):
    """Class defining a tract component in the lateral horn.
    """
    def __init__(self, *kwargs):
        super().__init__(*kwargs)


def generate_default_al():
    """Generates a simple antennal lobe circuit with two glomeruli (DM4 and DL5)."""
    C = AbstractCircuit()
    C.add_cell_type("LNs", LN_uniques)
    DM4 = Glomerulus(C, 'DM4 Glomerulus')
    DM4.add_cell_type('ORNs', 
                      [i for i in ORN_uniques if 'DM4' in i], 
                      'olfactory receptor neuron (ORN) that expresses Or59b')
    DM4.add_cell_type('PNs', 
                      [i for i in PN_uniques if 'DM4' in i and 'ad' in i], 
                      'Adult uniglomerular antennal lobe projection neuron with dendrites that mainly innervate antennal lobe glomerulus DM4')
    
    DL5 = Glomerulus(C, 'DL5 Glomerulus')
    DL5.add_cell_type('ORNs', 
                      [i for i in ORN_uniques if 'DL5' in i], 
                      'olfactory receptor neuron (ORN) that expresses Or59b')
    DL5.add_cell_type('PNs', 
                      [i for i in PN_uniques if 'DL5' in i], 
                      'Adult uniglomerular antennal lobe projection neuron with dendrites that mainly innervate antennal lobe glomerulus DL5')
    C.add_feedforward_connection('ORNs', 'PNs', ORN_PN_pos)
    C.add_feedback_connection('ORNs', 'LNs', ORN_LN_pos, LN_ORN_pos)
    C.add_feedback_connection('LNs', 'PNs', LN_PN_pos, PN_LN_pos)
    neuron_models, synapse_models, interaction_models = setup_spiking_default(ORN_LN_gain=10., ORN_PN_gain=50., LN_PN_gain=1e-2, interaction_gain=2e-3)

    spiking_circuit_simple(C.cell_types, C.synapse_types, 
                           neuron_models, synapse_models, interaction_models, name='DM4_DL5_model', syn_filter=5)
    return C, [DM4, DL5]