 def setup_interactive_antennal_lobe():
    my_client = fbl.get_client()

    with open('G_auto.svg', 'r') as file:
        data = file.read()

    my_client.tryComms({'widget':'GFX', 
                        'messageType': 'loadCircuitFromString', 
                        'data': {'string':data, 'name':'custom'}})

    # Run the submodule called onGraphVizLoad2 to make the diagram interactive
    my_client.tryComms({'widget':'GFX', 
                        'messageType': 'eval', 
                        'data': {'data':"window.fbl.loadSubmodule('data/FBLSubmodules/onGraphVizLoadFree.js');", 'name':'custom'}})

    with open('additional_conn.js','r') as f:
        additional_data = f.read()
        
    my_client.tryComms({'widget':'GFX', 
                        'messageType': 'eval', 
                        'data': {'data':additional_data, 'name':'custom'}})