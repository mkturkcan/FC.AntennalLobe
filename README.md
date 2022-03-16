# FC.AntennalLobe

The functional logic of brain circuits of Drosophila neuropils is largely determined by local/intrinsic neurons. Detailed connectomics datasets such as Hemibrain reveal, in many neuropils, a massive number of nested feedback loops among their input, output and local neurons. Dissecting the role of these feedback circuits is key to the understanding the computation taking place in these neuropils. 

We introduce a circuit library for exploring the functional logic of the massive number of feedback loops (motifs) in the fruit fly brain. This library:
1. Provides tools for interactively visualizing and exploring the feedback loops in the antennal lobe (AL);
2. Enables users to instantiate an executable circuit of the feedback circuit model
3. Contains loading, visualization and analysis functions to explore the I/O behavior of executed circuits.

## Installation

We use pre-cached data obtained via FlyBrainLab queries on the Hemibrain dataset for faster circuit generation.

* You will need to have a working Neurokernel/Neurodriver installation so that the models can be executed efficiently on GPUs. This is provided in the FlyBrainLab docker images; alternatively, refer to [here](https://github.com/neurokernel/neurodriver).
* Download supporting files for Hemibrain for fast in-memory processing [here](https://drive.google.com/drive/u/0/folders/1HlgpnZLQCwkwjeOOuV7SD2ndWkVsr21F).
* Put the components in the NDComponents folder to your Neurodriver NDComponents folders. These constitute the neuron, odorant transduction and synapse models we use for antennal lobe.

## How to Use

Whereas this library was designed to work with FlyBrainLab, it can also be used independently to specifically generate models of the antennal lobe and run them.

* The ALGen.ipynb example shows how to generate specific AL circuits.
* The ALRun.ipynb example shows how to run specific AL circuits.
