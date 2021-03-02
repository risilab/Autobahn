Overview
========

This repository code for the paper __Autobahn: Automorphism-based Graph Neural Networks__.
In contrast to most graph neural networks where neurons are associated with individual graph vertices,
in Autobahn neurons correspond to subgraphs.
Each neuron then applies convolutions that are equivariant to each subgraph's automorphism group, 
followed by a fixed nonlinearity.
Ensuring that neurons are equivariant to the local automorphism group 
is enough to make the entire network is equivariant to permutation of the input.

<img width="100%" src="https://raw.githubusercontent.com/risilab/Autobahn/master/single_layer.png" />

In our code, we construct an Autobahn network where neurons correspond to paths and cycles within each graph.
The resulting neurons reflect our domain knowledge of chemical graphs and preserve the intuitive meaning of convolution on sequential data.

* Free software: MIT license

Installation
============

Clone the repository and run 
    
```{bash}
    pip install -e .
```


Training networks
================

Training scripts are located in the `src/autobahn/experiments` folder. The training scripts automatically attempt to download the required datasets,
and run the model.

```{bash}
python -um autobahn.experiments.train_combo_on_zinc dataset=zinc_subset model=subset_128
```

