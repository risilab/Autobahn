Overview
========

This repository code for the paper [Autobahn: Automorphism-based Graph Neural Network](https://arxiv.org/abs/2103.01710).
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

Pre-trained models
==================

We provide pre-trained models for the datasets studied in the paper at the links below. After downloading a checkpoint for a given dataset,
you may test the downloaded model by running:
```{bash}
python -um autobahn.experiments.test_combo_on_zinc checkpoint=/path/to/your/checkpoint
```

Alternatively, the testing scripts will automatically download a checkpoint from our website if none is specified.
You can then specify the dataset configuration to select which dataset to test on. For example, to test on zinc subset
run the following command:
```{bash}
python -um autobahn.experiments.test_on_zinc data.use_subset=True
```
and to test on molpcba use the following command:
```{bash}
python -um autobahn.experiments.test_on_ogb data.data_name=ogbg-molpcba
```

| Dataset | Checkpoint |
|---------|------------|
| Zinc (subset) | [Checkpoint](https://users.flatironinstitute.org/~wzhou/autobahn/checkpoints/zinc_subset.ckpt) |
| Zinc (full)   | [Checkpoint](https://users.flatironinstitute.org/~wzhou/autobahn/checkpoints/zinc_full.ckpt)   |
| ogb-molpcba   | [Checkpoint](https://users.flatironinstitute.org/~wzhou/autobahn/checkpoints/molpcba.ckpt)     |
| ogb-molmuv    | [Checkpoint](https://users.flatironinstitute.org/~wzhou/autobahn/checkpoints/molmuv.ckpt)      |
| ogb-molhiv    | [Checkpoint](https://users.flatironinstitute.org/~wzhou/autobahn/checkpoints/molhiv.ckpt)      |

