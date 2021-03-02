Overview
========


Simple neural network with activations defined on subgroups.

* Free software: MIT license

Installation
============

Clone the repository and run 
    
```{bash}
    pip install -e .
```


Training networks
================

Training scripts are located in the `src/subgroupnn/experiments` folder. The training scripts automatically attempt to download the required datasets,
and run the model.

```{bash}
python -um subgroupnn.experiments.train_combo_on_zinc dataset=zinc_subset model=subset_128
```

