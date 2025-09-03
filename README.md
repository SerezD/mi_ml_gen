
# A Mutual Information Perspective on Multiple Latent Variable Generative Models for Positive View Generation

This is the official github repo for the paper: A Mutual Information Perspective on Multiple Latent Variable Generative Models for Positive View Generation (Accepted at TMLR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2412.03453-red)](https://arxiv.org/abs/2412.03453)
<!--  [![WACV](https://img.shields.io/badge/WACV-2025-blue)](https://openaccess.thecvf.com/content/WACV2025/html/Serez_Pre-Trained_Multiple_Latent_Variable_Generative_Models_are_Good_Defenders_Against_WACV_2025_paper.html) -->

## INSTALLATION

```
# Dependencies Install 
conda env create --file environment.yml
conda activate mi_ml_gen

# package install (in development mode)
conda develop .
```

## MLVGMS REFERENCES AND PRE-TRAINED MODELS

### BigBiGAN

paper:  
github: 
pretrained model: 

### StyleGAN-2

paper:  
github: 
pretrained model: 

### NVAE 

paper: [NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898)  
github (official): [https://github.com/NVlabs/NVAE](https://github.com/NVlabs/NVAE)  
github (used implementation): [https://github.com/SerezD/NVAE-from-scratch](https://github.com/SerezD/NVAE-from-scratch)  
pretrained model:

## OBTAIN DATASETS

We load the pre-computed ffcv files for train, validation and testing at:  

## TRAIN $T_\mathbf{z}(\mathbf{z})$

Run: 

```
python ... 
```

The pre-trained models that we used in the experiments are available at:  


## TRAIN ENCODERS (SimCLR, SimSiam, Byol)

For the SimCLR results, we ...

The pre-trained models that we used in the experiments are available at:  

## TRAIN AND TEST LINEAR CLASSIFIERS

## NVAE ABLATIONS

## CONTINUOS SAMPLING

To compare performance speed and reproduce results of Figure X, run: 

```
...
```

## CITATION

```
@inproceedings{serez2025pretrained,
    author    = {Serez, Dario and Cristani, Marco and Del Bue, Alessio and Murino, Vittorio and Morerio, Pietro},
    title     = {A Mutual Information Perspective on Multiple Latent Variable Generative Models for Positive View Generation},
    booktitle = {Transaction on Machine Learning Research (TMLR},
    month     = {September},
    year      = {2025},
    pages     = {xxxx-xxxx}
}
```
