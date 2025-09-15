
# A Mutual Information Perspective on Multiple Latent Variable Generative Models for Positive View Generation

This is the official github repo for the paper: A Mutual Information Perspective on Multiple Latent Variable Generative Models for Positive View Generation (Accepted at TMLR)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2501.13718-red)](https://arxiv.org/abs/2501.13718) -->
<!--  [![TMLR](https://img.shields.io/badge/journal-TMLR-blue.svg)](URL_HERE) -->

## INSTALLATION

```
# Dependencies Install 
conda env create --file environment.yml
conda activate mi_ml_gen

# package install (in development mode)
conda develop ./mi_ml_gen
```

## MLVGMS REFERENCES AND PRE-TRAINED MODELS

### BigBiGAN

original paper: [Large Scale Adversarial Representation Learning](https://arxiv.org/abs/1907.02544)  
used pretrained model (pytorch): [https://github.com/lukemelas/pytorch-pretrained-gans](https://github.com/lukemelas/pytorch-pretrained-gans)

### StyleGAN-2

paper: [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) 
github (official-tensorflow): [https://github.com/NVlabs/stylegan2](https://github.com/NVlabs/stylegan2)
pretrained model (official-pytorch): [https://github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

<!--
### NVAE 

paper: [NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898)  
github (official): [https://github.com/NVlabs/NVAE](https://github.com/NVlabs/NVAE)  
github (used implementation): [https://github.com/SerezD/NVAE-from-scratch](https://github.com/SerezD/NVAE-from-scratch)  
-->

## DATASETS

For training the "real-data" encoders, we use datasets in the ffcv format. 
We load the precomputed files for ImageNet-1K and LSUN Cars at: [https://huggingface.co/SerezD/mi_ml_gen/tree/main/datasets](https://huggingface.co/SerezD/mi_ml_gen/tree/main/datasets)   

Datasets for downstream tasks can be generated with the script at `mi_ml_gen/data/create_image_beton_file.py`


## TRAIN $T_\mathbf{z}(\mathbf{z})$ and Monte Carlo Simulation

To train walkers, deciding which latents (chunks) to perturb, check the script at: `mi_ml_gen/src/noise_maker/cop_gen_training/train_navigator.py` 

For example, to train the walker on all latents except the first of bigbigan, run:
```
python train_navigator.py --generator bigbigan --g_path ~/runs/bigbigan/BigBiGAN_x1.pth --chunks 1_6
```

_Note: learning rates for walkers and InfoNCE loss may vary depending on selected chunks, generator, and batch size. A rule of thumb is to keep them low (1e-5 order of magnitude), allowing smooth learning of the walkers._

The pre-trained models that we used in the experiments are available at: [https://huggingface.co/SerezD/mi_ml_gen/tree/main/runs/walkers/](https://huggingface.co/SerezD/mi_ml_gen/tree/main/runs/walkers/)  

To generate Table 1 in the paper (Monte Carlo simulation), run the script at `mi_ml_gen/src/noise_maker/delta_estimation/monte_carlo_simulation.py`

## TRAIN ENCODERS (SimCLR, SimSiam, Byol)

For the SimCLR results, we follow the experimental procedure of previous work, available at: [https://github.com/LiYinqi/COP-Gen/tree/master](https://github.com/LiYinqi/COP-Gen/tree/master)

Therefore, this repo containst only the code for training _SimSiam_ and _Byol_ models, generating data with continuous sampling.  
The configuration `.yaml` file for each model, containing training hyperparameters, can be found in the `mi_ml_gen/configurations/encoders` path.

To train a new encoder from scratch, run the script: `mi_ml_gen/src/multiview_encoders/train_encoder.py`.  
For example, to train _SimSiam_ encoder on _Imagenet-1K_ dataset (real data), run:

```
python mi_ml_gen/src/multiview_encoders/train_encoder.py --seed 0 --encoder simsiam --conf simsiam_bigbigan/encoder_imagenet_baseline_real --data_path /home/user/datasets/imagenet/ffcv/ --logging
```

_Note: please check the script file for a description of each argument._

Pre-trained encoder models that we used in the experiments are available at: [https://huggingface.co/SerezD/mi_ml_gen/tree/main/runs/encoders/](https://huggingface.co/SerezD/mi_ml_gen/tree/main/runs/encoders/) 

## TRAIN AND TEST LINEAR CLASSIFIERS

<!-- ## NVAE ABLATIONS -->

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
    booktitle = {Transaction on Machine Learning Research (TMLR)},
    month     = {September},
    year      = {2025}
}
```

<!--
```
@inproceedings{serez2025pretrained,
    author    = {Serez, Dario and Cristani, Marco and Del Bue, Alessio and Murino, Vittorio and Morerio, Pietro},
    title     = {A Mutual Information Perspective on Multiple Latent Variable Generative Models for Positive View Generation},
    booktitle = {Transaction on Machine Learning Research (TMLR)},
    month     = {September},
    year      = {2025},
    pages     = {xxxx-xxxx}
}
```
-->
