# Energy-Based Test Sample Adaptation for Domain Generalization

This repository provides the official PyTorch implementation of our ICLR 2023 paper:    

> Energy-Based Test Sample Adaptation for Domain Generalization
> 
> Zehao Xiao, Xiantong Zhen, Shengcai Liao, Cees G. M. Snoek

For more dtails, please check out our [<ins>**paper**</ins>](https://openreview.net/pdf?id=3dnrKbeVatv). 

## Overview
This repository contains the implementation of our method for domain generalization with sample adaptation
using a discriminative energy-based model. 

## Prerequisites

### Hardware

This implementation is for the single-GPU configuration, evaluated on an NVIDIA Tesla V100. 

### Environments 
The code is tested on PyTorch 1.13.1. 

### Datasets 

We mainly evaluate the method on [PACS](https://domaingeneralization.github.io/#page-top), 
[Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), 
[DomainNet](https://ai.bu.edu/M3SDA/), 
and [PHEME](https://www.kaggle.com/datasets/usharengaraju/pheme-dataset) datasets.

Change ```data_path``` in ```ebm_dataset.py``` to your own data path.

## Training

You can run the code for training our method by running the following command:

```bash
python ebmdg_main.py --dataset PACS --test_domain art_painting --model ebmz --log_dir pacs_art --step_lr 20 --num_steps 50 --gpu 0 --net res50 --reslr 0.1
```

## Citation
If you find our code useful or our work relevant, please consider citing: 
```
@inproceedings{
xiao2023energy,
title={Energy-Based Test Sample Adaptation for Domain Generalization},
author={Xiao, Zehao and Zhen, Xiantong and Liao, Shengcai and Snoek, Cees GM},
booktitle={The Eleventh International Conference on Learning Representations}
year={2023},
}
```

## Acknowledgements
We thank the authors of [Improved Contrastive Divergence Training of Energy-Based Model](https://arxiv.org/pdf/2012.01316) for their open-source implementation on the energy-based model. 

