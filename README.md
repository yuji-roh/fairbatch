# FairBatch: Batch Selection for Model Fairness

#### Authors: Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh
#### In Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021
----------------------------------------------------------------------

This repo contains codes used in the ICLR 2021 paper: [FairBatch: Batch Selection for Model Fairness](https://arxiv.org/abs/2012.01696)

*Abstract: Training a fair machine learning model is essential to prevent demographic disparity. Existing techniques for improving model fairness require broad changes in either data preprocessing or model training, rendering themselves difficult-to-adopt for potentially already complex machine learning systems. We address this problem via the lens of bilevel optimization. While keeping the standard training algorithm as an inner optimizer, we incorporate an outer optimizer so as to equip the inner problem with an additional functionality: Adaptively selecting minibatch sizes for the purpose of improving model fairness. Our batch selection algorithm, which we call FairBatch, implements this optimization and supports prominent fairness measures: equal opportunity, equalized odds, and demographic parity. FairBatch comes with a significant implementation benefit -- it does not require any modification to data preprocessing or model training. For instance, a single-line change of PyTorch code for replacing batch selection part of model training suffices to employ FairBatch. Our experiments conducted both on synthetic and benchmark real data demonstrate that FairBatch can provide such functionalities while achieving comparable (or even greater) performances against the state of the arts. Furthermore, FairBatch can readily improve fairness of any pre-trained model simply via fine-tuning. It is also compatible with existing batch selection techniques intended for different purposes, such as faster convergence, thus gracefully achieving multiple purposes.*


## Setting
This directory is for simulating FairBatch on the synthetic dataset.
The program needs PyTorch and Jupyter Notebook.

The directory contains a total of 4 files and 1 child directory: 
1 README, 2 python files, 1 jupyter notebook, 
and the child directory containing 6 numpy files for synthetic data.

## Simulation
To simulate FairBatch, please use the **jupyter notebook** in the directory.

The jupyter notebook will load the data and train the models with three 
different fairness metrics: equal opportunity, equalized odds, and demographic parity.

Each training utilizes the FairBatch sampler, which is defined in FairBatchSampler.py.
The pytorch dataloader serves the batches to the model via the FairBatch sampler. 
Experiments are repeated 10 times each.
After the training, the test accuracy and fairness will be shown.

## Other details
The two python files are models.py and FairBatchSampler.py.
The models.py file contains a logistic regression architecture and a test function.
The FairBatchSampler.py file contains two classes: CustomDataset and FairBatch. 
The CustomDataset class defines the dataset, and the FairBatch class implements 
the algorithm of FairBatch as described in the paper.

More detailed explanations of each component can be found in the code as comments.
Thanks!

## Demos using Google Colab
We also release Google Colab notebooks for fast demos.
You can access both the [PyTorch version](https://colab.research.google.com/drive/192tZmf-jXg1uesHW2TSqv0LoDbhAW4X1?usp=sharing) and the [TensorFlow version](https://colab.research.google.com/drive/1VBc7osg-wRKTKav32k1wY2yfKWK-wnDW?usp=sharing).

## Reference
```
@inproceedings{
roh2021fairbatch,
title={FairBatch: Batch Selection for Model Fairness},
author={Yuji Roh and Kangwook Lee and Steven Euijong Whang and Changho Suh},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=YNnpaAKeCfx}
}
```

