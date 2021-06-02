# FairBatch: Batch Selection for Model Fairness

#### Authors: Yuji Roh, Kangwook Lee, Steven Euijong Whang, Changho Suh
#### In Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021
----------------------------------------------------------------------

This directory is for simulating FairBatch on the synthetic dataset.
The program needs PyTorch and Jupyter Notebook.

The directory contains a total of 4 files and one child directory: 
1 README, 2 python files, 1 jupyter notebook, 
and the child directory containing 6 numpy files for synthetic data.


#### To simulate FairBatch, please use the jupyter notebook in the directory.

The jupyter notebook will load the data and train the models with three 
different fairness metrics: equal opportunity, equalized odds, and demographic parity.

Each training utilizes the FairBatch sampler, which defines in FairBatchSampler.py.
The pytorch dataloader serves the batches to the model via the FairBatch sampler. 
Experiments are repeated 10 times each.
After the training, the test accuracy and fairness will be shown.

The two python files are models.py and FairBatchSampler.py.
The models.py contains a logistic regression architecture and a test function.
The FairBatchSampler.py contains two classes: CustomDataset and FairBatch. 
CustomDataset class defines the dataset, and FairBatch class implements 
the algorithm of FairBatch as described in the paper.

The detailed explanations about each component have been written 
in the codes as comments.
Thanks!
