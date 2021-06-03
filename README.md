# FairBatch: Batch Selection for Model Fairness

#### Authors: Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh
#### In Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021
----------------------------------------------------------------------

This directory is for simulating FairBatch on the synthetic dataset.
The program needs PyTorch and Jupyter Notebook.

The directory contains a total of 4 files and 1 child directory: 
1 README, 2 python files, 1 jupyter notebook, 
and the child directory containing 6 numpy files for synthetic data.


#### To simulate FairBatch, please use the jupyter notebook in the directory.

The jupyter notebook will load the data and train the models with three 
different fairness metrics: equal opportunity, equalized odds, and demographic parity.

Each training utilizes the FairBatch sampler, which is defined in FairBatchSampler.py.
The pytorch dataloader serves the batches to the model via the FairBatch sampler. 
Experiments are repeated 10 times each.
After the training, the test accuracy and fairness will be shown.

The two python files are models.py and FairBatchSampler.py.
The models.py file contains a logistic regression architecture and a test function.
The FairBatchSampler.py file contains two classes: CustomDataset and FairBatch. 
The CustomDataset class defines the dataset, and the FairBatch class implements 
the algorithm of FairBatch as described in the paper.

More detailed explanations of each component can be found in the code as comments.
Thanks!
