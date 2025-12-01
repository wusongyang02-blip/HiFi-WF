# Source code and datasets for our paper "HiFi-WF: Toward Realistic Website Fingerprinting with Multi-tab and Subpage Recognition".
The usage of this model's code is as follows, and the dataset can be obtained via [the shared link](https://drive.google.com/file/d/1v86rGzmXOrV2tAGfNvCzhTyi69bZNSUv/view?usp=sharing) in Google Drive Cloud.

## Basic Environment:
GPU: NVIDIA GeForce RTX 4060 (8GB)  
Python Version: 3.12  
PyTorch Version: 2.5.0  
CUDA Version: 12.4

## Description & Useage

### Install
You can use conda commands in the virtual environment provided by Anaconda to install the basic packages mentioned in the code that need to be loaded (such as numpy, os, sys, etc.), whereas pip commands do not need to be used within a conda virtual environment.

### Datasets Collection
You can use `traffic_collection.py` to obtain traffic data. We use the **tshark** tool for automatic packet capture. And as shown in the code, we visit web pages in the order of "HomepageA-SubpageA1-HomepageB-SubpageB1". This only requires setting 2 filters, but in reality, it involves capturing mixed traffic from 4 web pages. When labeling, each piece of traffic data contains hierarchical labels of "2 main labels and 2 sub-labels".

In the 3-tab scenario, each visit is conducted in the order of "HomepageA-SubpageA1-HomepageB-SubpageB1--HomepageC-SubpageC1". This requires setting 3 filters, and in reality, it involves mixed traffic from 6 web pages; the labels are also 3 main labels and 3 sub-labels.

Our Tor traffic data was collected using the same methodology as in the paper “Beyond Single Tabs: A Transformative Few-Shot Approach to Multi-Tab Website Fingerprinting Attacks” (published in WWW'25). The difference is that we collected traffic from the 2-tab scenario that includes subpages, as we described above.

For details about the datasets, you can refer to our paper.

### Datasets Details
Our datasets are built on 50 monitored websites (1 homepage per website, 10 subpages per website), forming 500 unique **"homepage-subpage pairs"** (e.g., "Homepage A + Subpage A1", "Homepage B + Subpage B1").   This yields 50 main labels (one per website) and 500 sub-labels (one per pair).  Thus, Sub-labels are much sparser compared to main labels(most sub-labels are 0 for any sample).

Each 2-tab browsing sequence is structured as: [Homepage-Subpage Pair 1] → [Homepage-Subpage Pair 2], where Pair 1 and Pair 2 belong to different websites (no overlapping main labels).   We construct the two datasets as follows:

**SP-MA(Single-Permutation, Multi-Access) Datasets:** 

We perform random sampling, assigning each of the 500 pairs as Pair 1, and matching each **Pair 1** with a **Pair 2** from a different website (i.e., with a distinct main label).  This way, we obtain 500 unique 2-tab sequences.  Each sequence is accessed repeatedly 20 times, generating 10,000 traffic samples—which constitutes the SP-MA dataset for the 2-tab scenario.  

If we further randomly sample a **Pair 3** that has distinct main labels from both **Pair 1** and **Pair 2**, and perform accesses following the sequence [Homepage-Subpage Pair 1] → [Homepage-Subpage Pair 2] → [Homepage-Subpage Pair 3], this is the construction method of the SP-MA dataset for the 3-tab scenario.

**MP-SA(Multi-Permutation, Single-Access) Datasets:**

We repeat the process used to generate the 500 unique 2-tab sequences in the SP-MA dataset 20 times, which allows us to obtain nearly non-repetitive 2-tab sequences totaling 10,000. Each sequence is accessed once, resulting in the MP-SA dataset for the 2-tab scenario. This dataset is specifically designed to evaluate the model's capability against multi-tab WF attacks involving mixed subpage traffic, in scenarios where subpages are not identified.

**Open-World Dataset:**

We selected 12,000 websites that are not included in our monitored website list. We then generated 12,000 unique access sequences following the same access sequence generation method, but at the 4-tab level—for instance, sequences in the form of **"Website A → Website B → Website C → Website D"**—and these sequences are also generated through random sampling. Furthermore, when visiting these websites, we randomly open some of their subpages, while others are not. This design increases traffic complexity and better aligns with users' real-world browsing behaviors.

### Modle Training & Evaluation
We loaded the `Chrome_MP_SA_2tab.npz` dataset and use the `website_level_train_and_test.py` to train and evaluate the HiFi WF model. In this scenario, the model only needs to identify the websites corresponding to the mixed traffic, rather than specific webpages.

We loaded the `Chrome_SP_MA_2tab.npz`, `Chrome_SP_MA_3tab.npz`, and `Tor_SP_MA_2tab datasets`, and use the `webpage_level_train_and_test.py` to train and evaluate the HiFi WF model. In this scenario, the model needs to identify the homepages and corresponding subpages of monitored websites, hence there are main label metrics and sub-label metrics.

We used `open_world.py` to load the `open_world_train.npz` dataset for training the model, and load the `open_world_temp.npz` dataset for validating and testing the model.

For the closed-world scenario, we used the model that achieved the best F1 score on the validation set for testing; for the open-world scenario, we used the model that achieved the best AUCM on the validation set for testing. Additionally, the optimal parameters used after multiple rounds of experiments are annotated in the code.
