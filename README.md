Not finished yet!!!
DATASET will be added soon

The repo may not be well organized. This is my first big project the file management is a bit chaotic ;(

# Content:
  1) Introduction
  2) CSI dataset
  3) Code description
  4) My observations

# Introduction
---
Implementation of **Federated Learning** (a.k.a collaborative learning) for **CSI data** transmission between UE (user equipment) and BS (base station).
**Federated learning** is a subpart of AI and Machine learning where the **Central model** (also refered as server model) is trained/improved using local (decentralized models) "user" models. Several implementations of Fed. learning is Amazon Alexa, Apple Siri, Google Keyboard and etc.

# CSI dataset
---
**CSI** or **Channel State Information**, in wireless communications is the known channel properties of a communication link between transmitter and receiver. This information describes how a signal propagates from the transmitter to the receiver and represents the combined effect of, for example, scattering, fading, and power decay with distance. The CSI makes it possible to adapt transmissions to current channel conditions, which is crucial for achieving reliable communication with high data rates in multiantenna systems.

I have sampled **unbalanced_dataset** (i will add it soon) from https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset

Dataset description you can find on the website. Note that all 64 antennas are not in phase.
**CSI dataset** consist of multiple csi data samples for training and validation. Decentralized models have training dataset of size 17000 and validation dataset of size 1500 csi datasamples.

Visualization of csi data sample.
Plot of 1 data sample 64x100 matrix on the left and it's 2D DFT on the right. As you can see a 2D DFT is very sparse.
![data1](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/csi_data_and_fft.png)

Plot of signal's magnitude in dB over 100 subcarriers of 1 antennas.
![data2](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/mag_subcarriers_data_over_1_antenna.png)

Plot of signal's phase in degrees over 100 subcarriers of 1 antennas.
![data2](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/phase_subcarriers_data_over_1_antenna.png)

# Code description
---
for now only variational autoencoder simultion is available.
Use VAE_simulation.py to train decentralized models. 

In this project autoencoder architecture is implemented in order to compress the CSI data and transmit it from UE to BS.
The goal of the project is to compress the data as much as possible and reach smallest reconstruction loss on the server.
However, there is one detail that must be mentioned, it is data heterogeneity. Remember that each local model is trained on its own dataset. This is done because in real life there won't be perfectly balanced UE dataset. (So, we also explore how biasness affects model performances)
Data heterogeneity describes the biasness of the data, and heterogeneity level or (epsilon) is variable that describes the data heterogeneity in %.

![data2](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/heterogeeity_level.png)

In our case we split dataset into 3 classes using L2-norm.

  CLASS 1: csi data with l2-norm value in interval [40, 45]
  
  CLASS 2: csi data with l2-norm value in interval [50, 55]
  
  CLASS 3: csi data with l2-norm value in interval [60, 65]

### Setup 1: UE selection
As for now only [setup 1] is available.
In this case pre-trained encoders are delivered to UE. When UE receives signal that it needs to send to BS the following algorithm is being executed under setup 1. 
UE compresses input using all 3 encoders that it has received. Now UE has 3 different encoded features of


main2 python code smaller copy of the main file which is not yet included in this repository.
You can use main2 to get familiar with our setup and etc.

Right now i'm still running simulations, I will add more infomation regarding this project later.

# My observations
---
Graveyard of failed ideas. 
