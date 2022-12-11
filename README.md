# Introduction
---
Implementation of **Federated Learning** (a.k.a collaborative learning) for **CSI data** transmission between UE (user equipment) and BS (base station).
**Federated learning** is a subpart of AI and Machine learning where the **Central model** (also refered as server model) is trained/improved using local (decentralized models) "user" models. Several implementations of Fed. learning is Amazon Alexa, Apple Siri, Google Keyboard and etc.

# CSI dataset
---
**CSI** or **Channel State Information**, in wireless communications is the known channel properties of a communication link between transmitter and receiver. This information describes how a signal propagates from the transmitter to the receiver and represents the combined effect of, for example, scattering, fading, and power decay with distance. The CSI makes it possible to adapt transmissions to current channel conditions, which is crucial for achieving reliable communication with high data rates in multiantenna systems.

I have sampled **CSI_DATASET** (you can find it in my repo) from https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset
Dataset description you can find on the website. Note that all 64 antennas are not in phase.
**CSI dataset** consist of multiple csi data samples for training and validation. Decentralized models have training dataset of size 17000 and validation dataset of size 1500 csi datasamples.

![alt text](https://github.com/Nurassyl-lab/Federated-Heterogeneous-CSI-Estimations/blob/main/pictures/csi_data_and_fft.png)

# Observations during simulations.
---
This part is not related to the project, and I do not implement any algorithms, methods, or ideas that I mention in this section.
These are just my failed ideas.



main2 python code smaller copy of the main file which is not yet included in this repository.
You can use main2 to get familiar with our setup and etc.

Right now i'm still running simulations, I will add more infomation regarding this project later.
