## CGNN: Conduit Graph Neural Network for Link Prediction on Biomedical Networks

### Overview
This repository contains codes necessary to run the CGNN algorithm. 

### Running Environment
* Windows environment, Python 3
* PyTorch >= 1.3.1
* Jupyter notebook

### Datasets
* All datasets are available at [CGNNdataset](http://bioinfo.nankai.edu.cn/kangcz.html).

* `python initialize_input_and_model.py` shows the details about input of CGNN for different datasets.
*  The examples of CGNN show how to use datasets.
* You should place dataset in corresponding folder:
`\\heterogeneous network\\lncRNA-disease network\\`: files related with lncRNA-disease network;
`\\heterogeneous network\\micRNA-disease network\\`: files related with micRNA-disease network;
`\\homogeneous network\\PPI\\`: files related with PPI network;
`\\homogeneous network\\DDI\\`: files related with DDI network;
`\\polypharmacy side effect dataset\\`: files related with polypharmacy side effect dataset;
All paths are relative paths
### Running code
The codes consist of **Module of CGNN** and **Example**.
About **Module of CGNN**, you can use the codes (`.py` files) about every module of CGNN to construct your own model or CGNN by yourself.
About **Example**,  we give three standard codes (`.ipynb` files) for three kinds of datasets: heterogeneous networks, homogeneous networks and polypharmacy side effect dataset. You can download the three ipynb files and click on 'Run' button to run them in Jupyter notebook.
#### Module of CGNN:
As presented in **Section 2** of the paper, the modules and base functions (`.py`) are listed.
* `python node_learning.py` contains the node learning module of CGNN, which generate latent feature of node for conduit node learning.
* `python conduit_node_learning.py` contains the conduit node learning module of CGNN, which generate latent feature of conduit node. If the dimension of output is 1, the module perform classification for conduit node.
* `python conduitGNN.py` is the framework of CGNN, which uses node learning and conduit node learning.
* `python base_function.py` contains the function of CGNN to compute acc, loss, etc.
* `python initialize_input_and_model.py` shows the details about input of CGNN for different datasets and how to initialize CGNN, optimizer and loss function.
#### Example:
According to the experiments on heterogeneous networks, homogeneous networks and polypharmacy side effect in paper, we present them with examples (`.ipynb`).
* `CGNN for lncRNA-disease network.ipynb` is the example for heterogeneous networks. If you want to use micRNA-disease network for experiment, you should change the path  to be `.\\heterogeneous network\\micRNA-disease\\` and keep codes invariable in .ipynb file except file name of input.
* `CGNN for PPI network.ipynb` is the example for homogeneous networks. If you want to use DDI network for experiment, you should change the path to be `\\homogeneous network\\DDI\\` and keep codes invariable in .ipynb file except `graph_num` and file name of input.
* `CGNN for polypharmacy side effect dataset.ipynb` is the example for polypharmacy side effect dataset. You should run the codes with memory larger than 32G.

The results of each example can be seen in last cell of notebook. If you want to output result, you can refer to my notes of last cell.
### Help
Please send any questions you might have about the code and/or the algorithm to [kangchuanze@mail.nankai.edu.cn](kangchuanze@mail.nankai.edu.cn).

