## CGNN: Conduit Graph Neural Network for Link Prediction in Biomedical Networks

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

You should place dataset in corresponding folder:

`\\heterogeneous network\\lncRNA-disease\\`: files related with lncRNA-disease network;

`\\heterogeneous network\\micRNA-disease\\`: files related with micRNA-disease network;

`\\homogeneous network\\PPI\\`: files related with PPI network;

`\\homogeneous network\\DDI\\`: files related with DDI network;

All paths are relative paths
### Running code
The codes consist of **Module of CGNN** and **Example**.
About **Module of CGNN**, you can use the codes (`.py` files) of every module to construct your own model or CGNN by yourself.
About **Example**,  we give two standard codes (`.ipynb` files) for two kinds of datasets: heterogeneous network data and homogeneous network data. You can download the two ipynb files and click on 'Run' button to run them in Jupyter notebook.
#### Module of CGNN:
As presented in **Materials and Methods** of the paper, the modules and base functions (`.py`) are listed.
* `python node_learning.py` contains the code of node learning in CGNN, which provides node embedding for conduit node learning.
* `python conduit_node_learning.py` contains the code of conduit node learning in CGNN, which computes latent embedding of link in the form of nodes. If the dimension of output is 1, the module perform classification for conduit nodes.
* `python conduitGNN.py` is the framework of CGNN, which fuses conduit node embedding of all layers into the final output for classification.
* `python base_function.py` contains the function related with CGNN to compute acc, loss, etc.
* `python initialize_input_and_model.py` shows the details about input of CGNN for different datasets and how to initialize CGNN, optimizer and loss function.
#### Example:
According to the experiments on heterogeneous network data and homogeneous network data in paper, we present them with examples (`.ipynb`).
* `CGNN for lncRNA-disease network.ipynb` is the example for heterogeneous network data. If you want to use micRNA-disease network for experiment, you should change the path  to be `.\\heterogeneous network\\micRNA-disease\\` and keep codes invariable in `.ipynb` file except file name of input.
* `CGNN for PPI network.ipynb` is the example for homogeneous network data. If you want to use DDI network for experiment, you should change the path to be `\\homogeneous network\\DDI\\` and keep codes invariable in `.ipynb` file except `graph_num` and file name of input.


The results of each example can be seen in last cell of notebook. If you want to output result, you can refer to my notes of last cell in `.ipynb`file.
### Help
Please send any questions you might have about the code and/or the algorithm to [kangchuanze@mail.nankai.edu.cn](kangchuanze@mail.nankai.edu.cn).

