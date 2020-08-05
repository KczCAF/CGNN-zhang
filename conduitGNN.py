import torch
import torch.nn as nn

'''
We select homogeneous networks as example:
The assignment of size is as follows:
    input_size:256
    size1:128
    size2:64
    size3:32
    
For homogeneous networks, we build three-layer CGNN.
    
self.gather is weight matrix of layer-wise updating rule.

The output of CGNN is prediction probability of sample.

The structure of node embedding is matrix to restore feature of one node.

    
For heterogeneous networks, we build three-layer CGNN.

The assignment of size is as follows:
    input_size:256
    size1:128
    size2:64
    size3:32
    
self.gather is weight matrix of layer-wise updating rule.

The output of CGNN is prediction probability of sample.

The structure of node embedding is list to restore feature of two different nodes.
'''

'''
For polypharmacy side effect dataset, we build two-layer CGNN.

The assignment of size is as follows:
    self.node_update_layer1 = graph_node_update(input_size=256,size1=128)
        
    self.conduit_layer1 = conduit_update_layer(size1=128,size2=964)
        
    self.node_update_layer2 = graph_node_update(size1=128,size3=64)
        
    self.conduit_layer2 = conduit_update_layer(size3=64,size2=964)
    
    self.gather_1 = nn.Linear(size2,size2,bias=True)
'''


class conduitGNN(nn.Module):
    
    def __init__(self, input_size, size1, size2, size3):
        
        super(conduitGNN,self).__init__()
        
        self.node_update_layer1 = node_learning(input_size,size1)
        
        self.conduit_layer1 = conduit_node_learning(size1,size2)
        
        self.node_update_layer2 = node_learning(size1,size2)
        
        self.conduit_layer2 = conduit_node_learning(size2,size3)
        
        self.node_update_layer3 = node_learning(size2,size3)
        
        self.conduit_layer3 = conduit_node_learning(size3,1)
        
        self.gather_1 = nn.Linear(size2,size3,bias=True)
        
        self.gather_2 = nn.Linear(size3,1,bias=True)
        
    def forward(self,
                pre_node_embedding,
                package,
                use_divce):
        
        use_divce = use_divce
        
        if use_divce == 1:
            
            output = torch.zeros((len(package[2]),1)).cuda()
            
            pre_node_embedding = pre_node_embedding.cuda()
            
        else:
            
            output = torch.zeros((len(package[2]),1)).cpu()
            
            pre_node_embedding = pre_node_embedding.cpu()
            
        ##########
        node_embedding_1 = self.node_update_layer1(128,
                                                   package,
                                                   pre_node_embedding,
                                                   use_divce)
        
        conduit_embedding_1 = self.conduit_layer1(package,
                                                  64,
                                                  node_embedding_1,
                                                  use_divce)

        ##########
        node_embedding_2 = self.node_update_layer2(64,
                                                   package,
                                                   node_embedding_1,
                                                   use_divce)
        
        conduit_embedding_2 = self.conduit_layer2(package,
                                                  32,
                                                  node_embedding_2,
                                                  use_divce)
        
        #########
        gather_conduit1 = torch.sigmoid(self.gather_1(conduit_embedding_1)+conduit_embedding_2)
        
        #########
        node_embedding_3 = self.node_update_layer3(32,
                                                   package,
                                                   node_embedding_2,
                                                   use_divce)
        
        conduit_embedding_3 = self.conduit_layer3(package,
                                                  1,
                                                  node_embedding_3,
                                                  use_divce)
        
        #########
        gather_conduit_embedding = torch.sigmoid(self.gather_2(gather_conduit1)+conduit_embedding_3)
        
        #########
        output += gather_conduit_embedding
        
        return output

