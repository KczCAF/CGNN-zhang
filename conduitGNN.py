import torch
import torch.nn as nn

'''
Each layer of CGNN is consituted by node learning and conduit node learning.

'''

class each_layer_of_conduitGNN(nn.Module):
    
    def __init__(self, in_size, out_size1, out_size2):
        super(each_layer_of_conduitGNN,self).__init__()
        
        self.node_update_layer = node_learning(in_size, out_size1)
        self.conduit_layer = conduit_node_learning(in_size, out_size2)
        
    def forward(self,
               package,
               pre_node_embedding,
               use_divce):
        
        node_embedding = self.node_update_layer(out_size1,
                                               package,
                                               pre_node_embedding,
                                               use_divce)
        conduit_embedding = self.conduit_layer(package,
                                              out_size2,
                                              node_embedding,
                                              use_divce)
        return conduit_embedding

'''
We build three-layer CGNN for heterogeneous network data and homogeneous network data.

For node learning, the dimensions of weight matrix are $F_0 = 256$, $F_1 = 128$, $F_2 = 64$, $F_3 = 32$. 
For conduit node learning, the dimensions of weight matrix are $F'_0=128$, $F'_1 = 64$, $F'_2 = 32$, $F'_3 = 1$.
For layer-wise fusing rules, $F'_{t=0} = 64$, $F'_{t=1} = 32$, $F'_{t=2} = 1$

The output of CGNN is predicted probability.
The framework of three-layer CGNN is shown as follows:

    input_size = 256
    size1 = 128
    size2 = 64
    size3 = 32
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
        
        self.fuse_1 = nn.Linear(size2,size3,bias=True)
        
        self.fuse_2 = nn.Linear(size3,1,bias=True)
        
    def forward(self,
                pre_node_embedding,
                package,
                use_divce,
                output_thred):
        
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
        fused_embedding1 = torch.sigmoid(self.fuse_1(conduit_embedding_1)+conduit_embedding_2)
        
        
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
        fused_embedding2 = torch.sigmoid(self.fuse_2(fused_embedding1)+conduit_embedding_3)
        
        #########
        output += fused_embedding2
        
        if output_thred == 1:
            
            output_dict = {}
            output_dict['conduit_embedding_1'] = conduit_embedding_1
            output_dict['conduit_embedding_2'] = conduit_embedding_2
            output_dict['fused_embedding1'] = fused_embedding1
            return output_dict, output
        
        else:
            
            return output

