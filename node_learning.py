import torch
import torch.nn as nn
import torch.nn.functional as F

'''
For heterogeneous network data (lncRNA-disease, micRNA-disease),
we need two different sets of weight matrix
to learn node embedding for gene(lncRNA, micRNA) and disease.

For homogeneous network data (PPI, DDI), 
we need two different sets of weight matrix to learn node embedding for 
nodes with degrees that are less than or equal to 1 and nodes with larger degrees.

The specific usage of 'package' parameter of 'forward' function is shown in 
example jupyter notebook of CGNN.
'''

class node_learning(nn.Module):
    def __init__(self, in_size, out_size):
        super(node_learning, self).__init__()

        self.nei_update1 = nn.Linear(in_size,
                                     out_size,
                                     bias=False)
        
        self.self_node_update1 = nn.Linear(in_size,
                                           out_size,
                                           bias=False)

    def forward(self,
                out_size,
                package,
                old_node_embedding,
                active_method=F.relu,
                use_divce):
        
        total_node = old_node_embedding.shape[0]
        
        if use_divce == 1:
            
            new_node_embedding = torch.zeros((total_node, out_size)).cuda()
            
        else:
            
            new_node_embedding = torch.zeros((total_node, out_size)).cpu()

        for node in range(total_node):
            
            nei_information, nei_gather = 0, 0
            
            self_information, total_information = 0, 0

            self_information += self.self_node_update1(
                        old_node_embedding[node, :])
                    
            nei_gather += torch.mm(package[1][node, :].reshape(1, -1),
                                           old_node_embedding)
                    
            nei_information += self.nei_update1(nei_gather)
                    
            total_information += active_method(nei_information +
                                                       self_information)
                
            new_node_embedding[node,:] = torch.add(new_node_embedding[node,:],total_information)
            
        return new_node_embedding