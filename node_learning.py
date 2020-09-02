import torch
import torch.nn as nn

'''
For heterogeneous networks (lncRNA-disease, micRNA-disease),
we need two different sets of parameter matrix
to learn node feature for gene(lncRNA, micRNA) and disease.
We select sigmoid as active_method.

For homogeneous networks (PPI, DDI), there exist nodes with the degree equal to 1,
which means that it has no neighbor node or it has just one neighbor node.
We need two different sets of parameter matrix
to learn node feature for nodes with 1 degree and nodes with larger degree.
We select sigmoid as active_method.

For polypharmacy side effect dataset, we just need one sets of parameter matrix
to learn feature for drug nodes.
We select ReLu as active_method.

nei_update is weight matrix W_Ne for neighbor nodes.

self_node_update is weight matrix W_0 for node itself.

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
                active_method,
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