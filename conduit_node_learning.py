import torch
import torch.nn as nn

'''
For heterogeneous network data, we need three different sets of weight matrix
to learn embedding for three conduit types.

For homogeneous network data, we need two different sets of weight matrix
to learn embedding for two conduit types.

The specific usage of 'package' parameter of 'forward' function is shown in 
example jupyter notebook of CGNN.
'''

class conduit_node_learning(nn.Module):
    
    def __init__(self,pre_size,next_size):
        
        super(conduit_node_learning,self).__init__()
        
        self.left_gate1 = nn.Linear(pre_size,next_size,bias=False)
        
        self.right_gate1 = nn.Linear(pre_size,next_size,bias=False)

        self.conduit_update1 = nn.Linear(pre_size,next_size,bias=True)
        
    def forward(self,
                package,
                next_size,
                node_embedding,
                #node_embedding1,
                #node_embedding2,
                use_divce):
        
        conduit_sample = package[2]
        
        index = list(range(len(conduit_sample)))
        
        if use_divce == 1:
            
            new_conduit_embedding = torch.zeros((len(conduit_sample),next_size)).cuda()
            
        else:
            
            new_conduit_embedding = torch.zeros((len(conduit_sample),next_size)).cpu()
            
        for j in index:
            
            i = conduit_sample[j]
            
            left_gate,right_gate,conduit_embedding = 0,0,0
            
            combined_feature,gather_information = 0,0
            
            gather_information += torch.add(node_embedding[i[0],:],node_embedding[i[1],:])
#            gather_information += torch.add(node_embedding1[i[0],:],node_embedding2[i[1],:])
                
            left_gate += torch.sigmoid(self.left_gate1(node_embedding[i[0],:]))
#             left_gate += torch.sigmoid(self.left_gate1(node_embedding1[i[0],:]))
                
            right_gate += torch.sigmoid(self.right_gate1(node_embedding[i[1],:]))
#             right_gate += torch.sigmoid(self.right_gate1(node_embedding2[i[1],:]))
                
            combined_feature += torch.tanh(self.conduit_update1(gather_information))

            conduit_embedding += left_gate*combined_feature + right_gate*combined_feature
            
            new_conduit_embedding[j,:] += conduit_embedding.reshape(-1)
            
        return new_conduit_embedding