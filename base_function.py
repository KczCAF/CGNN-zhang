import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def get_node_degree(adj_matrix,
                    node):
    
    return torch.sum(adj_matrix[node,:])

'''
For heterogeneous networks and homogeneous networks, 
the function to compute acc and recall is as follows:
'''
def Com_acc(output,
            lab):

    result = output.ge(0.5).float() == lab.reshape(-1,1)
    
    acc = result.float().mean()
    
    return acc

def Com_recall(output,
               lab):

    pred = output.ge(0.5).float()
    
    pred = pred.reshape(-1)
    
    posi_index = np.where(np.array(lab)==1)[0]
    
    posi_pred,posi_label = np.array(pred)[posi_index],np.array(lab)[posi_index]
    
    recall = np.sum(posi_pred == posi_label,dtype = np.float64)/(posi_index).shape[0]
    
    return recall

'''
For regularization function, we select heterogeneous network data as example.

For homogeneous network data, we select weight matrix of node_learning and conduit_node_learning for regularization (see CGNN for PPI network.ipynb).
'''

def regu(lambda1, lambda2):
    
    reg_loss = 0
    
    for name,param in conduit_GNN.named_parameters():
        
        if 'fuse' in name and 'weight' in name:
            
            l2_reg = torch.norm(param,p=2)
            
            reg_loss += lambda1*l2_reg
            
        if 'conduit_update' in name and 'weight' in name:
            
            l2_reg = torch.norm(param,p=2)
            
            reg_loss += lambda2*l2_reg
            
    return reg_loss
'''
The approach to computing AUC score is as follows:
test_label denotes the labels of test samples
y_proba denotes the predicted probability
'''
auc = roc_auc_score(test_label,y_proba)