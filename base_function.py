import numpy as np
import torch
import torch.nn.functional as F

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
For regularization function, we select heterogeneous networks as example.

For homogeneous networks and polypharmacy side effect dataset,
we select weight matrix in node_learning and conduit_node_learning module
for regularization.
'''

def regu(lambda1, lambda2):
    
    reg_loss = 0
    
    for name,param in conduit_GNN.named_parameters():
        
        if 'gather' in name and 'weight' in name:
            
            l2_reg = torch.norm(param,p=2)
            
            reg_loss += lambda1*l2_reg
            
        if 'conduit_update' in name and 'weight' in name:
            
            l2_reg = torch.norm(param,p=2)
            
            reg_loss += lambda2*l2_reg
            
    return reg_loss

'''
For polypharmacy side effect dataset, the function to compute acc is as follows.
We compute acc for each side effect and average all acc as final result.

p_mat is prediction probability matrix
p_mat.shape[1] = 964
'''

def class_acc(p_mat,
              plabel_mat,
              nlabel_mat):
    
    pre_p = (p_mat > 0.5)*plabel_mat
    
    pre_n = (p_mat <= 0.5)*nlabel_mat
    
    acc_list = []
    
    for c in range(p_mat.shape[1]):
        
        tp_num = torch.sum(pre_p[:, c])
        
        edge1_num = torch.sum(plabel_mat[:, c])
        
        tn_num = torch.sum(pre_n[:, c])
        
        edge2_num = torch.sum(nlabel_mat[:, c])
        
        acc = (tp_num + tn_num)/(edge1_num + edge2_num)
        
        acc_list.append(acc.item())
        
    return acc_list

'''
For polypharmacy side effect dataset, the function to compute loss is as follows.
We compute loss for each side effect and sum all loss as total loss.

We use np.random.choice to randomly select half of the training samples in every epoch.
'''

def my_loss(p_mat,
            plabel_num,
            nlabel_num,
            plabel_mat,
            nlabel_mat,
            weight=964*[1]):

    loss, loss_list = 0, []

    for i in range(p_mat.shape[1]):

        pos_p = np.array(torch.nonzero(plabel_mat[:, i]))

        pos_p = np.random.choice(pos_p.reshape(-1),
                               int(0.5 * plabel_num[i]),
                               replace=False)

        plabel = torch.ones(1, int(0.5 * plabel_num[i]))
        
        pos_n = np.array(torch.nonzero(nlabel_mat[:, i]))

        pos_n = np.random.choice(pos_n.reshape(-1),
                               int(0.5 * nlabel_num[i]),
                               replace=False)
        
        nlabel = torch.zeros((1, int(0.5 * nlabel_num[i])))
        
        pos = np.hstack((pos_p, pos_n))
        
        pos = torch.from_numpy(pos)
        
        label = torch.cat((plabel, nlabel), dim=1).reshape(-1)
        
        loss += F.binary_cross_entropy(p_mat[pos, i], label) * weight[i]
        
        loss_list.append((F.binary_cross_entropy(p_mat[pos, i], label) * weight[i]).item())

    return loss, loss_list