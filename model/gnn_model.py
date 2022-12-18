
import numpy as np
import sklearn.metrics
from numpy import r_, around
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data_processing.data_loader import ImageGraphDataset, minibatch_graphs
from .networks import init_graph_net
from . import evaluation
from data_processing.graph_io import project_nodes_to_img



BATCH_SIZE=6

'''
#Input#
model_type is a string that determines the type of graph learning layers used (GraphSAGE, GAT)
hyperparameters is a named tuple defined in utils/hyperparam helpers
train_dataset is an ImageGraphDataset with read_graph set to True.
'''

class GNN:
    def __init__(self,model_type,hyperparameters,train_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device", self.device)
        print(torch.cuda.get_device_name(self.device))
        class_weights = torch.FloatTensor(hyperparameters.class_weights).to(self.device)
        self.net=init_graph_net(model_type,hyperparameters)      
        self.net.to(self.device)
        self.optimizer=torch.optim.AdamW(self.net.parameters(),lr=hyperparameters.lr,weight_decay=hyperparameters.w_decay)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, hyperparameters.lr_decay, last_epoch=-1, verbose=False)
        self.loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=minibatch_graphs) if train_dataset is not None else None


    def run_epoch(self):
        self.net.train()
        losses=[]
        f1_score=[]

        for batch_mris,batch_graph,batch_feats,batch_label in self.train_loader:
            batch_graph = batch_graph.to(self.device)
            batch_feats = batch_feats.to(self.device)
            batch_labels = batch_label.to(self.device)
            logits = self.net(batch_graph,batch_feats)
            loss = self.loss_fcn(logits, batch_label)
            losses.append(loss.item())

            _, pred_classes = torch.max(logits, dim=1)
            pred_classes=pred_classes.detach().cpu().numpy()
            labels = batch_labels.detach().cpu().numpy()
            f1 = sklearn.metrics.f1_score(pred_classes,labels,average='micro')
            f1_score.append(f1.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.lr_decay.step()


        return np.mean(losses),np.mean(f1_score)

    #must be a Subset of an ImageGraphDataset
    def evaluate(self,dataset:ImageGraphDataset):
        assert(dataset.dataset.read_label==True)
        self.net.eval()
        #metrics stores loss,label counts, node dices,voxel dices,voxel hausdorff
        metrics = np.zeros((len(dataset)+1,10))
        counts = np.zeros((len(dataset)+1,8))
        i=0
        print("[Loss, WT Node Dice, CT Node Dice, ET Node Dice, WT Volex Dice, CT Volex Dice, ET Volex Dice, WT HD95, CT HD95, ET HD95]")
        for curr_ids,curr_graphs,curr_feat,curr_label in dataset:
            curr_graphs = curr_graphs.to(self.device)
            curr_feat = torch.FloatTensor(curr_feat).to(self.device)
            curr_label = torch.LongTensor(curr_label).to(self.device)
            with torch.no_grad():
                logits = self.net(curr_graphs,curr_feat)
                loss = self.loss_fcn(logits, curr_label)
            _, predicted_classes = torch.max(logits, dim=1)
            predicted_classes=predicted_classes.detach().cpu().numpy()
            metrics[i][0]=loss.item()
            ct, res = self.calculate_all_metrics_for_brain(curr_ids,dataset,predicted_classes,curr_label.detach().cpu().numpy())
            metrics[i][1:] = res
            counts[i]=ct
            i+=1

        avg_metrics = np.mean(metrics,axis=0)
        total_counts = np.sum(counts,axis=0)
        return avg_metrics, total_counts

    #Calculates a slew of different metrics that might be interesting such as the number of nodes of each label and voxel and node Dice scores
    def calculate_all_metrics_for_brain(self,mri_id,Dataset,node_pred,node_label):
        label_count = np.concatenate([evaluation.count_node_labels(node_pred),evaluation.count_node_labels(node_label)])
        node_dice = evaluation.calculate_node_dices(node_pred,node_label)
        #read in voxel_labels and supervoxel mapping to compute the image metrics

        #initial dataset into folds. There is likely a more elegant solution than this.
        sv_partitioning = Dataset.dataset.get_supervoxel_partitioning(mri_id)
        true_voxel = Dataset.dataset.get_voxel_labels(mri_id)
        pred_voxel = project_nodes_to_img(sv_partitioning,node_pred)
        voxel_metric = evaluation.calculate_brats_metrics(pred_voxel,true_voxel)
        return label_count,np.concatenate([node_dice,voxel_metric])

    def save_weights(self,folder,name):
        torch.save(self.net.state_dict(),f"{folder}{name}.pt")



