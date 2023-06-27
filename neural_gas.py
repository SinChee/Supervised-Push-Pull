import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

import matplotlib.pyplot as plt

from k_center_greedy import KCenterGreedy

# experiment/vte-augment/vte/models/heads/anomaly/utils

__authors__ = 'Sin Chee Chin'
__email__ = 'chenxz22@mails.tsinghua.edu.cn'

l2_loss = nn.MSELoss(reduction='sum')

def rayleigh_kernel(x, y, k=1, a=1):
    assert a > 0, "a should be greater than zero"
    assert k > 0, "k should be greater than zero"
    dx = l2_loss(x, y)
    return k/a*dx*torch.exp(-0.5*(dx/a)**2)

def negative_exp_kernel(x, y, k=1, a=1):
    assert a > 0, "a should be greater than zero"
    assert k > 0, "k should be greater than zero"
    dx = l2_loss(x, y)
    return k*torch.exp(-a*(dx)**2)


class SupervisedNeuralGas(nn.Module):
    def __init__(self, data, attract=None, repel=None, sampling_count = 100, *args, **kwargs) -> None:
        """attract (B, ...): good points repel (B, ...): bad points."""
        super().__init__(*args, **kwargs)

        self.data = data
        self.primal = data.clone()
        self.attracts = attract if attract is not None else None
        self.repels = repel if repel is not None else None

        self.network = nx.Graph()
        self.units_created = 0
        self.device = self.attracts.device
        self.sampling_count = sampling_count


    def initialize(self, attract=None, repel=None):
        if attract is not None:
            self.attracts = torch.concatenate([self.attracts, attract], dim=0)
        if repel is not None:
            self.repels = torch.concatenate([self.repels, repel], dim=0)
        self.max = torch.max(self.attracts, dim=0).values
        self.min = torch.min(self.attracts, dim=0).values

        for d in self.data:
            self.network.add_node(self.units_created, vector=d, error=0)
            self.units_created += 1

    def check_dim(self, data):
        for d in data:
            assert d.shape == data[0].shape


    def k_nearest(self, data, node, index, k=1):
        d_k = F.pairwise_distance(node, data, p=2) # l2 distance
        idxs = torch.ones(k, device=self.device)
        for i, idx in enumerate(torch.topk(d_k, k, largest=False).indices):
            idxs[i] = index[idx.item()]
        return idxs

    
    def _network_to_tensor(self, network):
        data = []
        for x in network.nodes(data=True):
            data.append(x[1]['vector'].unsqueeze(0))
        data = torch.cat(data, dim=0)
        return data


    def replace_points(self, prune_index, generate_s):
        # nodes_to_remove = []
        # for u in self.network.nodes():
        #     nodes_to_remove.append(u)
        # breakpoint()
        # for u in nodes_to_remove:
        #     self.network.remove_node(u)
        
        for u, s in zip(prune_index, generate_s):
            u = u.item()
            for neighbour in list(self.network.neighbors(u)):
                self.network.remove_edge(u, neighbour)
            self.network.nodes[u]['error'] = 0
            self.network.nodes[u]['vector'] = s
        self.data = self._network_to_tensor(self.network)
            
        
    def forward(self, epochs, 
                a_o=0.1, k_o=0.1, a_on=0.006, k_on=0.006, 
                a_r=0.1, k_r=0.1, a_rn=0.006, k_rn=0.006,
                prune_epoch=2, prune_threshold=0.01,
                verbose=False):
        self.initialize()
        count = 0
        previous_loss = torch.inf
        for epoch in tqdm(range(epochs)):
            freq_visit_guide = torch.zeros(self.attracts.shape[0])
            freq_visit_sample = torch.zeros(self.data.shape[0])
            for s1 in self.network.nodes(data=True):
                s1_index = s1[0]
                s1_vector = s1[1]['vector']

                # find nearest neighbour from the sample itself
                # use 6 points because first point is s1 itself
                nearest_samples = self.k_nearest(s1_vector, self.data, index=list(range(self.data.size(0))), k=6)

                for sk in nearest_samples[1:]:
                    freq_visit_sample[int(sk.item())] += 1
                    self.network.add_edge(s1_index, int(sk.item()))
                    self.network.nodes

                # find nearest neighbour from guide with reference to s1 and constrct the moving vector
                nearest_attracts = self.k_nearest(s1_vector, self.attracts, index=range(self.attracts.size(0)), k=3)
                move_s1 = torch.zeros_like(s1_vector)
                for gk_1 in nearest_attracts:
                    freq_visit_guide[int(gk_1.item())] += 1
                    gk_vector = self.attracts[int(gk_1.item())]
                    e_o = rayleigh_kernel(gk_vector, s1_vector, a_o, k_o)
                    move_s1 += e_o * (gk_vector - s1_vector)
                
                # find nearest neighbour from repel with reference to s1 and constrct the moving vector
                if self.repels is not None:
                    nearest_repel = self.k_nearest(s1_vector, self.repels, index=range(self.repels.size(0)), k=10)
                    for idx_repel in nearest_repel:
                        repel_vector = self.repels[int(idx_repel.item())]
                        e_r = negative_exp_kernel(repel_vector, s1_vector, a_r, k_r)
                        move_s1 -= e_r * (repel_vector - s1_vector)

                # move s1
                self.network.nodes[s1_index]['vector'] = self.network.nodes[s1_index]['vector'] + move_s1
                # self.data[s1_index] += move_s1

                # move each neighbour of s1
                for sk in self.network.neighbors(s1_index):
                    move_sk = torch.zeros_like(s1_vector)

                    # move towards the same direction as s1
                    for gk_1 in nearest_attracts:
                        gk_vector = self.attracts[int(gk_1.item())]
                        e_on = rayleigh_kernel(gk_vector, self.network.nodes[sk]['vector'], a_on, k_on)
                        move_sk += e_on * (gk_vector - self.network.nodes[sk]['vector'])
                    
                    # however, we have to avoids points that are cloest to sk itself
                    if self.repels is not None:
                        nearest_repel_neighbour = self.k_nearest(self.network.nodes[sk]['vector'], self.repels, index=range(self.repels.size(0)), k=10)
                        for idx_repel in nearest_repel_neighbour:
                            repel_vector = self.repels[int(idx_repel.item())]
                            e_rn = negative_exp_kernel(repel_vector, self.network.nodes[sk]['vector'], a_rn, k_rn)
                            move_sk -= e_rn * (repel_vector - self.network.nodes[sk]['vector'])

                    # move sk
                    self.network.nodes[sk]['vector'] = self.network.nodes[sk]['vector'] + move_sk
                    # self.data[sk] += move_sk
 
            if verbose:
                # visualize_tsne("tsne-epoch-"+str(epoch), 
                #             [self.primal, self.attracts, self.repels, self._network_to_tensor(self.network)],
                #             [0, 0, 1, 0],
                #             ['original', 'guide', 'repel', 'moved'],
                #             [0.5, 0.3, 0.3, 0.5])
                os.makedirs("visualize", exist_ok=True)
                moved = self._network_to_tensor(self.network).detach().cpu().numpy()
                good = self.attracts.detach().cpu().numpy()
                bad = self.repels.detach().cpu().numpy()
                plt.scatter(good[:, 0], good[:, 1], label='Good')
                plt.scatter(bad[:, 0], bad[:, 1], label='Bad')
                plt.scatter(moved[:, 0], moved[:, 1], label='Sample')
                plt.legend()
                plt.savefig("visualize/"+str(epoch)+".png")
                plt.close()
            
            # early stopping
            current_loss = 0
            for s1 in self.network.nodes(data=True):
                s1_vector = s1[1]['vector']

                nearest_guide = self.k_nearest(s1_vector, self.attracts, index=range(self.attracts.size(0)), k=20)

                d_s1_total = 0
                for gk in nearest_guide:
                    gk_vector = self.attracts[int(gk.item())]
                    d_s1_total += l2_loss(s1_vector, gk_vector)
                current_loss += d_s1_total
            
            if previous_loss < current_loss:
                count += 1
                if count >= 5:
                    break
            else:
                count = 0
                previous_loss = current_loss
                self.best_network = self.network.copy()

            # prune and add points here
            # if epoch % prune_epoch:
            #     k = int(prune_threshold * len(self.data))
            #     # find k points from sample that have the largest neighbour -> prune_s
            #     _, prune_s = torch.topk(freq_visit_sample, k)

            #     # find k points from guides
            #     sampler = KCenterGreedy(embedding=self.attracts, sampling_count=k)
            #     generate_s = sampler.sample_coreset()

            #     # prune prune_s and replace with generate_s
            #     self.replace_points(prune_s, generate_s)
                    

        
        data = []
        for x in self.best_network.nodes(data=True):
            data.append(x[1]['vector'].unsqueeze(0))
        self.check_dim(data)
        data = torch.cat(data, dim=0)
        
        return data
