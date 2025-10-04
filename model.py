import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, AvgPooling



class TCM_GNN(nn.Module):
    def __init__(self, relations, num_layer, in_dim, embed_dim, temp, dropout, use_co, use_attention, use_segpool):
        super(TCM_GNN, self).__init__()

        self.relations = relations
        self.num_relations = len(relations) 
        self.num_layer = num_layer
        self.embed_dim = embed_dim 
        self.temp = temp
        self.use_segpool = use_segpool
        self.TCM_GNN_layer = nn.ModuleList()
        self.TCM_GNN_layer.append(Layer(relations=relations, num_relations=self.num_relations,
                            in_dim=in_dim, out_dim=embed_dim, temp=temp, use_co=use_co, use_attention=use_attention))
        for _ in range(num_layer-1):
            self.TCM_GNN_layer.append(Layer(relations=relations, num_relations=self.num_relations,
                            in_dim=embed_dim, out_dim=embed_dim, temp=temp, use_co=use_co, use_attention=use_attention))
        if use_segpool:
            self.pool = SegPooling()
        else:
            self.pool = AvgPooling()
        self.regressor = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), 
                                       nn.Dropout(dropout), nn.Linear(embed_dim // 2, 1))

    def forward(self, graph, pool_ids): 
        pool_ids = pool_ids.float()
        all_feat = graph.ndata['feat'].float() 
        selected_tensors = []

        if 't1' in self.relations:
            selected_tensors.append(all_feat[:, :14])
        if 't1c' in self.relations:
            selected_tensors.append(all_feat[:, 14:28])
        if 't2' in self.relations:
            selected_tensors.append(all_feat[:, 28:42])
        if 'flair' in self.relations:
            selected_tensors.append(all_feat[:, 42:56])

        feat = torch.cat(selected_tensors, dim=1)
        
        for i in range(self.num_layer):
            feat = self.TCM_GNN_layer[i](graph, feat)

        if self.use_segpool:
            feat = self.pool(graph, feat, pool_ids)
        else:
            feat = self.pool(graph, feat)
        pred = self.regressor(feat).squeeze(dim=-1)

        return pred, feat
       
class Layer(nn.Module):
    def __init__(self, relations, num_relations, in_dim, out_dim, temp, use_co, use_attention):
        super(Layer, self).__init__()
        self.relations = relations
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.temp = temp
        self.use_co = use_co
        self.use_attention = use_attention
        if use_co:
            if self.temp is None:
                self.softplus = nn.Softplus(beta=1)
                self.act_gnn = GATConv(in_feats=in_dim, out_feats=6, num_heads=4, residual=False, 
                                        activation=nn.GELU(), allow_zero_in_degree=True)
            else:
                self.act_gnn = GATConv(in_feats=in_dim, out_feats=4, num_heads=4, residual=False, 
                                        activation=nn.GELU(), allow_zero_in_degree=True)
        self.env_gnn = GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=4, residual=True, 
                                activation=nn.GELU(), allow_zero_in_degree=True)
        if use_attention:
            self.cop_att = Attention(num_relations, out_dim)
        else:
            self.cop_att = nn.Linear(num_relations*out_dim, out_dim)
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True)
    def forward(self, graph, feat):
        h = torch.zeros(self.num_relations, graph.num_nodes(), self.out_dim, device=feat.device)
        with graph.local_scope():
            for i, relation in enumerate(self.relations):
                rel_graph = graph['bn', relation, 'bn']
                if self.use_co:
                    if self.temp is None:
                        logits = self.act_gnn(rel_graph, feat)
                        logits = logits.mean(dim=1)
                        in_logits = logits[:, :2]
                        temp_in = logits[:, 2].unsqueeze(-1)
                        temp_in = self.softplus(temp_in) + 0.5
                        temp_in = temp_in.pow_(-1)
                        temp_in.masked_fill_(temp_in == float('inf'), 0.)
                        out_logits = logits[:, 3:5]
                        temp_out = logits[:, -1].unsqueeze(-1)
                        temp_out = self.softplus(temp_out) + 0.5
                        temp_out = temp_out.pow_(-1)
                        temp_out.masked_fill_(temp_out == float('inf'), 0.)
                        in_logits = F.gumbel_softmax(logits=in_logits, tau=temp_in, hard=True)
                        out_logits = F.gumbel_softmax(logits=out_logits, tau=temp_out, hard=True)
                    else:
                        logits = self.act_gnn(rel_graph, feat)
                        logits = logits.mean(dim=1)
                        in_logits = logits[:, :2]
                        out_logits = logits[:, 2:]
                        in_logits = F.gumbel_softmax(logits=in_logits, tau=self.temp, hard=True)
                        out_logits = F.gumbel_softmax(logits=out_logits, tau=self.temp, hard=True)

                    edge_weight = self.create_edge_weight(rel_graph, 
                                        keep_in_prob=in_logits[:, 0], keep_out_prob=out_logits[:, 0])

                    h_out = self.env_gnn(rel_graph, feat, edge_weight)
                else:
                    h_out = self.env_gnn(rel_graph, feat)
                h_out = h_out.sum(dim=1)
                h[i] = h_out 

        h = self.norm(h) 
        if self.use_attention:
            h = self.cop_att(h) 
        else:
            h = h.permute(1, 0, 2).contiguous().view(-1, self.num_relations*self.out_dim)
            h = self.cop_att(h)
        return h

    def create_edge_weight(self, g, keep_in_prob, keep_out_prob):
        u, v = g.edges()

        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]

        edge_weight = edge_in_prob * edge_out_prob

        return edge_weight
    


class Attention(nn.Module):
    def __init__(self, num_relations, dim):
        super(Attention, self).__init__()
        self.num_relations = num_relations
        self.dim = dim
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wp = nn.Linear(num_relations*dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h):
        h = h.permute(1, 0, 2)
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        score = q @ k.transpose(1, 2) / math.sqrt(self.dim) 
        score = self.softmax(score)
        out = score @ v
        out = out.view(-1, self.num_relations*self.dim)
        out = self.wp(out)

        return out
    

    
class SegPooling(nn.Module):
    def __init__(self):
        super(SegPooling, self).__init__()

    def forward(self, graph, feat, pool_ids):
        with graph.local_scope():
            pool_ids = pool_ids.reshape(-1, 1) 
            feat = feat * pool_ids 
            graph.ndata["h"] = feat
            readout = dgl.mean_nodes(graph, "h")
            return readout
        