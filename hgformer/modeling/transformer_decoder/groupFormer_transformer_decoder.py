# this version is modified from v8_4_22
# compared to v8_4_22, in this version,
# in this version, we use ratio instead of fixed number of groups for super pixel learning
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union

from detectron2.config import configurable

from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
import numpy as np
import einops
import enum
import math


#######################################################################################################
class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2

class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.

    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        if layer_type == LayerType.IMP1:
            # Experimenting with different options to see what is faster (tip: focus on 1 implementation at a time)
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        else:
            # You can treat this one matrix as num_of_heads independent W matrices
            self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:  # simple reshape in the case of implementation 1
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params(layer_type)

    def init_params(self, layer_type):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3

    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        # NOTE: TODO: in the edge_index [i, j] and [j, i] always both exist.

        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        # import ipdb; ipdb.set_trace()
        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

class GAT(torch.nn.Module):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.

    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.

    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.

    """

    # def __init__(self, num_of_layers=2, num_heads_per_layer=[8, 8], num_features_per_layer=[256, 32, 32], add_skip_connection=True, bias=True,
    #              dropout=0.6, log_attention_weights=False):
    # def __init__(self, num_of_layers=2, num_heads=8, hid_dim=256, add_skip_connection=True, bias=True,
    #              dropout=0.0, log_attention_weights=False):
    # TODO: try experiments with dropout
    def __init__(self, num_of_layers=2, num_heads=8, hid_dim=256, add_skip_connection=True, bias=True,
                 dropout=0.0, log_attention_weights=False):
        super().__init__()
        num_heads_per_layer = [1] + [num_heads for i in range(num_of_layers)]  # trick - so that I can nicely create GAT layers below

        num_features_per_layer = [hid_dim // num_heads_per_layer[i] for i in range(len(num_heads_per_layer))]
        # import ipdb; ipdb.set_trace()
        assert (num_of_layers + 1) == len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        # GATLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        GATLayer = GATLayerImp3

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                # concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                concat=True,
                # activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                activation=nn.ELU(),
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, batch_first=True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.norm = nn.LayerNorm(d_model)
        # import ipdb; ipdb.set_trace()
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        # import ipdb; ipdb.set_trace()
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # print(f'normalize_before {self.normalize_before}---------------------------------------------------------------------------------------------')
        if self.normalize_before:  # [default: false]
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_layer=nn.ReLU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = act_layer()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def local_attention_imp2_2_featurev2(affinity, v, res, spix_idx_tensor_downsample, assign_eps=1e-16):
    # TODO: test this function

    def explicit_broadcast(this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    # This version borrow some ideas from gat
    B, neighbor, H, W = affinity.shape
    V_H, V_W, channels = v.shape[-3], v.shape[-2], v.shape[-1]

    n_spixl_h, n_spixl_w = res

    # TODO: think how to handle -inf in the interpolation of affinity
    affinity_downsample = F.interpolate(
        affinity,
        size=(V_H, V_W),
        mode="bilinear",
        align_corners=False
    )

    # [1, 9, V_H, V_W]
    pixel_idx_tensor_downsample = torch.arange(0, V_H * V_W).unsqueeze(0).unsqueeze(0).expand(1, 9, -1, -1).to(affinity.device)
    pixel_idx_tensor_downsample = pixel_idx_tensor_downsample.reshape(9 * V_H * V_W)
    # TODO: handle the border

    # lift src node features
    v = v.reshape(B, V_H * V_W, channels) # [B, V_H * V_W, channels]
    # import ipdb; ipdb.set_trace()
    nodes_features_proj_lifted = v.index_select(1, pixel_idx_tensor_downsample) # [B, 9*V_H*V_W, channels]

    # neighborhood aware softmax
    affinity_downsample = affinity_downsample.reshape(B, 9*V_H*V_W)
    exp_scores_per_edge = affinity_downsample.exp() # softmax # [B, 9*V_H*V_W]

    # calculate the denominator. cluster-wise norm. shape = (B, V_H*V_W)
    # in cluster wise style, the norm is performed between all clusters which are connected to a same pixel
    neighborhood_sums_cluster_wise = torch.zeros((B, V_H*V_W), dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
    pixel_idx_tensor_downsample = pixel_idx_tensor_downsample.unsqueeze(0).expand(B, -1) # [B, 9 * V_H * V_W]
    neighborhood_sums_cluster_wise.scatter_add_(1, pixel_idx_tensor_downsample, exp_scores_per_edge)

    # shape (B, V_H*V_W) -> (B, 9*V_H*V_W)
    neigborhood_aware_denominator_cluster_wise = neighborhood_sums_cluster_wise.index_select(1, pixel_idx_tensor_downsample[0])
    attentions_per_edge_cluster_wise = exp_scores_per_edge / (neigborhood_aware_denominator_cluster_wise + assign_eps)


    # calculate pixel wise softmax
    neighborhood_sums_pixel_wise = torch.zeros((B, n_spixl_h*n_spixl_w + 1), dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
    # import ipdb; ipdb.set_trace()
    spix_idx_tensor_downsample = spix_idx_tensor_downsample.reshape(B, 9*V_H*V_W)
    neighborhood_sums_pixel_wise.scatter_add_(1, spix_idx_tensor_downsample, attentions_per_edge_cluster_wise)
    neighborhood_sums_pixel_wise[:, -1] = 0.0
    neigborhood_aware_denominator_pixel_wise = neighborhood_sums_pixel_wise.index_select(1, spix_idx_tensor_downsample[0])
    attentions_per_edge_pixel_wise = attentions_per_edge_cluster_wise / (neigborhood_aware_denominator_pixel_wise + assign_eps)

    attentions_per_edge_pixel_wise = attentions_per_edge_pixel_wise.unsqueeze(-1)
    # E = 9*V_H*V_W
    # [B, 9*V_H*V_W, channels] * [B, 9*V_H*V_W, 1] -> [B, 9*V_H*V_W, channels]
    nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge_pixel_wise

    # aggregate neighbors, shape [B, n_spixl_h*n_spixl_w, channels]
    out_nodes_features = torch.zeros((B, n_spixl_h*n_spixl_w + 1, channels), dtype=v.dtype, device=v.device)

    trg_index_broadcasted = explicit_broadcast(spix_idx_tensor_downsample, nodes_features_proj_lifted_weighted)

    out_nodes_features.scatter_add_(1, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

    out_nodes_features = out_nodes_features[:, :n_spixl_h*n_spixl_w]

    return out_nodes_features

def local_attention_imp2_2_cls(affinity, spix_idx_tensor, pixel_idx_tensor, group_cls_scores, assign_eps=1e-16):
    # affinity: [B, 9, H, W]  e.g. [2, 9, 128, 256]
    # spix_idx_tensor: [B, 9, H, W] e.g. [2, 9, 128, 256]
    # pixel_idx_tensor: [B, 9, 1, H*W] e.g. [2, 9, 1, 32768]
    # group_cls_scores: [B, num_classes, num_groups]
    def explicit_broadcast(this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    # This version borrow some ideas from gat
    B, neighbor, H, W = affinity.shape

    # group_cls_scores: [B, num_classes, num_groups]
    num_classes = group_cls_scores.shape[1]
    # num_groups = n_spixl_h * n_spixl_w
    group_cls_scores = group_cls_scores.permute(0, 2, 1) # [B, num_groups, num_classes]
    group_cls_scores_append = torch.zeros(B, 1, num_classes).to(group_cls_scores.device)
    group_cls_scores = torch.cat((group_cls_scores, group_cls_scores_append), 1) # [B, num_groups+1, num_classes]
    # lift group cls scores
    # TODO: handle the hard code of 9
    spix_idx_tensor = spix_idx_tensor.reshape(B, 9*H*W)
    # [B, 9*H*W, num_classes]
    lift_group_cls_scores = group_cls_scores.index_select(1, spix_idx_tensor[0])

    # neighborhood aware softmax
    affinity = affinity.reshape(B, 9*H*W)
    exp_scores_per_edge = affinity.exp() # [B, 9*H*W]
    # calculate the denominator, cluster-wise norm. shape = (B, H*W)
    # cluster-wise norm
    neighborhood_sums_cluster_wise = torch.zeros((B, H*W), dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
    # import ipdb; ipdb.set_trace()
    pixel_idx_tensor = pixel_idx_tensor.reshape(B, 9*H*W)
    neighborhood_sums_cluster_wise.scatter_add_(1, pixel_idx_tensor, exp_scores_per_edge)

    # shape (B, H*W) -> (B, 9*H*W)
    neighborhood_sums_cluster_wise_denominator = neighborhood_sums_cluster_wise.index_select(1, pixel_idx_tensor[0])
    attentions_per_edge_cluster_wise = exp_scores_per_edge / (neighborhood_sums_cluster_wise_denominator + assign_eps)

    attentions_per_edge_cluster_wise = attentions_per_edge_cluster_wise.unsqueeze(-1)

    # [B, 9*H*W, num_classes] * [B, 9*H*W, 1] -> [B, 9*H*W, num_classes]
    lift_group_cls_scores_weighted = lift_group_cls_scores * attentions_per_edge_cluster_wise

    # aggregate cls scores from nearby clusters for each pixel, shape [B, H * W, num_classes]
    out_pixel_cls = torch.zeros((B, H*W, num_classes), dtype=group_cls_scores.dtype, device=group_cls_scores.device)

    trg_index_broadcasted = explicit_broadcast(pixel_idx_tensor, lift_group_cls_scores_weighted)

    out_pixel_cls.scatter_add_(1, trg_index_broadcasted, lift_group_cls_scores_weighted)

    return out_pixel_cls.permute(0, 2, 1).reshape(B, num_classes, H, W)

class SpixAttentionv2(nn.Module):
    def __init__(self, dim, qkv_bias=True, qk_scale=None, assign_eps=1., tau=0.07):
        super().__init__()
        # self.scale = 1.0 # TODO: carefully set the scale
        # self.scale = qk_scale or dim ** -0.5
        self.tau = tau

        # self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.assign_eps = assign_eps
        # self.norm = nn.LayerNorm(dim)
        # self.num_groups = num_groups

    def forward(self, q_feature, k_feature, v_feature, spix_res, spix_idx_tensor_downsample):
        """
          In this version, we keep q, k in the same embedding space, and return both the average of k and v
          In this version, we directly pass num_group queries to this function, so that we can iteratively perform
          super pixel clustering
          This variation, similar to the region proxy, use shallow layers to learn the affinity,
          and use high-level features for classification
          input: q_feature: (B, num_groups, C), k_feature: (B, C, H, W), v_feature: (B, C, V_H, V_W)
          output: attn_cluster_norm: (B, num_groups, H, W)
                  attn_pixel_norm: (B, num_groups, H, W)
                  out: (B, num_groups, C)
                  affinity: (B, 9, H, W)
        """

        n_spixl_h, n_spixl_w, Hh, Ww = spix_res

        # cluster center features q with shape (B, num_groups, C)
        V_H, V_W = v_feature.shape[-2:]
        v_feature = v_feature.permute(0, 2, 3, 1) # [B, V_H, V_W, C]
        v = self.v_proj(v_feature)# [B, V_H, V_W, C]

        k_feature_downsample = F.interpolate(k_feature,
                                             size=(V_H, V_W),
                                             mode="bilinear",
                                             align_corners=False
                                             )
        # [B, C, V_H, V_W]->[B, V_H, V_W, C]
        k_feature_downsample = k_feature_downsample.permute(0, 2, 3, 1)

        # (B, C, H, W) -> (B, H, W, C)
        k_feature = k_feature.permute(0, 2, 3, 1)
        # import ipdb; ipdb.set_trace()
        # k_proj = self.k_proj(k_feature)
        B, H, W, C = k_feature.shape
        k_feature = k_feature.reshape(B, H * W, C).unsqueeze(-1) # (B, H*W, C, 1)

        # q_proj = self.q_proj(q_feature)
        q_feature = q_feature.reshape(B, n_spixl_h, n_spixl_w, C).permute(0, 3, 1, 2)

        # im2col with padding: (B, C, n_spixl_h, n_spixl_w)->(B, C, 9, n_spixl_h, n_spixl_w)
        q = F.unfold(q_feature, kernel_size=3, padding=1).reshape(B, -1, 9, n_spixl_h, n_spixl_w)

        q = F.interpolate(q.reshape(B, -1, n_spixl_h, n_spixl_w), size=(H, W), mode='nearest')
        q = q.reshape(B, -1, 9, H, W).permute(0, 3, 4, 1, 2).reshape(B, H * W, C, 9)

        # TODO: in the next version, we may
        norm_q = F.normalize(q, dim=-2)
        norm_k = F.normalize(k_feature, dim=-2) # (B, H*W, C, 1)

        affinity = norm_k.transpose(-2, -1) @ norm_q  # (B, H*W, 1, 9)

        affinity = affinity.permute(0, 3, 1, 2).reshape(B, 9, H, W)

        # handle borders
        affinity = affinity.reshape(B, 9, n_spixl_h, Hh,
                                    n_spixl_w, Ww)
        affinity = einops.rearrange(affinity, 'B n H h W w -> B n h w H W')
        affinity[:, :3, :, :, 0, :] = float('-inf')  # top
        affinity[:, -3:, :, :, -1, :] = float('-inf')  # bottom
        affinity[:, ::3, :, :, :, 0] = float('-inf')  # left
        affinity[:, 2::3, :, :, :, -1] = float('-inf')  # right

        affinity = affinity / self.tau
        # raw affinity matrix
        affinity = einops.rearrange(affinity, 'B n h w H W -> B n (H h) (W w)')  # (B, 9, H, W)

        # import ipdb; ipdb.set_trace()
        assert v.shape == k_feature_downsample.shape
        out_cls_features = local_attention_imp2_2_featurev2(affinity, v, (n_spixl_h, n_spixl_w), spix_idx_tensor_downsample)

        # TODO: try to use high-resolution k_feature to update query features
        out_query_features = local_attention_imp2_2_featurev2(affinity, k_feature_downsample, (n_spixl_h, n_spixl_w), spix_idx_tensor_downsample)

        return out_query_features, out_cls_features, affinity


class PositionEmbeddingSineWShape(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x_shape, x_device, mask=None):
        if mask is None:
            mask = torch.zeros((x_shape[0], x_shape[2], x_shape[3]), device=x_device, dtype=torch.bool)
        # mask: e.g. shape [2, 16, 32], [B, H, W]
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B, H, W]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # [B, H, W]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # normalize the coordinates, then multiply 2pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x_device)  # [128]
        # import ipdb; ipdb.set_trace()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)  # [B, H, W, num_pos_feats]
        # import ipdb; ipdb.set_trace()
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # import ipdb; ipdb.set_trace()
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1,
                                                       2)  # [B, 2*num_pos_feats, H, W], 2 * num_pos_feats is equal to the number of feat channels
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


@TRANSFORMER_DECODER_REGISTRY.register()
class GroupFormerDecoder(nn.Module):
    """
    output tree-structured masks, and their logits
    set fixed areas for super pixels, instead of fixed number of super pixels
    """

    @configurable
    def __init__(
            self,
            *,
            num_classes: int,
            hidden_dim: int,
            contrastive_dim: int,
            tau: float,
            num_group_tokens: List[int],
            num_output_groups: List[int],
            num_heads: List[int],
            spix_res: List[int],
            pre_norm: bool,
            gat_num_layers: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            spix_atten_layers: int,
            downsample_rate: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            enc_layers: number of Transformer encoder layers
        """
        super().__init__()
        self.mask_classification = True
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSineWShape(N_steps, normalize=True)

        self.num_heads = num_heads
        self.num_layers = dec_layers
        self.spix_atten_layers = spix_atten_layers
        self.downsample_rate = downsample_rate
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        assert len(self.num_group_tokens) == 2
        # here is hardcoded
        self.num_spixs = self.num_group_tokens[0]
        self.num_queries = self.num_group_tokens[1]
        self.spix_res = spix_res
        self.tau = tau
        assert len(self.num_group_tokens) == len(self.num_output_groups)
        self.group_blocks = nn.ModuleList()
        self.spix_self_attention_layers = nn.ModuleList()
        self.spix_ffn_layers = nn.ModuleList()
        self.affinity_head_layers = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.query_embeds = nn.ModuleList()
        self.query_feat = nn.ModuleList()

        # n x local grouping stage (super pixel) + m x maskformer style grouping
        # spix self attention
        for i_layer in range(self.spix_atten_layers):
            self.spix_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.spix_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.affinity_head_layers.append(
                SpixAttentionv2(hidden_dim, tau=self.tau)
            )

        for i_layer in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        for i_layer in range(len(self.num_group_tokens)):
            self.query_embeds.append(nn.Embedding(self.num_group_tokens[i_layer], hidden_dim))
            self.query_feat.append(nn.Embedding(self.num_group_tokens[i_layer], hidden_dim))

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.class_embed_spix = nn.Linear(hidden_dim, num_classes)
        self.clsss_embed_pixel = nn.Linear(hidden_dim, num_classes)

        self.linear_fuse = nn.Conv2d(
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim,
            kernel_size=1
        )
        mask_dim = hidden_dim  # sorry, hard coded
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        # TODO: add mlp layers, similar to moco
        self.contrastive_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        # ret["in_channels"] = in_channels

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["contrastive_dim"] = cfg.MODEL.MASK_FORMER.CONTRASTIVE_DIM
        # Transformer parameters:

        ret["num_group_tokens"] = cfg.MODEL.SEM_SEG_HEAD.NUM_GROUP_TOKENS
        ret["num_output_groups"] = cfg.MODEL.SEM_SEG_HEAD.NUM_OUTPUT_GROUPS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["tau"] = cfg.MODEL.SEM_SEG_HEAD.TAU
        ret["spix_res"] = cfg.MODEL.SEM_SEG_HEAD.SPIX_RES
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["spix_atten_layers"] = cfg.MODEL.MASK_FORMER.SPIX_SELF_ATTEN_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["gat_num_layers"] = cfg.MODEL.SEM_SEG_HEAD.GAT_NUM_LAYERS
        ret["downsample_rate"] = cfg.MODEL.SEM_SEG_HEAD.DOWNSAMPLE_RATE

        return ret

    def forward(self, x, mask_features, mask=None):
        # x: list of multi-scale features,
        # shape of x[i]: [B, C, h, w]
        x.reverse()  # from high to low resolution

        x_feature = self.feature_fusion(x)
        pixel_level_logit = self.clsss_embed_pixel(x_feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        B, C, H, W = x_feature.shape
        # mask_feature_H, mask_feature_w = mask_features.shape[-2:]
        # super pixel stage
        # resolution for each super pixel anchor cell
        # TODO: check mask_features.shape
        # import ipdb; ipdb.set_trace()
        # print(f'x[0].shape{x[0].shape}')
        # mask_features: [B, C, mask_H, mask_W]->[B, C, mask_H*mask_W]->[B, mask_H*mask_W, C]
        ori_mask_H, ori_mask_W = mask_features.shape[-2], mask_features.shape[-1]

        q_k_features = mask_features.reshape(B, C, ori_mask_H * ori_mask_W).permute(0, 2, 1)
        q_k_features = self.contrastive_proj(q_k_features) # [B, mask_H*mask_W, C]
        q_k_features = q_k_features.reshape(B, ori_mask_H, ori_mask_W, C).permute(0, 3, 1, 2) # [B, C, mask_H, mask_W]
        spix_k_feature = F.avg_pool2d(q_k_features, (2, 2), (2, 2))
        mask_H, mask_W = spix_k_feature.shape[-2], spix_k_feature.shape[-1]
        # Ww, Hh = int(np.sqrt((mask_H * mask_W) / self.num_spixs)), int(np.sqrt((mask_H * mask_W) / self.num_spixs))
        # assert int(np.sqrt((mask_H * mask_W) / self.num_spixs)) == np.sqrt((mask_H * mask_W) / self.num_spixs)
        # (B, H, W, C)->(B, C, H, W)->(B, C, H/Hh, W/Ww)
        assert mask_H % self.downsample_rate == 0
        assert mask_W % self.downsample_rate == 0
        n_spixl_h = int(np.floor(mask_H / self.downsample_rate))
        n_spixl_w = int(np.floor(mask_W / self.downsample_rate))
        # import ipdb; ipdb.set_trace()
        spix_res = (n_spixl_h, n_spixl_w, self.downsample_rate, self.downsample_rate)

        spix_values = torch.arange(0, n_spixl_h * n_spixl_w).reshape(n_spixl_h, n_spixl_w).to(mask_features.device)
        spix_values = spix_values.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        p2d = (1, 1, 1, 1)
        spix_values = F.pad(spix_values, p2d, "constant", n_spixl_h * n_spixl_w)
        spix_idx_tensor_ = F.unfold(spix_values.type(mask_features.dtype), kernel_size=3).reshape(B, 1 * 9, n_spixl_h,
                                                                                             n_spixl_w).int()
        # target index
        spix_idx_tensor = F.interpolate(spix_idx_tensor_.type(mask_features.dtype), size=(mask_H, mask_W),
                                        mode='nearest').long()  # (B, 9, H, W)

        spix_idx_tensor_downsample = F.interpolate(spix_idx_tensor_.type(mask_features.dtype), size=(H, W),
                                                   mode='nearest').long()
        # src index
        # [B, 9, H, W]
        pixel_idx_tensor = torch.arange(0, mask_H * mask_W).unsqueeze(0).unsqueeze(0).expand(B, 9, -1, -1).to(mask_features.device)
        pixel_idx_tensor = pixel_idx_tensor.reshape(B, 9, mask_H, mask_W)

        x_shape = [B, C, n_spixl_h, n_spixl_w]
        x_device = x_feature.device
        pos = self.pe_layer(x_shape, x_device)
        pos = pos.reshape(B, C, n_spixl_h * n_spixl_w).permute(0, 2, 1)
        # TODO: try relative position embedding
        # spix_query_embed = self.query_embeds[0].weight.unsqueeze(0).repeat(B, 1, 1)
        # import ipdb; ipdb.set_trace()
        output_spix = F.avg_pool2d(spix_k_feature, (self.downsample_rate, self.downsample_rate), (self.downsample_rate, self.downsample_rate))
        output_spix = output_spix.reshape(B, C, n_spixl_h * n_spixl_w).permute(0, 2, 1)
        spix_v_feature = x_feature
        # predicitons_class_spix = []
        predicitons_class_spix_pixel_cls = []
        affinity_list = []
        for i in range(self.spix_atten_layers):
            # TODO: add pos in the affinity_head attention in the future, add a pos paramter in affinity_head_layers
            # import ipdb; ipdb.set_trace()
            output_spix, output_cls, affinity = self.affinity_head_layers[i](
                output_spix, spix_k_feature, spix_v_feature, spix_res, spix_idx_tensor_downsample)
            output_cls = self.spix_self_attention_layers[i](
                output_cls, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=pos
            )
            output_cls = self.spix_ffn_layers[i](
                output_cls
            )

            # TODO: check this line
            decoder_output_spix = self.decoder_norm(output_cls) # [B, num_spixs, channels], e.g. [2, 512, 256]

            outputs_class_spix = self.class_embed_spix(decoder_output_spix) # [B, num_groups, num_classes]
            outputs_class_spix = outputs_class_spix.permute(0, 2, 1) # [B, num_classes, num_groups]
            outputs_class_spix_pixel_cls = local_attention_imp2_2_cls(affinity, spix_idx_tensor, pixel_idx_tensor, outputs_class_spix)

            predicitons_class_spix_pixel_cls.append(outputs_class_spix_pixel_cls)
            affinity_list.append(affinity)

        # TODO: think how to add attn_mask here
        # import ipdb; ipdb.set_trace()
        # mask stage
        query_embed = self.query_embeds[1].weight.unsqueeze(0).repeat(B, 1, 1)
        output = self.query_feat[1].weight.unsqueeze(0).repeat(B, 1, 1)
        predictions_class = []
        predictions_mask = []
        # import ipdb; ipdb.set_trace()
        for i in range(self.num_layers):  #
            output = self.transformer_cross_attention_layers[i](
                output, output_cls,
                memory_key_padding_mask=None,
                pos=pos, query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            output = self.transformer_ffn_layers[i](
                output
            )

            # TODO: check the batch first issue in the layer norm, it seems that there is no bug
            # import ipdb; ipdb.set_trace()
            decoder_output = self.decoder_norm(output) # [B, num_queries, channels], e.g. [2, 64, 256]
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            outputs_class = self.class_embed(decoder_output)

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if self.training:
            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
               'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask
                ),
                "predicitons_class_spix_pixel_cls": predicitons_class_spix_pixel_cls,
                'pixel_level_logits': pixel_level_logit,
                # "k_proj_list": k_proj_list
                'mask_features': q_k_features  # used for contrastive loss
            }
        else:
            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask
                ),
                'affinity_list': affinity_list,
                "predicitons_class_spix_pixel_cls": predicitons_class_spix_pixel_cls,
                'pixel_level_logits': pixel_level_logit,
                "x_shape": x_shape,
            }

        return out

    def feature_fusion(self, features):
        x = []
        x.append(features[0])
        for i in range(1, len(features)):
            x.append(F.interpolate(features[i], scale_factor=np.power(2, i), mode="bilinear", align_corners=False))
        x = self.linear_fuse(torch.cat(x, dim=1))
        return x

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]