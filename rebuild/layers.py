import numpy as np
from torch import nn
import torch
from torch.nn import GELU
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, dim: int, variance_epsilon=1e-12,
                 *args, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 n_segments: int,
                 dim: int,
                 p_drop_hidden: int,
                 *args,
                 **kwargs):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim)  # token embedding
        self.pos_embed = nn.Embedding(max_len, dim)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, dim)  # segment(token type) embedding

        self.norm = LayerNorm(dim)
        self.drop = nn.Dropout(p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self,
                 dim: int,
                 p_drop_attn: float,
                 n_heads: int,
                 *args,
                 **kwargs):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self._split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = self._merge_last(h, 2)
        self.scores = scores
        return h

    def _split_last(self, x, shape):
        "split the last dimension to given shape"
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)

    def _merge_last(self, x, n_dims):
        "merge the last n_dims to a dimension"
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, dim: int, dim_ff: int, *args, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        self.activation = nn.GELU()

        # self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(self.activation(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, dim=768, dim_ff=3072, n_heads=12, p_drop_hidden=0.1,
                 p_drop_attn=0.1, variance_epsilon=1e-12, *args, **kwargs):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, p_drop_attn, n_heads)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = LayerNorm(dim, variance_epsilon)
        self.pwff = PositionWiseFeedForward(dim, dim_ff)
        self.norm2 = LayerNorm(dim, variance_epsilon)
        self.drop = nn.Dropout(p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, vocab_size, max_len, n_segments, dim=768, n_layers=8,
                 p_drop_hidden=0.1, p_drop_attn=0.1, dim_ff=3072,
                 n_heads=12, variance_epsilon=1e-12,
                 *args, **kwargs):
        super().__init__()
        self.embed = Embeddings(vocab_size, max_len, n_segments, dim, p_drop_hidden)
        self.blocks = nn.ModuleList([Block(dim,
                                           dim_ff,
                                           n_heads,
                                           p_drop_hidden,
                                           p_drop_attn,
                                           variance_epsilon) for _ in range(n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h


class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"

    def __init__(self, vocab_size, max_len, n_segments, dim=768, n_layers=8,
                 p_drop_hidden=0.1, p_drop_attn=0.1, dim_ff=3072,
                 n_heads=12, variance_epsilon=1e-12,
                 *args, **kwargs):
        super().__init__()
        self.transformer = Transformer(vocab_size, max_len, n_segments, dim, n_layers,
                                       p_drop_hidden, p_drop_attn, dim_ff,
                                       n_heads, variance_epsilon)
        self.fc = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(dim, dim)
        self.activ2 = nn.GELU()
        self.norm = LayerNorm(dim)
        self.classifier = nn.Linear(dim, 2)
        # decoder is shared with embedding layer
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf
