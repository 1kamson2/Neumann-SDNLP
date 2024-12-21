import torch
import torch.nn as nn
import torch.nn.functional as F


class NLPEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # [FIX] todo: Check initialization, because it doesnt have to be correct.
        self.mha = MultiHeadAttention(in_features, in_features, in_features, 8)
        self.ffn = FeedForwardNetwork(in_features, 2 * in_features)

    def forward(self, ex, q, k, v, valid_lens):
        out = ex + self.mha(q, k, v, valid_lens)
        out = nn.LayerNorm(out.shape[-1])(out)
        out = out + self.ffn(out)
        return out.to(self.device)


class NLPDecoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.mha = MultiHeadAttention(in_features, in_features, in_features, 8)
        self.ffn = FeedForwardNetwork(in_features, 2 * in_features)

    def forward(self, dx, ex):
        out = dx + self.mha(ex, ex, dx, None)
        out = nn.LayerNorm(out.shape[-1])(out)
        # [FIX] ?? shoud add to k / v / q ????????????
        # [FIX] Layer norm seems retarded
        out = out + self.mha(out + ex, ex, dx, None)
        out = nn.LayerNorm(out.shape[-1])(out)
        out = out + self.ffn(out)
        out = nn.LayerNorm(out.shape[-1])(out)
        return out.to(self.device)


class Transformer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.encoder = NLPEncoder(in_features, out_features)
        self.decoder = NLPDecoder(in_features)
        # [FIX] certainly bad
        self.lin = nn.Linear(16, 16)

    def forward(self, dx, ex):
        print("Encoder start.")
        ex = self.encoder(ex, ex, ex, dx, None)
        print("Encoder stop.")
        print("Decoder start.")

        # [FIX] Fix this hardcode, it should depend on batch size, also this is
        # the idea of transformer? I think not, rethink this.
        # [FIX] It doesnt work for now with bigger batches.
        # out = self.decoder(dx, ex.expand(-1, -1, 31, -1))
        out = self.decoder(dx, ex)
        print("Decoder stop.")
        out = self.lin(out)
        out = nn.Softmax()(out)
        return out.to(self.device)


"""
[FIX]
100% WON'T WORK THIS SHIT BELOW, FUCKING PIECE OF SHIT
INPUT SHOULD BE THE LENGTH OF THE EMBEDDING IN THIS CASE IT IS 7 OR 8
IF IT WAS ONE HOT ENCODING IT WOULD BE 128 OR 256
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
"""


class MultiHeadAttention(nn.Module):
    def __init__(self, k_sz, q_sz, v_sz, num_heads, dropout=0.1, bias=False):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_heads = num_heads
        # Weights of each layer
        self.WL_k = nn.Linear(k_sz, k_sz * num_heads, bias=bias)
        self.WL_q = nn.Linear(q_sz, q_sz * num_heads, bias=bias)
        self.WL_v = nn.Linear(v_sz, v_sz * num_heads, bias=bias)
        self.dropout = dropout
        # Output layer - fully connected layer
        # [FIX] should use d_model = 7 or 8 for better clarity
        self.fc = nn.Linear(524288, v_sz, bias=bias)

    def forward(self, q, k, v, valid_lens=None):
        q = self._transpose(self.WL_q(q), self.num_heads)
        k = self._transpose(self.WL_k(k), self.num_heads)
        v = self._transpose(self.WL_v(v), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )
        # [FIX] For now use this, but later do your own version
        print(q.shape, k.shape, v.shape)
        output = F.scaled_dot_product_attention(q, k, v, valid_lens, self.dropout)
        output_concat = self._transpose_output(output, self.num_heads)
        output_concat = self.fc(output_concat)
        return output_concat.to(self.device)

    def _transpose(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def _transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidout, dropout=0.1):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.w1 = nn.Linear(dim_in, dim_hidout)
        self.w2 = nn.Linear(dim_hidout, dim_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.w2((self.w1(x)))
        out = self.dropout(out)
        return out.to(self.device)


"""
TODO: Add masked attention, for now skipped

"""
