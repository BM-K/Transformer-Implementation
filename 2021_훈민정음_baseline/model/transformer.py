import torch
import torch.nn as nn
import math
from transformers import BertModel, BertConfig
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


def get_attn_pad_mask(args, seq_q, seq_k, pad_idx):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k).to(args.device)


def get_attn_subsequent_mask(args, seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask.to(args.device)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = int(args.d_model / args.n_heads)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        scores.masked_fill_(attn_mask, -1e9)
        last_attention_weight = scores

        attn = self.dropout(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)

        return context, attn, last_attention_weight


class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.args = args
        self.d_k = int(args.d_model / args.n_heads)
        self.d_v = int(args.d_model / args.n_heads)
        self.n_heads = args.n_heads
        self.W_Q = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_K = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_V = nn.Linear(args.d_model, self.d_v * args.n_heads)
        self.li1 = nn.Linear(args.n_heads * self.d_v, args.d_model)
        #self.layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, Q, K, V, attn_mask, embeddings=None):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)

        context, attn, last_attention_weight = ScaledDotProductAttention(self.args)(q_s, k_s, v_s, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.li1(context)

        return output + residual, attn, last_attention_weight


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.feedforward, out_channels=args.d_model, kernel_size=1)
        #self.layer_norm = nn.LayerNorm(args.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        residual = inputs
        output = self.dropout(self.relu(self.conv1(inputs.transpose(1, 2))))
        output = self.dropout(self.conv2(output).transpose(1, 2))

        return output + residual


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.args = args
        self.enc_self_attn = MultiheadAttention(self.args)
        self.pos_ffn = PoswiseFeedForwardNet(self.args)
        self.layer_norm1 = nn.LayerNorm(args.d_model)
        self.layer_norm2 = nn.LayerNorm(args.d_model)

    def forward(self, enc_inputs, enc_self_attn_mask, embeddings):
        enc_inputs = self.layer_norm1(enc_inputs)
        enc_outputs, attn, _ = self.enc_self_attn(enc_inputs,
                                                  enc_inputs,
                                                  enc_inputs,
                                                  enc_self_attn_mask,
                                                  embeddings)
        enc_outputs = self.layer_norm2(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiheadAttention(args)
        self.dec_enc_attn = MultiheadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)
        self.layer_norm1 = nn.LayerNorm(args.d_model)
        self.layer_norm2 = nn.LayerNorm(args.d_model)
        self.layer_norm3 = nn.LayerNorm(args.d_model)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask, embeddings):
        dec_inputs = self.layer_norm1(dec_inputs)
        dec_outputs, dec_self_attn, last_attention_weight = self.dec_self_attn(dec_inputs,
                                                                               dec_inputs,
                                                                               dec_inputs,
                                                                               dec_self_attn_mask,
                                                                               embeddings)
        dec_outputs = self.layer_norm2(dec_outputs)
        dec_outputs, dec_enc_attn, last_attention_weight = self.dec_enc_attn(dec_outputs,
                                                                             enc_outputs,
                                                                             enc_outputs,
                                                                             dec_enc_attn_mask)
        dec_outputs = self.layer_norm3(dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, last_attention_weight


class Decoder(nn.Module):
    def __init__(self, args, vocab_size, pad_ids):
        super(Decoder, self).__init__()
        self.args = args
        self.pad_ids = pad_ids
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.src_emb = nn.Embedding(vocab_size, args.d_model)
        self.pos_embedding = PositionalEncoding(args.d_model, args.max_len)
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(self.args.d_model)

    def forward(self, enc_inputs, dec_inputs, enc_outputs):
        dec_outputs = self.src_emb(dec_inputs) + self.pos_embedding(dec_inputs)
        dec_outputs = self.dropout(self.layer_norm(dec_outputs))
        embeddings = dec_outputs

        dec_self_attn_pad_mask = get_attn_pad_mask(self.args, dec_inputs, dec_inputs, self.pad_ids)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(self.args, dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(self.args, dec_inputs, enc_inputs, self.pad_ids)

        for layer in self.layers:
            dec_outputs, last_attention_weight = layer(
                dec_outputs,
                enc_outputs,
                dec_self_attn_mask,
                dec_enc_attn_mask,
                embeddings)

        return dec_outputs, last_attention_weight


class Encoder(nn.Module):
    def __init__(self, args, vocab_size, pad_ids):
        super(Encoder, self).__init__()

        self.args = args
        self.pad_ids = pad_ids
        self.d_model = args.d_model

        self.src_emb = nn.Embedding(vocab_size, self.d_model)
        self.pos_embedding = PositionalEncoding(self.d_model, args.max_len)
        self.layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(self.args.n_layers)])

        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs) + self.pos_embedding(enc_inputs)
        enc_outputs = self.dropout(self.layer_norm(enc_outputs))
        embeddings = enc_outputs

        enc_self_attn_mask = get_attn_pad_mask(self.args, enc_inputs, enc_inputs, self.pad_ids)

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask, embeddings)

        return enc_outputs


class Transformer(nn.Module):
    def __init__(self, args, vocabsize):
        super(Transformer, self).__init__()
        self.pad_ids = 1
        self.vocab_size = vocabsize

        self.encoder = Encoder(args, self.vocab_size, self.pad_ids)
        self.decoder = Decoder(args, self.vocab_size, self.pad_ids)
        self.projection = nn.Linear(args.d_model, self.vocab_size)

    def forward(self, enc_inputs, dec_input):

        enc_outputs = self.encoder(enc_inputs)
        dec_outputs, last_attention_weight = self.decoder(enc_inputs, dec_input, enc_outputs)
        last_hidden_state = self.projection(dec_outputs)

        return last_hidden_state.view(-1, self.vocab_size)