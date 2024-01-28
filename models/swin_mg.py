# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import numpy as np
from scipy import ndimage

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


from .swin_transformer import SwinTransformer


def swish(x):
    return x * torch.sigmoid(x)

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.attention_dropout_rate)
        self.proj_dropout = Dropout(config.attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, k_v=None, attn_mask=None):

        is_cross_attention = k_v is not None

        # print('hidden_states size:', hidden_states.size())
        mixed_query_layer = self.query(hidden_states)
        if is_cross_attention: #
            # print('kv size:', k_v.size())
            # print('self.key:', self.key.weight.data.size())
            mixed_key_layer = self.key(k_v.cuda())
            mixed_value_layer = self.value(k_v)
        else: 
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attn_mask is not None:
            # print('before attn_mask:', attn_mask.size())
            # print(self.num_attention_heads)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)
            # print('attn_mask:', attn_mask.size())
            attention_scores = attention_scores.masked_fill_(attn_mask.bool(), -np.inf)

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config, vis):
        super(DecoderBlock, self).__init__()
        self.self_attn = Attention(config, vis)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.cross_attn = Attention(config, vis)
        self.cross_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x, encoder_y, self_attn_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, weights = self.self_attn(x, attn_mask=self_attn_mask)
        x = x + residual

        residual = x
        x = self.cross_attn_layer_norm(x)
        x, cross_weights = self.cross_attn(x, encoder_y)
        x = x + residual
        
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x, cross_weights 

class Decoder(nn.Module):
    def __init__(self, config, vis):
        super(Decoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.decoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.decoder_num_layers):
            layer = DecoderBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, encoder_y, seq_mask=None):
        weights_list = []
        for layer_block in self.layer:
            x, weights = layer_block(x, encoder_y, seq_mask)
            weights_list.append(weights)
        return x, weights_list

class LabelEmbeddings(nn.Module):
    """Construct the embeddings from label, position embeddings.
    """
    def __init__(self, config):
        super(LabelEmbeddings, self).__init__()
        self.tgt_vocab_size = config.num_labels
        self.emb_size = config.hidden_size
        self.embedding = nn.Embedding(self.tgt_vocab_size, self.emb_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.label_level, config.hidden_size))

        self.dropout = Dropout(config.dropout_rate)

    def forward(self, x):
        # print('x size:',x.size())
        # print('self.position_embeddings size',self.position_embeddings.size())
        x = self.embedding(x)
        # print('after embed x size:',x.size())
        embeddings = x + self.position_embeddings[:,0:x.size(1),:]
        embeddings = self.dropout(embeddings)
        return embeddings

class SwinMG(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        self.label_embeddings = LabelEmbeddings(config)
        self.decoder = Decoder(config, vis=True)
        self.linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        # self.criterion = nn.CrossEntropyLoss()
    

    def forward(self, x, labels_input):
        encoded = self.encoder.forward_features_wo_pool(x)
        # print('encoded size:', encoded.size())

        seq_mask = sequence_mask(labels_input).cuda()
        label_embedding_output = self.label_embeddings(labels_input)
        decode_out, weights = self.decoder(label_embedding_output, encoded, seq_mask)
        output = self.linear(decode_out)
        scores = output.view(-1, output.size(2))
        return scores, weights

    def generate(self, input_ids):

        encoded = self.encoder.forward_features_wo_pool(input_ids)
        # print('generate encoded size():', encoded.size())
        dec_input = torch.zeros(input_ids.size(0),0).to(torch.int64).cuda()
        terminal = False
        next_symbol = torch.zeros(input_ids.size(0),1).to(torch.int64).cuda()
        weights_list = []
        for _ in np.arange(4):
            dec_input = torch.cat([dec_input, next_symbol],-1)
            # print('dec_input:', dec_input)
            label_embedding_output = self.label_embeddings(dec_input)
            seq_mask = sequence_mask(dec_input).cuda()
            # print('test seq_mask', seq_mask.size())
            # print('label_embedding_output size:', label_embedding_output.size())
            decode_out, weights = self.decoder(label_embedding_output, encoded, seq_mask)
            weights_list.append(weights)
            output = self.linear(decode_out)
            # print('output size:', output.size())
            prob = output.max(dim=-1, keepdim=False)[1]
            # print('prob size:', prob.size())
            # print('prob:', prob)
            next_symbol = prob[:,-1].unsqueeze(-1)
            # print('next_symbol:', next_symbol)

        return torch.cat([dec_input[:,1:], next_symbol],-1),  weights_list
