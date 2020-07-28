import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
    

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn
    

class SelCa(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(SelCa, self).__init__()

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, args.d)
        self.user_random_fc = nn.Linear(args.d, args.d)
        self.item_embeddings = nn.Embedding(num_items, args.d, padding_idx=0)
        self.category_embeddings = nn.Embedding(args.topic_num, args.d)
        self.embedding_dropout = nn.Dropout(args.embedding_dropout)
        
        self.dims = args.d
        n_head = args.n_head
        d_k = self.dims // n_head
        d_v = self.dims // n_head
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(self.dims, self.dims*4, n_head, d_k, d_v, dropout=args.dropout) for _ in range(args.n_attn)])

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(args.L, self.dims, None), freeze=True)
        
        self.attention_vec = nn.Parameter(torch.FloatTensor([[1/args.L] for _ in range(args.L)]))
        self.attention_vec.requires_grad = True
        self.gate_fc = nn.Linear(self.dims*3, 1)

        self.W2 = nn.Embedding(num_items, self.dims*2)
        
        # dropout
        self.dropout = nn.Dropout(args.dropout)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.category_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        
        self.cache_x = None

    def forward(self, seq_var, user_var, item_var, probs, device, use_cache=False, for_pred=False):
        if not use_cache:
            # Embedding Look-up
            item_embs = self.item_embeddings(seq_var)
            item_embs = self.embedding_dropout(item_embs)
            user_emb = self.user_embeddings(user_var).squeeze(1)
            user_emb = self.embedding_dropout(user_emb)
            
            noise_vector = self.user_random_fc(torch.randn_like(user_emb).to(device))
            user_emb += noise_vector
            
            q = item_embs + self.position_enc.weight
            q_ = q
            
            for enc_layer in self.layer_stack:
                q, _ = enc_layer(q)
                q = q + q_
                q_ = q

            attn_vec = torch.softmax(self.attention_vec, dim=0)
            item_seq_vec = (attn_vec * q).sum(dim=1)
            
            # categorical vector
            category_embeddings = self.category_embeddings.weight
            category_embeddings = self.embedding_dropout(category_embeddings)
            categorical_vector = probs @ category_embeddings
            if categorical_vector.dim() == 1:
                categorical_vector = categorical_vector.reshape(1, self.dims)
            
            gate_input = torch.cat([categorical_vector, user_emb, categorical_vector * user_emb], dim=1)
            gate = torch.sigmoid(self.gate_fc(gate_input))
            x = self.dropout(gate * categorical_vector + (1 - gate) * item_seq_vec)
            x = torch.cat([x, user_emb], dim=1)
            
            self.cache_x = x

        else:
            x = self.cache_x

        w2 = self.W2(item_var)
        if not for_pred:
            results = []
            for i in range(item_var.size(1)):
                w2i = w2[:, i, :]
                result = (x * w2i).sum(1)
                results.append(result)
            res = torch.stack(results, 1)
        else:
            w2 = w2.squeeze()
            res = (x * w2).sum(1)

        return res
