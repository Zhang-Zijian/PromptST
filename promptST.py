import torch
import torch.nn as nn
import math
from torch.nn.modules.normalization import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0,max_len = 500):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)

class TemporalPositionalEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0,max_len = 300):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.transpose(0,1).unsqueeze(1).unsqueeze(0)
        x = x + weight[:,:,:,:x.shape[-1]]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, in_dim, layers=1, dropout = 0):
        super(TemporalEmbedding, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim,hidden_size=in_dim,num_layers=layers,dropout=dropout)

    def forward(self, input):
        ori_shape = input.shape
        x = input.permute(3, 0, 2, 1)
        x = x.reshape(ori_shape[3], ori_shape[0] * ori_shape[2], ori_shape[1])
        x,_ = self.rnn(x)
        x = x.reshape(ori_shape[3], ori_shape[0], ori_shape[2], ori_shape[1])
        x = x.permute(1, 3, 2, 0)
        return x

class TemporalEmbeddingTransformer(nn.Module):
    def __init__(self, in_dim, layers=2, mlp_ratio=4, num_head=4, dropout = 0):
        super(TemporalEmbeddingTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(in_dim, num_head, in_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers)
        # self.rnn = nn.LSTM(input_size=in_dim,hidden_size=in_dim,num_layers=layers,dropout=dropout)

    def forward(self, input):
        ori_shape = input.shape
        x = input.permute(3, 0, 2, 1)
        x = x.reshape(ori_shape[3], ori_shape[0] * ori_shape[2], ori_shape[1])
        x = self.transformer_encoder(x)
        x = x.reshape(ori_shape[3], ori_shape[0], ori_shape[2], ori_shape[1])
        x = x.permute(1, 3, 2, 0)
        return x

class SpatialEncoder(nn.Module):
    def __init__(self,in_dim,layers=1,dropout=0,heads=8):
        super(SpatialEncoder, self).__init__()
        self.heads = heads
        self.pos = PositionalEncoding(in_dim,dropout=dropout)
        self.lpos = LearnedPositionalEncoding(in_dim, dropout=dropout)
        _encoder_layers = TransformerEncoderLayer(in_dim, heads, in_dim*4, dropout)
        self.trans = TransformerEncoder(_encoder_layers, layers)

    def forward(self,x):
        x = x.permute(1,0,2)
        x = self.pos(x)
        x = self.lpos(x)
        x = self.trans(x)
        return x.permute(1,0,2)

    def _gen_mask(self,input):
        l = input.shape[1]
        mask = torch.eye(l)
        mask = mask.bool()
        return mask

class ttnet(nn.Module):
    def __init__(self, dropout, in_dim, out_dim=12, hid_dim=32, ts_depth_spa=6, ts_depth_tem=6, pmt_pretrain=False):
        super(ttnet, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=hid_dim,
                                    kernel_size=(1, 1))
        self.start_embedding_list = [
            TemporalPositionalEncoding(hid_dim, dropout=dropout),
            TemporalEmbeddingTransformer(hid_dim, layers=ts_depth_tem, dropout=dropout),
            ]
        self.start_embedding = nn.Sequential(*self.start_embedding_list)
        self.network = SpatialEncoder(in_dim=hid_dim, layers=ts_depth_spa, dropout=dropout)
        self.end_conv = nn.Linear(hid_dim, out_dim)
        # self.end_conv = nn.ModuleList([nn.Linear(hid_dim, out_dim) for _ in range(in_dim)])
        # self.attr_bn = nn.BatchNorm2d(in_dim)

    def forward(self, input):
        input = input.permute(0, 3, 2, 1)
        input_shape = input.shape
        # bs, in_dim, num_nodes, in_cnl 32, 2, 207, 12
        # input = self.attr_bn(input)

        for i in range(input_shape[1]):
            x = self.start_conv(input[:,i,:,:].unsqueeze(1))
            # bs, emb_size, num_nodes, in_cnl 32, 64, 207, 12
            # print(f'conv shape: {x.shape}')
            x = self.start_embedding(x)[..., -1]
            # bs, emb_size, num_nodes
            x = x.transpose(1, 2)
            # bs, num_nodes, emb_size
            x = self.network(x)
            # bs, num_nodes, emb_size
            x = self.end_conv(x)
            # x = self.end_conv[i](x)
            # bs, num_nodes, out_cnl
            x = x.unsqueeze(-1).transpose(1, 2)
            # bs, out_cnl, num_nodes, 1
            if i == 0:
                output = x
            else:
                output = torch.cat((output, x), dim=-1)
        return output


 
class prompt_ttnet(ttnet):
    def __init__(self, dropout, num_attr_spa_pmt, num_attr_temp_pmt, num_st_pmt, num_nodes, \
        in_dim=2, out_dim=12, hid_dim=32, ts_depth_spa=6, ts_depth_tem=6, basic_state_dict=None):
        super().__init__(dropout, in_dim, out_dim, hid_dim, ts_depth_spa, ts_depth_tem)

        self.basic_state_dict = basic_state_dict
        self.ts_depth_spa = ts_depth_spa
        self.ts_depth_tem = ts_depth_tem
        self.hid_dim = hid_dim
        self.num_attr_spa_pmt = num_attr_spa_pmt
        self.num_attr_temp_pmt = num_attr_temp_pmt
        self.num_st_pmt = num_st_pmt
        self.num_nodes = num_nodes
        self.single_end_conv = nn.Linear(hid_dim, out_dim)
        self.reset_head()

    def load_basic(self):
        if self.basic_state_dict is not None:
            self.load_state_dict(self.basic_state_dict, False)
            print(f'load basic state dict')
        else:
            assert 1==0, 'basic state dict load failed.'

    def init_pmt(self, pmt_init_type):

        def init_module(w, pmt_init_type):

            if pmt_init_type == 'none':
                pass
            elif pmt_init_type == 'xuni':
                a = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(w, gain=a)
            elif pmt_init_type == 'xnor':
                nn.init.xavier_normal_(w)
            elif pmt_init_type == 'uni':
                nn.init.uniform_(w)
            elif pmt_init_type == 'nor':
                nn.init.normal_(w)
            elif pmt_init_type == 'kuni':
                nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
            elif pmt_init_type == 'knor':
                nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

        if self.num_attr_spa_pmt > 0:
            self.attr_spa_prompt_tokens = nn.Parameter(torch.ones((self.ts_depth_spa), self.num_attr_spa_pmt, self.hid_dim))
            init_module(self.attr_spa_prompt_tokens, pmt_init_type)
            # print(f'attr spa: {self.attr_spa_prompt_tokens}')

        if self.num_attr_temp_pmt > 0:
            self.attr_temp_prompt_tokens = nn.Parameter(torch.ones((self.ts_depth_tem), self.num_attr_temp_pmt, self.hid_dim))
            init_module(self.attr_temp_prompt_tokens, pmt_init_type)
            # print(f'attr temp: {self.attr_temp_prompt_tokens}')

        if self.num_st_pmt > 0:
            self.st_prompt_tokens = nn.Parameter(torch.ones((self.ts_depth_tem), self.num_st_pmt, self.num_nodes, self.hid_dim))
            init_module(self.st_prompt_tokens, pmt_init_type)
            # print(f'st pmt: {self.st_prompt_tokens}')


    def reset_head(self):
        nn.init.xavier_normal_(self.single_end_conv.weight)
        nn.init.constant_(self.single_end_conv.bias, 0)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        if self.num_attr_spa_pmt > 0:
            self.attr_spa_prompt_tokens.requires_grad = True
        if self.num_attr_temp_pmt > 0:
            self.attr_temp_prompt_tokens.requires_grad = True
        if self.num_st_pmt > 0:
            self.st_prompt_tokens.requires_grad = True

        for param in self.single_end_conv.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        # bs, in_dim, num_nodes, in_cnl 32, 2, 207, 12
        # x = self.attr_bn(x)
        x = self.start_conv(x)
        # bs, emb_size, num_nodes, in_cnl 32, 64, 207, 12

        if self.num_st_pmt > 0 or self.num_attr_temp_pmt > 0:

            x = self.start_embedding_list[0](x)

            model_layers = self.start_embedding_list[1].transformer_encoder.layers
            tem_num_pmt = 0
            if self.num_st_pmt > 0:
                assert len(model_layers) == self.ts_depth_tem, f'{len(model_layers)}, {self.ts_depth_tem}'
                tem_num_pmt += self.st_prompt_tokens.shape[1]
            if self.num_attr_temp_pmt > 0:
                assert len(model_layers) == len(self.attr_temp_prompt_tokens), f'{len(model_layers)}, {len(self.attr_temp_prompt_tokens)}'
                tem_num_pmt += self.attr_temp_prompt_tokens.shape[1]

            for i in range(self.ts_depth_tem):
                if self.num_st_pmt > 0:
                    _st_prompt_tokens = self.st_prompt_tokens[i].unsqueeze(0)
                    _st_pmt_shape = _st_prompt_tokens.shape
                    _st_prompt_tokens =  _st_prompt_tokens.expand(x.shape[0], _st_pmt_shape[1], _st_pmt_shape[2], _st_pmt_shape[3]).transpose(1,3)

                    x = torch.cat((x, _st_prompt_tokens), dim=-1)

                if self.num_attr_temp_pmt > 0:
                    _attr_temp_prompt_tokens = self.attr_temp_prompt_tokens[i].unsqueeze(0).unsqueeze(-1)
                    _att_pmt_shape = _attr_temp_prompt_tokens.shape
                    _attr_temp_prompt_tokens =  _attr_temp_prompt_tokens.expand(x.shape[0], _att_pmt_shape[1], _att_pmt_shape[2], x.shape[2]).permute(0, 2, 3, 1)

                    x = torch.cat((x, _attr_temp_prompt_tokens), dim=-1)

                ori_shape = x.shape
                x = x.permute(3, 0, 2, 1)
                x = x.reshape(ori_shape[3], ori_shape[0] * ori_shape[2], ori_shape[1])
                x = model_layers[i](x)
                x = x.reshape(ori_shape[3], ori_shape[0], ori_shape[2], ori_shape[1])
                x = x.permute(1, 3, 2, 0) 
                len_pmt_x = x.shape[-1]
                x = x[:,:,:,:len_pmt_x-tem_num_pmt]

            x = x[..., -1]

        else:
            x = self.start_embedding(x)[..., -1]
        # bs, emb_size, num_nodes
        x = x.transpose(1, 2)
        # bs, num_nodes, emb_size

        x_shape = x.shape

        x = x.permute(1,0,2)

        # num_nodes, batch_size, embedding_size
        x = self.network.pos(x)
        x = self.network.lpos(x)

        if self.num_attr_spa_pmt > 0:
            num_pmt = self.attr_spa_prompt_tokens.shape[1]
            pmt_shape = self.attr_spa_prompt_tokens.shape
            # ts_depth*2, num_pmt, embedding_size
            _attr_spa_prompt_tokens = self.attr_spa_prompt_tokens.unsqueeze(1).expand(pmt_shape[0], x.shape[1], pmt_shape[1], pmt_shape[2])
            _attr_spa_prompt_tokens = _attr_spa_prompt_tokens.permute(0, 2, 1, 3)
            pmt_norm = LayerNorm(self.hid_dim, eps=1e-5).to(x.device)
            # ts_depth*2, num_pmt, batch_size, embedding_size

            encoder_norm = LayerNorm(self.hid_dim, eps=1e-5).to(x.device)
            decoder_norm = LayerNorm(self.hid_dim, eps=1e-5).to(x.device)

            memory = x
            model_layers = self.network.trans.layers
            assert len(model_layers) == len(_attr_spa_prompt_tokens) ==self.ts_depth_spa, f'{len(model_layers)}, {self.ts_depth_spa}, {len(_attr_spa_prompt_tokens)}'
            for (mod, pmt) in zip(model_layers, _attr_spa_prompt_tokens):
                memory = torch.cat((memory, pmt), dim=0)
                memory = pmt_norm(memory)
                memory = mod(memory)
                num_tokens = memory.shape[0]
                memory = memory[:num_tokens-num_pmt,:,:]

            memory = encoder_norm(memory)

            x = memory

            x = x.permute(1,0,2)
        else:
            x = self.network.trans(x)
            x = x.permute(1,0,2)

        x = self.single_end_conv(x)
        x = x.unsqueeze(-1).transpose(1, 2)
        return x
