import torch
import torch.nn as nn

from . import model_utils

class BOREP(nn.Module):

    def __init__(self, params):
        super(BOREP, self).__init__()
        self.params = params
        self.max_seq_len = params.max_seq_len
        self.proj = self.get_projection()
        if params.gpu:
            self.cuda()

         
    def get_projection(self):
        proj = nn.Linear(self.params.input_dim, self.params.output_dim)
        if self.params.init == "normal":
            nn.init.normal_(proj.weight, std=0.1)
        elif self.params.init == "uniform":
            nn.init.uniform_(proj.weight, a=-0.1, b=0.1)
        elif self.params.init == "kaiming":
            nn.init.kaiming_uniform_(proj.weight)
        elif self.params.init == "xavier":
            nn.init.xavier_uniform_(proj.weight)

        nn.init.constant_(proj.bias, 0)

        if self.params.gpu:
            proj = proj.cuda()
        return proj

    
    def borep(self, x):
        batch_sz, seq_len = x.size(1), x.size(0)
        out = torch.FloatTensor(seq_len, batch_sz, self.params.output_dim).zero_()
        for i in range(seq_len):
            out[i] = self.proj(x[i])
        return out

    
    def forward(self, batch, se_params):
        lengths, out = model_utils.embed(batch, self.params, se_params)

        out = self.borep(out)
        out = model_utils.pool(out, lengths, self.params)
        
        return out

    
    def encode(self, batch, params):
        return self.forward(batch, params).cpu().detach().numpy()