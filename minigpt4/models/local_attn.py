import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAttention(nn.Module):
    """ 
        Global Attention module 
    """
    def __init__(self, img_dim, txt_dim, dim):
        super(LocalAttention, self).__init__()
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.txt = nn.Linear(txt_dim, dim)
        self.img = nn.Linear(img_dim, dim)

    def forward(self, images, texts, mask=None):
        b, l, d = texts.shape  # [batch_size, sequence_len, dim] [6, 48, 4096] -> CLIP [6, 48, 512]
        b, n, c = images.shape  # [batch_size, patch_num, dim] , patch_num = h*w , dim = channel [6, 1025, 1408]
        texts = self.txt(texts) # [batch_size, sequence_len, dim] [6, 48, 1408]
        images = self.img(images) # [batch_size, patch_num, dim]  patch_num = h*w [6, 1025, 1408]     
        dots = torch.einsum('bid,bjd->bij', texts, images)  # [batch_szie, sequence_len, patch_num] [6, 48, 1025] 
        # dots = dots / torch.max(dots)    # add for ood    
        dots = torch.einsum('bid,bjd->bij', texts, images) * self.scale  # [batch_szie, sequence_len, patch_num]

        if mask is not None:
            dots.masked_fill(~mask, float('-inf'))

        flatten_dots = dots.view(b, -1) # [batch_size, sequence_len * patch_num] [6, 49200]
        norm_dots = flatten_dots.softmax(dim=1).view(b, l, n) # [batch_size, sequence_len, patch_num] [6, 48, 1025]
        local_attn = torch.sum(norm_dots, dim=1, keepdim=True) # [batch_size, 1, patch_num] [6, 1, 1025]
        local_attn = local_attn.view(b, n) # [batch_size, patch_num] # 2024/5/11 1024
        
        out = local_attn.unsqueeze(-1) * images # [batch_size, patch_num, dim] [6, 1025, 1408]
        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w) # if x.shape is: b, c, h, w

        return out, norm_dots, local_attn # ZY 2024/4/3 add for visualization
        

class MultiHeadLocalAttention(nn.Module):
    """
        Multi-Head Local Attention module
    """
    def __init__(self, img_dim, txt_dim, dim, heads = 8):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.heads = heads
        
        assert dim % heads == 0, 'dim must be divisible by num_heads'
        self.dim_per_head = dim // heads
        
        self.query = nn.Linear(txt_dim, dim)
        self.key = nn.Linear(img_dim, dim)
        self.value = nn.Linear(img_dim, dim)
    
    def forward(self, queries, keys, values, mask=None):
        b, n, _, h = *queries.shape, self.heads # [batch_size, sequence_length, dim, num_heads]
        
        queries = self.query(queries) # [batch_size, seq_len, dim]
        keys = self.key(keys)   # [batch_size, h*w, dim]
        values = self.value(values) # [batch_size, h*w, dim]

        queries = queries.view(b, -1, h, self.dim_per_head).transpose(1, 2) # [batch_size, num_heads, sequence_length, dim_per_head]
        keys = keys.view(b, -1, h, self.dim_per_head).transpose(1, 2) # [batch_size, num_heads, h*w, dim_per_head]
        values = values.view(b, -1, h, self.dim_per_head).transpose(1, 2) # [batch_size, num_heads, h*w, dim_per_head]
        
        # dots = torch.matmul(queries, keys.transpose(-2,-1)) * self.scale
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale # [batch_szie, num_heads, sequence_length, h*w]

        if mask is not None:
            # mask = F.pad(mask.flatten(1), (1,0), value=True)
            mask = mask.unsqueeze(1).repeat(1, h, 1, 1)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrrect dimensions'
            # mask = mask[:, None, :].expend(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
        attn = torch.softmax(dots, dim=-1) # [batch_size, num_heads, sequence_length, dim_per_heads]
        out = torch.einsum('bhij,bhjd->bhid', attn, values)  # [batch_size, num_heads, sequence_length, dim_per_heads]
        out = out.transpose(1,2).contiguous().view(b, n, -1)
        
        return out

if __name__ == '__main__':
    batch_size = 3
    seq_len = 5
    txt_dim = 4096
    h = 10
    w = 10
    img_dim = 1408
    dim = 1408
    
    # input_ids = torch.tensor([[100, 200, 300, 300, 0],
    #              [22, 33, 44, 0, 0],
    #              [66, 55, 66, 30, 0]], dtype=torch.long)
    # mask = input_ids.eq(0)  #  # 逻辑矩阵mask：将填充位置标记为True，其他位置标记为False
    # mask = mask.unsqueeze(1).expand(batch_size, seq_len, h*w)
    
    mask = torch.rand(batch_size, seq_len, h*w) > 0.5
    # img: [b, c, h, w]->[b, hw, c]
    input_img = torch.randn((batch_size, h*w, img_dim))
    input_txt = torch.randn((batch_size, seq_len, txt_dim))
    
    global_att = LocalAttention(img_dim=img_dim, txt_dim=txt_dim, dim=dim)
    print(f'global_att module:\n{global_att}')
    out = global_att(images=input_img, texts=input_txt, mask=mask)
    print(out.shape)

