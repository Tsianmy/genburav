import torch
from torch import nn

class RCE(nn.Module):
    def __init__(self,num_cls,ratio=1):
        super().__init__()
        self.num_cls=num_cls
        self.ratio=ratio

    def forward(self,input,target):
        reverse_logits=-input
        reverse_logits=torch.exp(reverse_logits)/(
            (torch.sum(torch.exp(reverse_logits),dim=-1)).view(-1,1)
        )
        reverse_logits=reverse_logits.unsqueeze(-1)
        batch_size=input.size(0)
        target = target.argmin(1)
        reverse_lable=torch.ones(size=(batch_size,self.num_cls),dtype=torch.float)*(self.ratio/(self.num_cls-1))
        reverse_lable=reverse_lable.cuda()

        for idx,label in enumerate(target):
            reverse_lable[idx,label]=0.
        reverse_lable=reverse_lable.unsqueeze(1)
        entropy=-torch.bmm(reverse_lable,torch.log(reverse_logits))
        return torch.mean(
            entropy
        )