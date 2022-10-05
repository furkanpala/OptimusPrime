import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_feat=512, out_feat=64) -> None:
        super().__init__()
        self.in_feat = in_feat,
        self.out_feat = out_feat
        self.value_fc = nn.Linear(in_feat, out_feat)
        self.query_fc = nn.Linear(in_feat, out_feat)
        self.key_fc = nn.Linear(in_feat, out_feat)


    def forward(self, value, key, query, mask=None):
        """
            value: torch.Tensor of shape (N, seq_len, in_feat)
            key: torch.Tensor of shape (N, seq_len, in_feat)
            query: torch.Tensor of shape (N, seq_len, in_feat)
            mask: torch.Tensor of shape (N, seq_len, seq_len)
        """
        value = self.value_fc(value) # N, seq_len, out_feat
        key = self.key_fc(key)
        query = self.query_fc(query)
        weights = torch.bmm(query, torch.transpose(key, 1, 2))
        weights /= torch.sqrt(torch.tensor(self.out_feat)) # N, seq_len, seq_len
        if mask:
            weights *= self.mask
        weights = torch.softmax(weights, dim=1)
        return torch.bmm(weights, value)



def test():
    x = torch.randn(8, 20, 512)
    sa = SelfAttention()
    y = sa(x,x,x)
    print(y.shape)

if __name__ == "__main__":
    test()
