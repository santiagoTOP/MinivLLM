import torch 
import torch.nn as nn


class SamplerLayer(nn.Module):
    """
    A custom sampler layer that selects elements from the input tensor
    based on provided indices.
    """

    def __init__(self):
        super().__init__()

    @torch.compile  # 优化gpu 计算
    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        # logits: (batch_size, vocab_size)
        # temperature：（batch_size，）
        logits/= temperature.unsqueeze(-1) # 根据温度来调整 logits 的分布，温度小于 1 拉开差距，选择更加稳定，温度大于 1 缩小差距，选择更加均匀
        probs = torch.softmax(logits, dim=-1) # 将得分转化为概率分布
        # 概率：       [0.1, 0.6, 0.3]
        # 某次随机数：  [0.5, 1.0, 0.2]
        # 相除得到：    [0.2, 0.6, 1.5]
        # 下面这段代码的作用是用来保证每个 token 被采样到的概率等于他的真实概率，比如 10%，60%，30%，避免变成贪心解码
        # exponential_(1)表示指数随机数的的速率为 1
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens