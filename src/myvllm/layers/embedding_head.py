import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from myvllm.utils import get_context


# vocabparallelembedding
# shard over the number of vocab, not the embedding size

class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_size = dist.get_world_size() # 全局有多少个掌控GPU，即会将 embedding 分配到多少个GPU上并行
        self.tp_rank = dist.get_rank() # 当前 GPU 的 rank，即当前 GPU 在全局中的编号

        # keep the original num_embeddings
        self.num_embeddings = num_embeddings  # 词表大小
        # pad to make it divisible by tp_size
        """
        # 向上取整，计算每个进程需要多少行，在词表的维度上
        rows_per_rank = (num_embeddings + self.tp_size - 1) // self.tp_size

        # 得到补齐后的总行数
        padded_num_embeddings = rows_per_rank * self.tp_size
        """
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        # this is the num_embeddings per partition in this current GPU
        # 得到每个 gpu 上可以分配的词表大小
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim  # 每个词或者每个 token 的隐藏状态维度

        # 在每个 gpu 上初始化一个空的参数，用于存储该 gpu 上的词表权重
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 从完整的词表权重中加载当前 gpu 上的词表权重
        # 是给参数附加一个自定义加载方法，供外部加载器调用。它不是 PyTorch 自动执行的加载钩子，必须由加载代码显式调用。
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        loaded weights: [vocab_size, hidden_size],表示在 checkpoint 中的词表权重
        param: [vocab_size_per_partition, hidden_size],表示当前 gpu 上的词表权重
        """
        param_data = param.data # 当前 gpu 上的词表权重，形状为 [vocab_size_per_partition, hidden_size]

        offset = self.tp_rank * self.num_embeddings_per_partition # 当前 gpu 上的词表权重在全局词表中的起始位置
        shard_size = self.num_embeddings_per_partition # 当前 gpu 上的词表权重的大小

        # calculate how much of the original vocab falls in this partition
        actual_start = min(offset, self.num_embeddings) # 当前 gpu 上的词表权重在全局词表中的起始位置
        actual_end = min(offset + shard_size, self.num_embeddings) # 当前 gpu 上的词表权重在全局词表中的结束位置
        actual_size = max(0, actual_end - actual_start) # 当前 gpu 上的词表权重的大小

        if actual_size > 0:
            # load the actual weights
            # 从 checkpoint 中加载当前 gpu 上的词表权重，这里的 0 表示词表的维度
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size) 
            # 将加载的词表权重赋值给当前 gpu 上的词表权重
            param_data[:actual_size].copy_(sharded_weights)

        # pad the rest with zeros if needed
        if actual_size < shard_size:
            # 如果当前 gpu 上的词表权重的大小小于 shard_size，说明当前 gpu 上的词表权重的大小不足
            # 所以需要将当前 gpu 上的词表权重的剩余部分填充为 0
            # 这是为了在后续的计算中，当前 gpu 上的词表权重的大小与全局词表的大小一致
            param_data[actual_size:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mask for tokens in this partition's range and within original vocab size
        # x 常规输入是 [batch_size, seq_len], 但是在本项目中是将其压缩到了 [batch_size * seq_len]

        # 例：真实词表大小为 10，3 个进程补齐到 12 行，每个分片 4 行，embedding 维度为 2。
        # rank 1 保存全局 token 4～7，输入 x = [1, 4, 6, 9]。
        # 此时条件为 (x >= 4) & (x < 8) & (x < 10)，mask = [False, True, True, False]。
        # x < num_embeddings 排除补齐位置：例如 rank 2 的范围为 [8, 12)，但 ID 10、11 无效。
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
               (x < self.num_embeddings)
        # 减去分片起点 4 得到 [-3, 0, 2, 5]，乘 mask 后得到本地索引 [0, 0, 2, 0]。
        # token 4、6 对应本地第 0、2 行；不属于本分片的 token 1、9 暂用第 0 行，避免越界。
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        # 假设本地 weight = [[40, 41], [50, 51], [60, 61], [70, 71]]，分别对应 token 4～7。
        # 查表结果为 [[40, 41], [40, 41], [60, 61], [40, 41]]，其中第 1、4 个向量是占位结果。
        output = F.embedding(x, self.weight)

        if dist.get_world_size() > 1:
            # need to mask again, otherwise the embedding for the out-of-range ids will be the embedding of id 0
            # mask 从 [4] 扩展为 [4, 1]，清零占位向量，得到 [[0, 0], [40, 41], [60, 61], [0, 0]]。
            output = mask.unsqueeze(1) * output
            # 各进程逐元素求和；token 1、9 的向量由其他进程提供，最终每个进程都得到完整结果。
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

# weight tying with embedding layer
# LM Head 将模型输出的隐藏向量映射为词表得分（logits），供后续采样器选择下一个 token。
# 继承 VocabParallelEmbedding 是为了复用词表分片、weight 和 weight_loader；forward 在这里被覆盖。
# 两者都使用 [本地词表行数, hidden_size] 的权重，但 embedding 是按 ID 查行，LM Head 是做点积。
# 继承本身不会与模型的 embedding 共享参数；模型需显式执行 lm_head.weight = embed_tokens.weight。
# 以下例子：真实词表大小 V=10，tp_size=3，hidden_size=2，补齐后每个进程保存 4 行。
# 假设正确加载的完整权重满足 W[i] = [10*i, 10*i+1]（i=0～9），补齐的两行全零。
# rank 0 保存 token 0～3；rank 1 保存 token 4～7；rank 2 保存 token 8、9 和两行补齐权重。
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # 父类初始化后，本进程的 weight 形状为 [4, 2]，并附带按词表行切分的加载方法。
        super().__init__(num_embeddings, embedding_dim)

    # x 是 Transformer 输出的浮点隐藏向量，不是整数 token ID。
    # 本项目 prefill 输入形状为 [本轮拼接的 token 数 N, hidden_size]，decode 为 [batch_size, hidden_size]。
    # weight: [vocab_size_per_partition, hidden_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取当前推理阶段及拼接序列的边界信息；各并行进程处理相同的请求和隐藏向量。
        context = get_context()
        if context.is_prefill:
            # 例：batch 中有 A、B 两条请求，本轮分别处理 2、3 个 token，cu_seqlens_q = [0, 2, 5]。
            # 拼接后的 x = [[9, 9], [1, 2], [8, 8], [7, 7], [3, 1]]，形状为 [5, 2]。

            # A 对应 x[0:2]，B 对应 x[2:5]；生成下一个 token 只需各请求最后一个位置的隐藏向量。
            # 去掉起始边界 0，再将结束边界减 1：last_token = [2, 5] - 1 = [1, 4]。
            # 这一步的目的是为了获取每个请求的最后一个 token 的索引，即 [1, 4]。它包含了被预测的下一个 token 的信息。
            # 需要解码它获取下一个 token 的 ID。
            last_token = context.cu_seqlens_q[1:] - 1  # exclude the first element which is 0
            # 选取后 x = [[1, 2], [3, 1]]，形状从 [5, 2] 变为 [2, 2]；contiguous 保证连续存储。
            x = x[last_token].contiguous()
        # decode 已经是每条请求一个隐藏向量，例如同样的 [[1, 2], [3, 1]]，无需再选择位置。

        # logits: [batch_size, vocab_size_per_partition]，本例每个进程都输出 [2, 4]。
        # 这里的2表示有两个请求都预测出了下一个 token，4 表示每个 token 在当前进程上的 4 个词的得分。

        # F.linear automatically transpose the weight
        # 不传 bias 时等价于 x @ weight.T；每个隐藏向量与本地每个 token 的权重向量做点积。
        # rank 1 的 weight = [[40, 41], [50, 51], [60, 61], [70, 71]]（token 4～7）。
        # 请求 A 对 token 4 的得分：1*40 + 2*41 = 122；请求 B 的得分：3*40 + 1*41 = 161。
        # 因此 rank 1 的 logits = [[122, 152, 182, 212], [161, 201, 241, 281]]。
        # rank 0 的 logits = [[2, 32, 62, 92], [1, 41, 81, 121]]。
        # rank 2 的 logits = [[242, 272, 0, 0], [321, 361, 0, 0]]，最后两列对应补齐位置。
        # logits 是未经 softmax 的得分，不是概率；本类不负责采样。
        logits = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            # 每个进程都有自己的 logits，形状都是 [2, 4]：2 条请求，各自对本分片的 4 个词计算得分。
            # 此时三个进程分别持有 token 0～3、4～7、8～11 的列，rank 0 还没有其他进程的得分。
            # 条件表达式等价于：
            #   if self.tp_rank == 0:
            #       all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(3)]
            #   else:
            #       all_logits = None
            # 列表推导式执行 3 次，每次独立分配一个 [2, 4] 张量；_ 是未使用的循环变量。
            # logits.size() 决定接收张量的形状；device=logits.device 让它们位于接收进程 rank 0 的设备。
            # empty 只分配空间，初始内容未定义；这一行并未收集数据，稍后的 gather 才会填充这些张量。
            # rank 0 的 all_logits 是含 3 个张量的 Python 列表，不是一个 [3, 2, 4] 张量。
            # all_logits[0]、[1]、[2] 分别预留给 rank 0、1、2；三个接收张量都位于 rank 0 的设备。
            # rank 1、2 的 all_logits 则为 None：它们只发送自己的 logits，不分配接收列表。
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # dist.gather collects the logits from all GPUs to GPU 0
            # gather 的三个参数：
            #   logits：当前调用进程提供的本地张量；rank 0 也要提供自己的那一份。
            #   gather_list：目标进程的接收列表；非目标进程必须传 None。
            #   dst=0：指定 rank 0 为接收目标，不表示只收集第 0 列或只让 rank 0 调用。
            # 因此，三个进程都执行这一行：
            #   rank 0：gather(logits_rank0, gather_list=[buffer0, buffer1, buffer2], dst=0)
            #   rank 1：gather(logits_rank1, gather_list=None, dst=0)
            #   rank 2：gather(logits_rank2, gather_list=None, dst=0)
            # 完成后 rank 0 的接收列表按 rank 顺序填充：
            #   all_logits[0] = [[2, 32, 62, 92], [1, 41, 81, 121]]
            #   all_logits[1] = [[122, 152, 182, 212], [161, 201, 241, 281]]
            #   all_logits[2] = [[242, 272, 0, 0], [321, 361, 0, 0]]
            # gather 直接写入接收缓冲区，无需用返回值接收；各进程的本地 logits 仍为原来的 [2, 4]。
            # 此时尚未拼接：rank 0 持有 3 个分片张量，下一步 cat 才将它们组成 [2, 12]。
            # rank 1、2 的 all_logits 仍为 None，也不会获得完整词表得分。
            dist.gather(logits, gather_list=all_logits, dst=0)
            # concatenate
            if self.tp_rank == 0:
                # 沿词表维度拼接，形状 [2, 4] × 3 -> [2, 12]，恢复全局 token ID 的列顺序。
                # 请求 A：[2, 32, 62, 92, 122, 152, 182, 212, 242, 272, 0, 0]。
                # 请求 B：[1, 41, 81, 121, 161, 201, 241, 281, 321, 361, 0, 0]。
                logits = torch.cat(all_logits, dim=-1)
                # trim to original vocab size
                # 只保留真实 token 0～9 的列，形状 [2, 12] -> [2, 10]，避免采样到补齐的 ID 10、11。
                logits = logits[..., :self.num_embeddings]

        # 多进程时：rank 0 返回完整 [2, 10] logits，其他 rank 仍返回本地 [2, 4] logits。
        # 单进程时：无需 gather，直接返回完整 [batch_size, V] logits。
        return logits
