from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    # auto的作用是自动为每个枚举值分配一个唯一的整数，从 1 开始递增
    WAITING = auto() 
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count() # 全局自动递增的序列 id，用于唯一标识每个序列

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        self.block_size = block_size # number of tokens per block
        # record sequence id
        self.seq_id = next(Sequence.counter) # 从全局 counter 中获取下一个序列 id
        # status
        self.status = SequenceStatus.WAITING # 初始状态为等待
        # token ids, need copy so that it is a new list, won't be affected by outside changes
        self.token_ids = copy(token_ids) # 复制 token_ids 到 self.token_ids，避免外部修改 self.token_ids 导致序列状态异常
        # last token
        # Decode 阶段只需输入上一步生成的 token，历史 token 的 K/V 已保存在 KV Cache 中。
        # 张量并行时，各 rank 需要相同的输入；主进程可只传递最新 token，
        # 避免重复序列化和传输完整 token_ids，减少通信量及临时内存开销。
        # 初始化时保存 prompt 的末尾 token，生成新 token 后由 append_token() 更新。
        self.last_token = self.token_ids[-1] if self.token_ids else None
        # num_tokens, num_prompt_tokens
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        # num_cached_tokens = 0
        self.num_cached_tokens = 0 # 被缓存命中的 token 数量，一定是 block size 的整数倍
        # block_table
        # 当前序列的逻辑块到 KV Cache 物理块的映射：
        # 下标为逻辑块编号，元素为物理块 ID；分配缓存块时填充。
        self.block_table = []
        # sampling_params' related things
        self.temperature = sampling_params.temperature # 采样的温度参数，用于控制采样的随机性
        self.max_tokens = sampling_params.max_tokens  # 最大生成 token 数量，超过后停止采样
        self.ignore_eos = sampling_params.ignore_eos # 是否忽略 EOS token
        self.max_model_length = sampling_params.max_model_length # 最大输入给模型长度（prompt+新生成的），超过后截断

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self): # 已生成的 token 数量，即 prompt 后的 token 数量
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self): # 返回 prompt token ids 列表
        return self.token_ids[:self.num_prompt_tokens]
    
    @property
    def completion_token_ids(self): # 返回已生成的 token ids 列表
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self): # 返回被缓存命中的逻辑块数量
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self): # 返回总逻辑块数量
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self): # 返回最后一个逻辑块的 token 数量
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1: # 最后一个逻辑块
            return self.token_ids[-self.last_block_num_tokens:] # 返回最后一个逻辑块的 token ids 列表
        else:
            start_idx = i * self.block_size # 计算当前逻辑块的起始 token id 索引
            end_idx = start_idx + self.block_size # 计算当前逻辑块的结束 token id 索引
            return self.token_ids[start_idx : end_idx] # 返回当前逻辑块的 token ids 列表

    def append_token(self, token_id): # 追加一个 token 到序列末尾
        self.token_ids.append(token_id)
        self.last_token = token_id  # 更新 last_token 为最新生成的 token
        self.num_tokens += 1 

    def __getstate__(self):  # pickle 序列化时自动调用，提取需要传给 worker 的序列状态
        return (
            self.num_tokens, # 总 token 数量
            self.num_prompt_tokens, # prompt token 数量
            self.num_cached_tokens, # 命中的前缀缓存 token 数量
            self.block_table, # 当前序列的逻辑块到 KV Cache 物理块的映射
            # 尚未生成 token 时返回完整 token_ids，否则只返回最新 token
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state):
        # pickle 反序列化时自动调用；worker 据此恢复主进程传来的本轮推理状态
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            last_token_or_ids
        ) = state  # __getstate__() 返回并经 pickle 传来的状态元组
        # 按是否已有生成 token 还原数据格式；此条件并非直接判断 is_prefill
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens  # 已生成的 token 数量
        if num_completion_tokens == 0:
            # 尚未生成 token：恢复完整 token_ids，供首次 prefill 使用
            self.token_ids = last_token_or_ids
        else:
            # 已有生成 token：只恢复最新 token 的列表，供正常 decode 使用
            # worker 不持有完整 token 历史，num_tokens 仍记录完整序列长度
            self.token_ids = [last_token_or_ids]
        # 根据恢复后的 token_ids 设置 last_token；空列表时为 None
        self.last_token = self.token_ids[-1] if self.token_ids else None
