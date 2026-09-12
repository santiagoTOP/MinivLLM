from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # 管理 KV Cache 物理块：共 max_cached_blocks 块，每块容纳 block_size 个 token 的 K/V
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        # 每轮调度的输入 token 数量预算，不是生成 token 的总上限：
        # prefill 按 len(seq) 计数；decode 每条序列只输入最新 token，计 1 个
        self.max_num_batched_tokens = max_num_batched_tokens
        # 每轮调度选入 batch 的序列数量上限
        self.max_num_sequences = max_num_sequences
        # 等待首次 prefill，或被抢占后等待重新 prefill 的序列
        self.waiting: deque[Sequence] = deque()
        # 已被接纳且尚未结束的序列，包括本轮刚选入 prefill 的序列
        self.running: deque[Sequence] = deque()
        self.eos = eos

    # 用来判断当前队列中是否还有没有完成的任务
    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.
        capacity = len(self.block_manager.blocks) # 获取当前队列管理的缓存块
        if sequence.num_blocks > capacity: # 判断当前加入的序列所需要的缓存块是否能被满足
            raise ValueError(
                f"Sequence {sequence.seq_id} needs {sequence.num_blocks} blocks "
                f"({len(sequence)} tokens at block_size={self.block_manager.block_size}) "
                f"but the KV cache only holds {capacity}. "
                f"Raise max_cached_blocks or block_size, or shorten the prompt."
            )
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        # 这里的调度指的是当一次推理结束以后下一次放到gpu 上去推理的序列，这里主要从两个方面去选择：优先等待序列随后才是上一轮在 running 中的序列
        scheduled_sequences = [] # 当前推理轮次准备推理的序列
        current_scheduled_tokens = 0 # 当前被调度的 token 数，不要超过 max_num_batched_tokens
        # An empty schedule is only legitimate when this call freed blocks by
        # preempting, so the next call can make progress. See the guard below.
        preempted = False # 是否发生过抢占
        # try schedule for prefilling from waiting queue if not exceeding limits
        # 优先从等待的队列中选择序列进行调度
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0] # 准备调度等待序列中的第一
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq) # 分配缓存块
                seq.status = SequenceStatus.RUNNING # 改变序列的状态
                self.running.append(seq) # 加入runing队列
                scheduled_sequences.append(seq) # 加入被调度的列表中
                current_scheduled_tokens += len(seq) # 累加tokens
            else:
                break
        if scheduled_sequences: # 存在被调度的序列
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        # 如果等待序列中没有，就从正在运行中的队列中调度
        while self.running:
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            if not self.block_manager.can_append(seq): # 判断当前正在 running 中的序列能否继续后续的推理，确保每个正在运行的序列都能正常分配缓存空间
                preempted = True # 如果不能就发生抢占
                if self.running: # 如果正在运行的队列中还有其他队列
                    self.running.appendleft(seq) # 放回到原来的位置，即最开始的位置
                    self.preempt(self.running.pop()) # 抢占最新来的序列
                else:
                    self.preempt(seq) # 否则抢占自己
                    break
            else:
                # 从运行的队列中选择当前轮次的推理序列，直到超过设定的限制，不能继续选择
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                # append one token
                self.block_manager.append(seq) # 将上一轮生成的新token加入到缓存管理中，这个时候他还没有具体进行kv计算，需要在下一轮推理的时候才能进行计算
                scheduled_sequences.append(seq) # 加入当前推理序列中
                # 这里的+1 本质上是将上一轮推理得到的新 token 个数累加
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences)) # 恢复取之前的顺序
        elif not preempted and (self.waiting or self.running):
            # 走到这个分支就表示当前没有发生调度，没有抢占，且还存在没有结束的序列，说明发生了错误
            # 除了 KV Cache 容量不足或缓存块泄漏，单条序列的 prefill 长度超过 max_num_batched_tokens 等限制也可能触发这个分支
            # Nothing was scheduled and nothing was preempted, so no engine state
            # changed: every later schedule() would take the same decisions and
            # LLMEngine.generate() would spin forever. Fail loudly instead.
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} of "
                f"{len(self.block_manager.blocks)} blocks free. "
                "This means either a sequence that cannot fit in the KV cache, or "
                "blocks leaked because their ref_count never returned to 0."
            )

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        # 抢占的逻辑
        self.block_manager.deallocate(seq) # 释放自己占用的缓存空间
        seq.status = SequenceStatus.WAITING # 将序列状态转化为等待
        self.waiting.appendleft(seq) # 加入等待队列       


    # 处理本轮生成结果，检查停止条件，并回收已完成序列持有的缓存块引用
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        # 每条序列完成本轮推理后产生一个新 token，将其加入对应序列
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # 先追加新 token，后续长度判断包含本轮生成的 token
            # 未忽略 EOS，且新生成的 token 是 EOS，则结束生成
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            # 已生成的 token 数达到最大生成数量
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            # 设置了总长度上限，且 prompt + 已生成 token 的总数已达到上限
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED  # 任一停止条件满足，标记序列完成
                self.block_manager.deallocate(seq)  # 归还缓存块引用，引用归零的块可复用
                self.running.remove(seq)  # 移出运行队列，不再参与后续调度
