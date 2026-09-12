import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id # 当前块的id号
        self.hash = -1 # 当前物理块的哈希值
        self.ref_count = 0 # 当前块被引用的次数
        self.token_ids = [] # 当前块中存储的token id ，一个块最多256个


    def update(self, h: int, token_ids: list[int]):
        self.hash = h  # 更新物理块的哈希值
        self.token_ids = token_ids # 当前物理块中存储的ids

    def reset(self):
        self.hash = -1  # 重置这个物理块的哈希值
        # reset() is only reached via BlockManager._allocate_block, which takes the
        # block off the free list on behalf of exactly one sequence. Allocation is
        # therefore the first reference: leaving this at 0 makes the matching
        # deallocate() drive ref_count to -1, so the block is never freed.
        self.ref_count = 1 # 因为这里的重置是空闲块被分配给一个请求使用的，因此被分配以后必定存在一个引用
        self.token_ids = [] # 将这个物理块的 ids 清零

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        # block_size: number of tokens per block
        self.block_size: int = block_size # 每个块的大小
        # list of all blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 对每个块进行编号，一共有1024个块
        # hash to block id: this is for prefix caching
        self.hash_to_block_id: dict[int, int] = {} # 块的哈希值 → 物理块 ID
        # free block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 当前可以自由使用的物理块个数
        # used block ids
        self.used_block_ids: set[int] = set() # 当前已经使用的物理块

    # given token_ids, compute the hash value
    # use prefix_hash_value to compute the hash in a context-sensitive way
    # 哈希的计算逻辑，计算哈希的目的是为了快速找到对应的缓存块，计算每个缓存块的哈希时候需要考虑这个缓存块的前缀信息
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # move this block to used list
    def _allocate_block(self, block_id: int) -> Block: # 把指定的物理块分配出去
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None: # 回收指定的物理块
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        # Clearing token_ids deliberately keeps the prefix cache scoped to blocks that
        # are still referenced: a freed block can no longer match in allocate(), so a
        # cache hit never spans a finished sequence. Reuse across sequences is not
        # enabled yet because the prefill path cannot consume it -- the Triton kernel
        # in layers/attention.py attends only over the K/V computed in that pass and
        # ignores context.block_tables, and qwen3 derives RoPE positions from
        # cu_seqlens_q, so both would be wrong by num_cached_tokens. Enabling reuse
        # means a paged prefill kernel (cu_seqlens_q != cu_seqlens_k) plus a position
        # offset; until then this line is what keeps the engine correct.
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # whether we can allocate a block for this sequence
    def can_allocate(self, seq: Sequence) -> bool: # 判断是否能够给指定的序列分配物理块
        return len(self.free_block_ids) >= seq.num_blocks # 必定是要空闲的块大于等于需求的块


    def allocate(self, seq: Sequence) -> None:
        h = -1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i) # 在指定逻辑块上需要被缓存到物理块上的 ids
            # only compute hash for full blocks, always -1 for partial blocks
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1) # 判断是否存在被缓存命中的物理块id
            
            # if cache miss or hash collision
            # 防止哈希碰撞以及哈希映射已经过时了
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: # 命中且ids一致，否则被认定为没有命中缓存
                no_cache_found = True # 这里设置为true 表示没有被命中

            if not no_cache_found: # 缓存被命中
                # update sequence information
                seq.num_cached_tokens += self.block_size # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                if block_id not in self.used_block_ids: 
                    # 进入当前分支的前提是采用了：保留空闲缓存并复用的策略，即一个物理块里面存了东西，当时没有在被使用的列表中，同时前缀哈希也还存在，只是引用为 0
                    # Unreachable while _deallocate_block clears token_ids: a freed
                    # block has token_ids == [] and fails the match above. Kept for
                    # when cross-sequence reuse is enabled.
                    block = self._allocate_block(block_id) # 调用这个的前提是这个物理块的引用为0
                else:
                    # update block information
                    block = self.blocks[self.hash_to_block_id[h]] # 获取这个物理块
                    block.ref_count += 1 # 并将引用加一
            else:
                # cache miss
                block = self._allocate_block(self.free_block_ids[0]) # 没有命中，就新分配一个物理块使用
                block.update(h=h, token_ids=token_ids)
                if h != -1: # -1 表示当前物理块还没有被占满，因此无法计算它的哈希值
                    self.hash_to_block_id[h] = block.block_id
            seq.block_table.append(block.block_id) # 在这条序列的块表中加入这个块 id，直到当前序列使用了哪些缓存块
        
    def deallocate(self, seq: Sequence) -> None:  
        # 当一条序列推理结束以后或者缓存不足的时候需要调用这个函数
        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1 # 释放对这个缓存块的引用
            if block.ref_count == 0: # 当物理块的引用为 0 的时候回收这个物理块
                self._deallocate_block(block_id)
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0

    # this is to check whether we can append tokens to this sequence
    # when that token would require allocating a new block.
    def can_append(self, seq: Sequence) -> bool: 
        # 判断当前的缓存够不够在接下来的推理中为刚刚生成的 token 存储它的 kv cache
        # 生成的新 token 已经计入 seq.num_tokens，但还没有为它写入 K/V
        # Called after the new token is already counted in seq.num_tokens, so the
        # condition must match append()'s allocation branch: a fresh block is only
        # needed when that token is the first of a new block (num_tokens % size == 1).
        # At num_tokens % size == 0 the token still fits in the block the sequence
        # already holds, and append() merely finalizes its hash.
        if seq.num_tokens % self.block_size == 1: # 为0 表示当前物理块刚好够存储，为 1 表示需要新加一个物理块，其他表示当前有物理块没有满
            return len(self.free_block_ids) > 0 # 如果为 1，且还存在空余的物理块表示可以新增加一个物理块
        return True

    # 最新 token 已追加到序列并计入 num_tokens，但其 K/V 尚未经过本轮前向计算。
    # 此处准备缓存块并维护元数据，实际 K/V 写入由后续模型前向完成。
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table  # 与序列的块表引用同一个列表对象
        last_block_for_seq_id = block_tables[-1]  # 当前已分配的末尾物理块 ID

        # 最新 token 恰好填满当前逻辑块：无需新块，为完整块建立哈希索引。
        if seq.num_tokens % self.block_size == 0:
            # 结合前块哈希和本块 token IDs 计算；第一块没有前块，使用 -1。
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))  # 更新哈希和 token ID 元数据
            self.hash_to_block_id[h] = block.block_id  # 建立或更新哈希到物理块 ID 的映射
        # 最新 token 是下一个逻辑块的第一个 token：需要分配新物理块。
        elif seq.num_tokens % self.block_size == 1:
            # 此时已分配的末尾块是前一块，应已填满并建立有效哈希。
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])  # 分配空闲物理块
            block_tables.append(block.block_id)  # 原地修改共享列表，seq.block_table 同步更新
        # 最新 token 仍位于已有的未满块中，无需分配新块或计算完整块哈希。
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
