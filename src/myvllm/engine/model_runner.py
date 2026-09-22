import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *

class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config # 全局配置文件涉及到模型配置、kv 缓存配置
        self.event = event # 子进程只有一个通知事件，主进程是一个通知列表

        # set distributed config
        self.block_size = config['block_size'] # 每个缓存块的大小
        self.world_size = config['world_size'] # 全局有多少个并行
        # 是否需要立即执行，设置为true表示需要立即执行，那么在 decode阶段，不会使用 cuda graph 重放，cuda graph 是每一次计算的计算图，可以存下来复用
        # 如何设置为true表示不进行图复用直接开始执行，设置为false表示复用计算图，设置为false意味着在初始化阶段可能需要花点时间、同时还需要维护图和固定缓冲区
        # 但是在长序列生成的时候后续的加速会抵消掉这部分耗时。
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank # 可以理解为这是全局的第几个进程
        # 初始化分布式通信组，凑够 world_size 以后才进行初始化，也就是说在整个通信组里面每个都是平等的
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank) # 使用对应的 gpu

        # set model
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    # scale 是用来调整注意力权重的，scale为1表示标准的注意力缩放，scale 大于 1 表示注意力权重集中在高分部分，scale 越小表示注意力越平均
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    # rmsnorm 归一化时使用的参数
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    # 用来调整位置编码的频率
                    base=config['base'],
                    # 模型运行的可接受的最大序列长度，这个和位置编码有关系，因此扩展这个的长度需要 base 的配合来保证长序列的效果
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    # 表示语言头是否和Embedding 权重实现共享，节约显存
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank) # 将模型运行在对应的 gpu 上

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            # 这里需要深入理解模型原始的架构以及当前推理框架的模型架构的体现形式
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)
        
        # 构建一个采样器，根据输出的 logits 和温度为每条序列采样出下一个 token
        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype() # 保存当前全局默认的数据类型

        # Debug flag for first decode step
        self._first_decode = False # 预留的首次 decode 调试标记，目前没有使用

        # warm up model so that we know peak memory usage
        self.warmup_model() # 记录在预填充阶段的最大显存占用，方便用于后续的的 kv 缓存块的分配
        # allocate kv cache
        """
        KV Cache 显存预算 = warmup 清理后的空闲显存 × 利用率 - (峰值占用 - 当前占用)
        峰值与当前占用之差用于估计推理临时显存；空闲显存已排除模型等现有占用
        
        GPU 总显存：      24 GB
        当前占用：        6 GB（包含模型等）
        当前空闲：        18 GB
        试跑峰值：        10 GB
        显存利用率：      0.9

        临时显存估计 = 10 − 6 = 4 GB
        KV Cache 预算 = 18 × 0.9 − 4 = 12.2 GB
        """
        self.allocate_kv_cache()  # 分配kv cache 缓存空间
        # capture cuda graph for decoding
        if not self.enforce_eager:  # 是否直接执行，如果设置为true 表示直接执行，如果设置为 false 表示不直接执行，这个时候需要获取之前的计算图
            self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}') # 设置当前进程的运行显卡设备
        torch.set_default_dtype(self.default_dtype) # 设置当前运行的默认数据类型

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        # 这里共享内存的作用是，用于在同一通信组内进行指令传递
        if self.world_size > 1:  # 对于多进程或者说对于多卡并行来说
            # Synchronize before setting up shared memory
            dist.barrier() # 等待所有进行都到达这个点，等待前面工作完成
            if self.rank == 0:  # 对于主进程
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='myvllm') # 连接之前的共享内存
                    old_shm.close() # 关闭当前主进程对共享内存的访问
                    old_shm.unlink() # 删除这个共享对象，其他的已连接的这个对象不受影响，他们只根据名字来
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20) # 创建一个同名的大小为1MiB
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier() # 主进程先完成必须在这里等待，等待说有子进程都到达
            else: 
                # Wait for rank 0 to create shared memory
                dist.barrier() # 子进程会先到达这里，同时等待主进程，这个和 140 行的等待属于同一个等待
                self.shm = SharedMemory(name='myvllm') # 子进程连接到新的共享内存中去
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        # 确保只有工作进程或者子进程才能读
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        # 等待主进程通知，即主进程设置 event.set()，否则执行到这儿就等待
        self.event.wait()
        # self.shm.buf[:4] 读取缓存中的前四个字节，这个表示消息的长度，little 表示小端序，将权重低的放在前面，高的放在后面
        # 比如300这个数字，一个字节最多表示256，,300需要分成两个字节 300 = 256^0 * 44 + 256^1 * 1, 44 的权重为 0 放在前面
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        # 先读取字节数据 4：n+4 然后再进行序列化
        """
        主进程：
        ("run", seqs, True)
            ↓ pickle.dumps()
        字节数据
            ↓ 写入共享内存
        ────────────────────────
        工作进程：
        读取字节数据
            ↓ pickle.loads()
        ("run", seqs, True)
        """
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear() # 重置 event 状态，当主进程有新消息的时候再设置，如果不重置 event.wait() 没有作用，下次还是读取旧的信息，直到更新
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        # 只能主进程写入缓存中去
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        # 序列化数据
        data = pickle.dumps((method_name, *args))
        # 计算数据长度
        n = len(data)
        # 前4个字节存储长度
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        # 后面的 n 个字节存储数据
        self.shm.buf[4:n+4] = data
        # 写入以后通知子进程拿数据
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close() # 在退出的时候主、子进程断开对共享缓存的连接
            if self.rank == 0:
                self.shm.unlink() # 如果是主进程会删除这个共享块
        # enforce_eager=False 时，初始化阶段才会捕获 CUDA Graph 并创建下面两个属性。
        # enforce_eager=True 使用普通执行模式，没有这些图资源，因此跳过清理。
        if not self.enforce_eager:
            # graphs 是 {batch_size: CUDAGraph} 字典，保存不同批大小对应的 GPU 操作图。
            # 重放图仍会重新计算；这里删除属性、解除引用，让不再被引用的图对象得到回收。
            del self.graphs
            # graph_vars 保存图使用的固定缓冲区：input_ids、slot_mapping、context_lens、
            # block_tables 和 outputs。重放依赖固定内存地址，每轮只更新缓冲区内容。
            # 退出后不再重放，解除这些张量的引用；其他引用和显存分配器会影响实际回收时机，
            # 并不保证显存立即归还驱动。这两行也没有直接删除模型权重或各层持有的 KV Cache。
            del self.graph_vars
        # CUDA 操作通常异步提交：Python 已执行到这里，GPU 可能仍有未完成的工作。
        # 等待当前设备（前面通过 set_device(rank) 选定）所有流上的任务完成，再清理通信组。
        # 这是本进程等待 GPU 完成工作，不是 dist.barrier() 那样让所有 rank 到达集合点，
        # 也不会清空显存；按当前顺序，等待发生在上面解除图及缓冲区引用之后。
        torch.cuda.synchronize()
        # 检查默认进程组是否已初始化且尚未销毁，避免对不存在的通信组执行清理。
        if dist.is_initialized():
            # 每个 rank 都清理自己的分布式通信资源（本项目使用 NCCL）。
            # 销毁通信组不会终止 Python 进程；工作进程还需跳出 loop() 并返回入口函数。
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        # 子进程一直循环等待通知，直到主进程说退出
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm() # 这里面会等待主进程通知
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        # 如果是主进程应该将自己的要做的事情写入缓存通知其他子进程一起做
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None) # 主进程和子进程都要做这件事情
        if method:
            return method(*args) # 执行对应的方法
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache() # 清理未使用的显存缓存
        torch.cuda.reset_peak_memory_stats() # 重新记录显存峰值
        max_tokens = self.config['max_num_batch_tokens'] # 获取每次推理的最大token 数量
        max_model_length = self.config['max_model_length'] # 获取模型最大的输入token 数量
        batch_size = max_tokens // max_model_length # 计算在满配情况下的batch size 大小
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)] # 模拟构建满配的输入序列
        # 准备输入 → 模型前向 → 语言头计算 logits → rank 0 采样
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache() # 释放试跑以后的显存缓存

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info() # 查询当前空闲显存，单位为字节以及 gpu 总的显存单位也为字节
        total_free_mem = free_mem * self.config['gpu_memory_utilization'] # 从空闲显存（已经排除了模型的占用）取出一定比例作为初步预算
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak'] # 先前统计中获取到的显存峰值
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current'] # 当前已经被分配显存（模型占用等）
        # reserve some room for peak memory usage during model execution
        # KV Cache 预算 = 18 × 0.9 − 4 = 12.2 GB
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage) # 总的可用的减去在满配情况下推理的显存占用
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers'] # 当前模型的层数，因为每一层都需要保存 kv cache
        num_kv_heads = self.config['num_kv_heads'] // self.world_size # 根据当前的显卡数量均分注意力机制的头数
        # 计算出每一个头的维度，方便计算每个头的占用的显存大小
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block
        # 每个缓存块的大小 * 每个token 对应一个 k 和 v * 整个模型结构的层数 * 每个头的维度 * 每个数量的字节占有量
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        # 计算可以分配多少个显存块
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            # 把当前 gpu 中可以使用的缓存块数变成一个0 维张量
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            # 聚合一个通信组里面的所有 per_rank_blocks_tensor ，目的是为了保持下标的一致性在全局，因此每个 token 在每一层都有对应的存储，因此位置也应该是一样的
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            # 记录当前最大可分配的缓存块数量
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        # 在真正的 gpu 上建立缓存 [K 或 V, 层编号, 物理块编号, 块内 token 位置, KV 头编号, 头内维度]
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')
        # 将对应的缓存空间分配到对应的层中去
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id] 
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache
        input_ids = [] # 当前推理批次中未被缓存部分的token ids
        # length: sum of all input_ids after prefix cache
        slot_mappings = [] # 每个输入的新 token 的 kv 应该写入到那个物理槽位，这里是一个映射关系
        # length: num_seqs
        seqlens_q = [] # 本轮输入的长度，即排除已经被缓存命中的 token 长度
        # length: num_seqs
        seqlens_k = [] # 每条序列完整的长度，包含了已经被缓存命中的 token 数量
        # length: num_seqs + 1
        cu_seqlens_q = [0] # query 长度的前缀和，有N条序列既有 N+1 个元素
        # length: num_seqs + 1
        cu_seqlens_k = [0] # key的前缀和
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = [] # 各序列逻辑块到物理块的映射
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                # 跳过已经命中的缓存块，他们的kv已经存在
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    # 判断是不是最后一个逻辑块，如果不是最后一个块，且没有被缓存中，那说明是要生成一个满块，用来存储 token 的 kv
                    """
                    物理块 2 → slot 8, 9, 10, 11
                    物理块 5 → slot 20, 21, 22, 23
                    物理块 1 → slot 4, 5, 6, 7
                    """
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids


    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids    

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay the graph
            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        reset_context()
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # output logits
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )