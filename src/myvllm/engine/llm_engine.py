import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering，方便及时打印日志
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)  # Line buffering，方便及时打印错误日志

    model_runner = ModelRunner(config, rank, event) # 初始化子进程的模型推理器
    model_runner.loop() # 让子进行的worker进入循环等待主进程的指令


class LLMEngine:
    def __init__(self, config: dict):
        self.config = config  # 模型配置文件
        world_size = config.get("world_size", 1) # 全局进程数，主要是用于分布式推理
        ctx = mp.get_context("spawn") # 获取 spawn 方式的多进程上下文；用它创建的子进程是全新进程，不继承父进程状态（CUDA 安全）
        self.processes = []  # 子进程列表，方便在退出时 `join()`，避免僵尸进程
        self.events = []  # 事件列表，用于主进程同步子进程状态，避免轮询，省 CPU 资源
        for i in range(1, world_size):
            # event.wait 表示正在等待主进程通知数据已经更新
            # event.set 表示主进程通知每个子进程，开始加载数据
            # event.clear 表示清除通知标志，等待一下次通知
            event = ctx.Event() # 主进程创建事件，用于同步子进程状态，虽然使用的是共享内存，但是 event 的使用是为了方便主进程通知数据已经更新
            process = ctx.Process(target=worker_process, args=(config, i, event)) # 创建子进程对象
            self.events.append(event) 
            self.processes.append(process) 
            process.start() # 启动子进程
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events) # 初始化主进程的模型推理器，这里的 event 是一个事件列表，用于主进程同步子进程状态，避免轮询，省 CPU 资源
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        # 负责决定每一步前向传播时，哪些序列、以什么方式组 batch，continuous batching的核心逻辑
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16), # 最大序列数，每个 batch 最多 16 个序列
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024), # 控制每次前向传播的 token 数量
            max_cached_blocks=config.get("max_cached_blocks", 1024), # 最大缓存 block 数量
            block_size=config.get("block_size", 256), # 每个 block 的 token 数量
            eos=config.get("eos", 50256) # 推理的结束符 token id
        )

        atexit.register(self.exit) # 注册正常退出时的回调函数，确保所有子进程也退出


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join() # 等待所有子进程退出，避免僵尸进程

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule() # 调度调度器，返回待处理的序列和是否为预填充
        num_processed_tokens = 0 # 记录被处理的 token 数量，如果是预填充，就是序列长度，否则就是 batch 大小
        if not scheduled_sequences: # 如果没有待处理的序列，直接返回空列表
            return [], num_processed_tokens, is_prefill
        # run the model
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill) # 调用模型推理器，返回模型输出
        # Move outputs to CPU and convert them to a list
        if outputs is not None:
            outputs = outputs.cpu().tolist() # 将模型输出从 GPU 移动到 CPU 并转换为列表，tensor 转 list of list of int32
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs) # 后处理模型输出，更新序列状态和缓存 block

        # 当一个序列推理完成时，返回序列的序列 id 和推理生成的 token ids 列表
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        # 加入等待队列，等待被调度
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'], sampling_params=sampling_params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            # 调用模型推理器，返回模型输出
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            # 输出一下推理的速度，单位是 token/sec
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            # 记录序列 id 和推理生成的 token ids 列表
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})
        # 按照序列 id 排序，确保与输入的 prompt 顺序一致
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        # 转换为文本
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
