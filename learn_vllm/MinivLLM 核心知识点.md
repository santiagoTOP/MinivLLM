# MinivLLM 核心知识点

> 结合项目源码持续整理 MinivLLM 的核心概念、执行流程与实现细节。
> 当前涵盖：请求调度、多进程驱动与通信、token 生成流程、KV Cache 与前缀缓存复用；后续继续补充其他知识点。

---

## 1. `world_size`：全局进程数

```python
world_size = config.get("world_size", 1) # 全局进程数，主要是用于分布式推理
```

- 从配置读取总进程数，默认 `1`（单进程推理）。
- 循环 `range(1, world_size)` 为 rank 1..N-1 各启动一个 worker 进程；主进程本身作为 rank 0。
- `world_size=4` ⇒ 3 个 worker + 主进程，共 4 个进程协同，各进程内部通过 `dist.init_process_group()` 建立通信。

---

## 2. `mp.get_context("spawn")`：获取多进程上下文

```python
ctx = mp.get_context("spawn")
```

- `ctx` 是**多进程上下文对象（工厂）**，不是进程本身；真正的进程由 `ctx.Process(...)` 创建。
- 这里的 `mp` 是 `torch.multiprocessing`（标准库 `multiprocessing` 的封装，兼容 CUDA 张量传递）。

### 为什么用 `spawn` 而不用默认的 `fork`？

| | `fork` | `spawn` |
|---|---|---|
| 创建方式 | 复制父进程完整内存 | 启动全新 Python 解释器，从零开始 |
| 子进程初始状态 | 父进程"快照"（带 CUDA 上下文） | 干净，只继承 pickle 传入的参数 |
| CUDA 安全性 | ❌ fork 出的子进程不能重新初始化 CUDA | ✅ 推荐，PyTorch 分布式标准做法 |

- 用 `get_context()` 而非全局 `set_start_method()`，可避免污染全局状态，只影响本处创建的进程。
- 子进程只知道自己要运行 `worker_process(config, rank, event)`，其余一切从零加载。

---

## 3. `self.processes` / `self.events`：句柄列表

```python
self.processes = []  # 进程列表
self.events = []     # 事件列表
```

两个列表用途完全不同，缺一不可：

| 列表 | 创建处 | 消费处 | 缺失后果 |
|---|---|---|---|
| `self.processes` | `__init__` 循环中 append | `exit()` 中 `process.join()` | **僵尸进程**（无法回收） |
| `self.events` | `__init__` 循环中 append | rank 0 `write_shm` 中遍历 `event.set()` | **死锁**（指令发不出，worker 永久阻塞在 `wait()`） |

- **进程列表 → 给"退出"用**：不保存句柄就无法 `join()` 等待回收，程序退出时子进程残留为僵尸进程。
- **事件列表 → 给"广播"用**：不保存事件，rank 0 就没有办法挨个通知所有 worker，worker 会一直阻塞在 `event.wait()`。

两者都是把循环中分散创建的对象集中管理起来：一个管"怎么回收进程"，一个管"怎么给进程发信号"。

---

## 4. `ctx.Process()`：创建子进程对象

```python
event = ctx.Event()
process = ctx.Process(target=worker_process, args=(config, i, event))
```

**注意：这行只是创建进程对象，并没有真正启动进程**（OS 里还没有这个子进程），真正启动是下一行的 `process.start()`。

| 部分 | 含义 |
|---|---|
| `ctx.Process(...)` | 创建一个 Process 对象（"注册"子进程） |
| `target=worker_process` | 子进程入口函数：启动后执行 `worker_process(config, rank, event)` |
| `args=(config, i, event)` | 传给入口函数的参数：`config`（配置）、`i`（rank）、`event`（同步事件） |

完整生命周期：

```python
process = ctx.Process(...)           # 创建对象（未启动）
self.processes.append(process)       # 保存句柄
process.start()                      # 真正启动子进程，执行 worker_process(config, i, event)
```

> 细节：`spawn` 子进程不继承父进程内存，所以 `config`、`event` 等参数**通过 pickle 序列化**传到子进程，子进程反序列化后再执行 `worker_process`。

### `ctx.Event()`：跨进程同步事件（"信号灯"）

- `Event` 是进程间共享的布尔标志：`set()` 置真、`clear()` 复位、`wait()` 阻塞直到为真。
- 作为参数传给子进程后，**父子进程看到的是同一个事件**，一方 `set()` 另一方立即感知。

### 在 worker 侧的用法（`model_runner.py`）

```python
# read_shm（worker 侧）：等信号 → 读数据 → 复位
self.event.wait()                       # 阻塞直到 rank 0 发出通知
n = int.from_bytes(self.shm.buf[:4], 'little')
method_name, *args = pickle.loads(self.shm.buf[4:n+4])
self.event.clear()                      # 复位，等待下一条指令

# write_shm（rank 0 侧）：写数据 → 广播通知
data = pickle.dumps((method_name, *args))
self.shm.buf[:4] = n.to_bytes(4, 'little')
self.shm.buf[4:n+4] = data
for event in self.event:
    event.set()                         # 通知所有 worker：有新指令
```

---

## 5. 为什么用"共享内存 + Event"这套设计？

张量并行下每个 rank 只持有模型分片，前向依赖 `all_reduce` 等**集合通信**，要求所有 rank **以相同顺序、在相同数据上**同步执行。但调度器只存在于 rank 0，worker 不知道何时跑、跑哪个 batch ⇒ 需要 **单驱动 + 多执行者（driver/worker）模式**（vLLM 等框架同理）。

| 组件 | 职责 |
|---|---|
| **共享内存** | 传**数据**：rank 0 直接写入 pickle 后的指令，worker 直接读，零拷贝、低延迟（1MB 固定块足够） |
| **Event** | 传**信号**：共享内存无法通知"有新数据"，Event 让 worker `wait()` 阻塞休眠（省 CPU），被 `set()` 唤醒才去读，读完 `clear()` |

> 一句话：**共享内存负责传数据（快），Event 负责传信号（准且省 CPU）**，两者配合实现低开销的主从同步循环。

---

## 6. 指令分发完整链路（"告诉所有 worker"）

以一次调度 `run` 为例，指令从 rank 0 流向所有 worker：

```
① llm_engine.step() 调度出 batch
   ↓  self.model_runner.call("run", scheduled_sequences, is_prefill)

② call()（rank 0）：先 write_shm(method_name, args) 把指令写入共享内存
   ↓  再继续执行本地 method(*args)

③ write_shm()：pickle 打包 (method_name, *args) → 写入共享内存 → set() 所有 worker 的 Event

④ worker loop()：read_shm() 阻塞在 event.wait() 被唤醒 → 读出指令 → clear()
   ↓  self.call(method_name, *args) 执行同一个 "run"

⑤ 所有 rank 进入同一次前向传播，NCCL 集合通信得以对齐
```

关键代码位置：

```python
# llm_engine.py
outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)   # ① 发送端

# model_runner.py
def call(self, method_name, *args):                                        # ②③
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, args)   # 写共享内存 + set 事件
    method = getattr(self, method_name, None)
    return method(*args)                    # 本地也执行（rank 0）

def loop(self):                                                            # ④ worker 侧
    while True:
        method_name, args = self.read_shm() # 等事件、读指令
        self.call(method_name, *args)
        if method_name == 'exit':
            self.exit()
            break
```

---

## 7. worker 进程的入口：`worker_process` 函数详解

```python
def worker_process(config, rank, event):
    # ① 改行缓冲，让 print 实时可见
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
    # ② 初始化模型推理器
    model_runner = ModelRunner(config, rank, event)
    # ③ 进入指令接收死循环
    model_runner.loop()
```

### ① 行缓冲：为什么必须改 stdout（L18-19）

```python
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
```

拆解：`fileno()` 取底层文件描述符 → `os.fdopen(fd, 'w', buffering=1)` 用同一 fd 包装新文件对象，`buffering=1` 即**行缓冲** → 替换全局 `sys.stdout`。

Python 缓冲策略取决于输出目标：

| 输出目标 | 默认缓冲 | 效果 |
|---|---|---|
| 终端（TTY） | 行缓冲 | 一行一行实时显示 |
| **管道/文件（非 TTY）** | **块缓冲（4KB/8KB）** | 攒够一大块才输出，或等进程结束 |

多进程场景下 stdout 通常不再直连终端（被重定向/多进程共用），落入块缓冲——worker 的 print 会**静默积压**，直到缓冲满或进程退出。而 worker 的两个特点放大了危害：

1. **初始化阶段有汇合屏障**（`init_process_group` 卡很久）——块缓冲下完全看不到走到了哪一步，卡住了也不知道卡在哪。
2. **之后进入 `loop()` 死循环等待**——运行期日志直到进程退出才出现，实时调试无从谈起。

> 细节：stderr 按惯例默认就不缓冲，那行属于"顺手统一保险"，真正的价值在 stdout 那行。

### ② + ③ worker 的完整一生

```
worker_process 启动
  ├─ ModelRunner(...)   → 初始化：init_process_group（汇合屏障）→ 加载模型 → warmup → KV cache
  └─ loop()             → 死循环：wait → read_shm → call → wait → ...
                           └─ 直到收到 "exit" 指令，break，进程结束
```

- `while True` 意味着 worker 永不主动退出，rank 0 发什么它执行什么（`run`、`exit` 等任何方法名），彻底没有自己的意志——"被动执行者"的全部体现。
- **隐藏的坑**：如果 rank 0 在写共享内存前就崩溃，`read_shm` 里的 `event.wait()` 会永远阻塞，worker 变成孤儿进程——这也是主进程用 `atexit` 兜底的原因。
- **角色分岔点**：同一段 `ModelRunner` 代码，worker 进 `loop()` 当执行者，rank 0 不进 `loop()`、把控制权交还给 `LLMEngine` 继续创建 scheduler/tokenizer，之后由 `step()` 主动发起调用。

---

## 8. 初始化汇合屏障：`init_process_group` 的内部机制

```python
# model_runner.py L26（所有 rank 都会执行这一行）
dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
```

**项目里没有任何一行显式写着"等待 rank 0"**——这个等待发生在 `init_process_group` **内部（PyTorch 库里）**。

### 内部原理（TCPStore + 屏障）

1. **建立会合点**：`init_method="tcp://localhost:12345"` 让所有进程通过该地址连到一个小型键值存储（TCPStore）。
2. **各自注册**：每个 rank 调用此函数时，都往 store 里写入"我到了"。
3. **store-based barrier**：函数结尾轮询检查，**直到凑够 `world_size` 个注册才返回**。

### 实际时序（谁先到谁等）

```
① llm_engine L37  process.start()   → 主进程启动所有 worker（异步，不等待）
② worker 执行 __init__，率先到达 L26 → 在 init_process_group 内部注册自己 → 卡在这一行轮询等待
③ 主进程 L39 创建 rank 0 ModelRunner → rank 0 到达 L26 → 注册自己
④ 凑齐 world_size 个 → 所有 rank 的 init_process_group 同时返回 → 继续往下初始化（加载模型等）
```

- `process.start()` 是异步的：主进程启动 worker 后立刻继续执行 L39，不等 worker 初始化完，所以 **worker 确实比 rank 0 更早进入 `__init__`**。
- `init_process_group` 是**双向汇合屏障**：先到的一方阻塞等后到的一方（这里就是 worker 等 rank 0）。
- worker 不是"等 rank 0 到达才开始初始化"，而是**初始化的第一步（L26）本身就卡住了**，后面的模型加载等代码自然执行不到。

### 验证方法

在 L26 前后加 print，会看到 worker 先打出 `before` 后停住，等 rank 0 到达后所有进程才一起打出 `after`：

```python
print(f"rank {rank}: before init")
dist.init_process_group(...)
print(f"rank {rank}: after init")
```

### 项目内的间接证据

- `llm_engine.py` L42-46 注释：`init_process_group() which is a collective barrier ... rendezvous`。
- `model_runner.py` L105/L116/L119 还有**显式的** `dist.barrier()`（共享内存创建阶段）——那是作者需要自己控制同步时手写的屏障；初始化阶段的屏障 PyTorch 帮你做了。

---

## 9. "领导者"辨析：通信对等 vs 控制集中（角色分岔点）

**初始化阶段：对等，无领导者。** `init_process_group` 是集体操作，所有 rank 平等加入通信组，谁先到谁等。

**运行阶段：有明确的领导者（rank 0）。** 领导者不体现在初始化，而体现在**控制权**：

| | rank 0（driver） | worker（executor） |
|---|---|---|
| 有没有 scheduler | ✅ 有（调度器只在主进程创建） | ❌ 没有 |
| 谁决定"跑哪个 batch、何时跑" | ✅ 只有它能决定 | ❌ 只能等指令 |
| 主动/被动 | 主动 `write_shm` + `set()` 发指令 | 被动 `wait()` → `read_shm()` → 执行（`loop()`） |

worker 进程从启动到结束，**每一步做什么都完全由 rank 0 决定**——这正是"驱动者"的含义。rank 0 不发指令，worker 就永远阻塞在 `event.wait()`。

> 一句话总结：**通信对等（NCCL 集体操作）、控制集中（rank 0 单点驱动）**。底层集合通信人人平等，上层调度控制只有一个大脑（vLLM 等框架同理）。

---

## 10. rank 0 的 `ModelRunner` 实例（`llm_engine.py` L39）

```python
# worker 进程（L21）：单个 event
model_runner = ModelRunner(config, rank, event)
# 主进程（L39）：event 是列表
self.model_runner = ModelRunner(config, rank=0, event=self.events)
```

这一行做了三件事：

1. **以 rank 0 身份加入分布式通信组**——触发上述汇合屏障，与先到的 worker 会合。
2. **拿到所有 worker 的事件列表**——worker 只等自己的信号，单个 event 够用；rank 0 要向**所有** worker 广播指令，必须持有全部事件（`write_shm` 里 `for event in self.event: event.set()`）。
3. **完成本进程的模型初始化**——模型构建、权重加载、warmup、KV cache 分配都在这次 `__init__` 里完成。

之后 `step()` 里的每次前向都通过它发起（`llm_engine.py` L74）：`call()` 先把指令广播给所有 worker，然后 rank 0 自己也执行一次 `run`——所有 rank 同步进入同一次前向。

---

## 11. Scheduler：调度器（引擎的大脑）

```python
self.scheduler = Scheduler(
    max_num_sequences=config.get("max_num_sequences", 16),
    max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
    max_cached_blocks=config.get("max_cached_blocks", 1024),
    block_size=config.get("block_size", 256),
    eos=config.get("eos", 50256)
)
```

### 是管理 prompt 吗？不只是

管理 prompt（waiting queue）只是职责之一。当前实现使用**两个队列**：

| 队列 | 内容 | 管什么 |
|---|---|---|
| waiting | 等待预填充的序列 | prompt 的排队 |
| running | 已接纳、持有缓存且尚未结束的序列 | 运行请求管理 |

被抢占的请求回到 waiting，当前没有独立 swapped 队列。

**为什么必须有调度器？** 核心矛盾：请求是随机到达的、长度不一的，而 GPU 前向是一次算一个 batch。必须有人回答三个问题：

1. **谁上 GPU**？——`max_num_sequences`：本轮 batch 最多选择多少条序列，未选中的留待后续调度
2. **batch 塞多少**？——`max_num_batched_tokens`：一次前向最多多少 token
3. **显存不够怎么办**？——`max_cached_blocks` + `block_size`：KV cache 用 block 管理，不够就 preempt

这就是 **continuous batching（连续批处理）**：每步前向后，完成的序列离开、等待请求在资源允许时进入后续 batch、被抢占的请求等待重算——batch 动态重组，而非静态 batching 等凑齐一批一起进一起出。

> scheduler 必须在 model_runner **之后**创建：不是功能依赖，而是 rank 0 创建 ModelRunner 会卡在汇合屏障上，等所有 worker 到齐才放行，之后再建 scheduler 保证初始化顺序清晰。

### `max_num_batched_tokens`：一次前向的 token 预算

控制**单次前向传播能处理的 token 总数上限**——主要约束 **prefill（预填充）阶段**（decode 阶段每序列每步只产 1 个 token，一般不构成约束）。

```
batch = [seq1(800), seq2(600), seq3(500)] 共 1900 tokens
max_num_batched_tokens = 1024 → 这批塞不下，调度器削减（组不进去的留 waiting）
```

- **太大**：单次前向计算量和 KV cache 写入量暴增，可能 OOM 或延迟抖动
- **太小**：单轮能接纳的请求更少；单条序列长度超过预算时无法进入 prefill，当前代码不会自动切分。

**常见误解辨析：**

1. 它控制本轮输入 token 预算，与模型支持的上下文长度不同。但当前未实现 chunked prefill，单条 2000-token 序列不能在 1024-token 预算下自动切成两轮；若调度无法推进，会触发无进展检查。
2. ❌ "1000 之后会变 1600，超了再卸载调整" → ✅ 两个层面都不成立：
   - **组装时准入**：调度器边组装边计数（`len(seq) + current_scheduled_tokens <= max_num_batched_tokens`），再塞就要超就立即 break——超标的 batch 根本组不出来，不存在"先装再卸"。
   - **decode token 断崖下降**：prefill 之后每条序列每步只产 1 个 token（`current_scheduled_tokens += 1`），10 条 × 100 tokens 的 prefill 后 decode batch 只有 10 个 token，不可能涨到 1600。

### "卸载/替换"确实存在——但由 KV cache 显存触发（preempt 机制）

token 预算超标时序列只是留在 waiting；真正"被踢出去"发生在 **KV cache 块不足**（`max_cached_blocks` 管辖）时：

```python
# scheduler.py：decode 循环里发现块不够
if not self.block_manager.can_append(seq):
    preempted = True
    self.preempt(self.running.pop())   # 踢掉 running 队尾的序列

def preempt(self, seq):
    self.block_manager.deallocate(seq)  # 释放它的全部 KV cache 块
    seq.status = SequenceStatus.WAITING
    self.waiting.appendleft(seq)        # 扔回 waiting 队头重新排队
```

被 preempt 的序列扔回 waiting，重新 prefill 时**重算**（recompute 策略，不是 swap 到 CPU 换回）。

### 两套独立的保护机制

| | `max_num_batched_tokens` | KV cache 不足（preempt） |
|---|---|---|
| 保护对象 | 单次前向的计算量 | 显存容量 |
| 检查时机 | **组装时准入**（塞不下就不塞） | 运行中触发（decode 前发现块不够） |
| 溢出处理 | 序列留 waiting，下轮再试 | 踢序列、释放块、回队头重排 |
| 效果 | batch 永不超标 | 换出-重填（recompute） |

### 其他参数

- `max_num_sequences` — 当前实现中，本轮 batch 的序列数上限
- `max_cached_blocks` × `block_size` — KV cache 总显存预算；block 化管理是 PagedAttention 的基础（按块分配，避免为最长序列预留整段显存）
- `eos` — 判断序列生成结束的 token id，调度器用它标记 finish
- `add_sequence` 里还有前置校验：序列所需块数 > 总容量直接 raise（避免序列在 waiting 里永远排不上、引擎卡死无提示）

---

## 12. atexit.register(self.exit)：退出兜底保险

```python
atexit.register(self.exit)
```

### atexit 是什么

`atexit` 是 Python 标准库，作用一句话：**"在程序结束前，自动替你执行一个函数"**。

```python
import atexit

def cleanup():
    print("收尾工作")

atexit.register(cleanup)   # 只是登记，不执行

print("程序主体")           # 跑完后自动打印 "收尾工作"
```

关键：**从头到尾没有显式调用 `cleanup()`**。Python 解释器退出前会检查登记簿，把登记过的函数挨个执行一遍。所以 `atexit.register(self.exit)` 的意思是：**将来这个进程不管从哪里正常结束，都先自动执行一次 `self.exit()`**。

### 为什么非要有这一行

核心是一个操作系统事实：**主进程结束 ≠ 子进程结束**。

假设用户写了个脚本用完引擎就结束（没写清理代码），worker 此刻在干什么？看 `loop()`：

```python
while True:
    method_name, args = self.read_shm()   # ← 第一步是 event.wait()
```

worker 正阻塞在 `event.wait()`——等主进程点亮信号灯。可主进程已经死了，信号灯**永远不会再亮**。于是 worker 变成**孤儿进程**：

- 占着 GPU 显存（模型权重 + KV cache）
- 占着 NCCL 通信资源
- 占着共享内存段（名为 `myvllm` 的 1MB）
- 只能手动 `kill -9` 清掉

打比方：老板（主进程）下班直接走人，没宣布散会，员工（worker）就永远坐在工位上等指示。`atexit.register` 强制规定：**老板离开办公室前必须先宣布散会**。

### 退出时实际发生什么（完整时序）

```
① self.model_runner.call("exit")
   ├─ rank 0 的 call() → write_shm("exit")
   │    ├─ 把 "exit" 指令 pickle 后写进共享内存
   │    └─ for event in self.event: event.set()  ← 点亮所有 worker 的信号灯
   ├─（worker 侧）event.wait() 被唤醒
   │    ├─ read_shm() 读出 "exit"
   │    ├─ 执行 ModelRunner.exit()：
   │    │    shm.close() / del graphs / torch.cuda.synchronize()
   │    │    / dist.destroy_process_group()
   │    └─ loop() 里 method_name == 'exit' → break 跳出死循环
   │         → worker_process 返回 → 子进程自然结束 ✅
   ├─ rank 0 自己也执行一遍 exit()（close + unlink 共享内存）
② for process in self.processes:
       process.join()   ← 阻塞等每个 worker 真正退出并回收（避免僵尸进程）
```

关键设计：**exit 善后动作本身就是一次正常的指令广播**——和发 `"run"` 走完全相同的链路（共享内存 + event），worker 不需要任何特殊代码，只是"收到一条叫 exit 的指令，执行后发现自己该退出了"。

### atexit 救不了的情况（第二层保险）

| 退出方式 | atexit 触发？ |
|---|---|
| 脚本自然跑完 / `sys.exit()` / Ctrl+C | ✅ 触发 |
| `kill -9`（SIGKILL）、断电、内核崩溃 | ❌ 不触发 |
| 段错误（CUDA 非法内存直接打死进程） | ❌ 不触发 |
| `os._exit()` | ❌ 不触发 |

硬崩溃时 worker 照样变孤儿。所以 ModelRunner 初始化共享内存前有**第二层保险**——下次启动时主动清理残留：

```python
try:
    old_shm = SharedMemory(name='myvllm')
    old_shm.close()
    old_shm.unlink()      # 强制删掉上次没清理干净的共享内存
except FileNotFoundError:
    pass                   # 没有残留，正常
self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
```

两层防御各管一头：

- **atexit**：管"正常退出但忘了写清理代码"——什么都不用做，它兜底
- **启动时 unlink**：管"上次异常崩溃留下的残留"——否则 `SharedMemory(create=True)` 会因重名报错

---

## 13. `last_token`：从首次输入到多进程 decode 的完整链路

### 为什么初始化时就保存 prompt 的最后一个 token？

```python
# sequence.py：Sequence.__init__
self.token_ids = copy(token_ids)
self.last_token = self.token_ids[-1] if self.token_ids else None
```

`last_token` 表示**当前序列末尾的 token ID**。刚创建时序列只有 prompt，所以它暂时保存 prompt 的最后一个 token。空列表时赋值 `None`，避免 `[-1]` 抛出 `IndexError`；这不代表后续推理支持空输入。

**保存这个字段，不代表首次推理只输入它，也不代表 prefill 结束后还会把 prompt 末尾 token 单独输入一次。** 正常首次 prefill 从 `token_ids` 构造输入，生成新 token 后，`append_token()` 会覆盖 `last_token`。

以下以单条请求为例：prompt 编码为 `[A, B, C]`，随后生成 `D、E、F`。假设没有前缀缓存命中、没有抢占重算，且生成 `D` 后尚未满足停止条件。字母代表 token ID。

### 主进程的调用链：创建 → prefill → 更新 → decode → 返回

```text
LLMEngine.generate(prompts, sampling_params)
  │
  ├─ add_prompt(prompt, sampling_params)
  │    ├─ tokenizer.encode(prompt) → [A, B, C]
  │    ├─ Sequence(...)
  │    │    token_ids = [A, B, C]，last_token = C
  │    │    num_tokens = num_prompt_tokens = 3
  │    └─ scheduler.add_sequence(seq) → 加入 waiting
  │
  └─ while not scheduler.is_finished(): step()
       │
       ├─ 首次调度：scheduler.schedule()
       │    从 waiting 取出 → 分配 KV cache 块 → 放入 running
       │    返回 (seqs, True)，本轮执行 prefill
       │
       ├─ model_runner.call("run", seqs, True)
       │    ├─ 多进程时先 write_shm，通知 worker 执行同一次 run
       │    └─ 本地 run(seqs, True)
       │         ├─ prepare_prefill(seqs) → 输入 [A, B, C]
       │         ├─ run_model() → 模型前向 → compute_logits()
       │         └─ rank 0 采样 → 返回 [D]
       │
       ├─ outputs.cpu().tolist()
       ├─ scheduler.postprocess(seqs, [D])
       │    ├─ seq.append_token(D)
       │    │    token_ids = [A, B, C, D]，last_token = D
       │    └─ 检查 EOS、max_tokens、max_model_length
       │
       ├─ 后续调度到该请求执行 decode：scheduler.schedule()
       │    从 running 选择请求，检查/安排缓存块
       │    返回 (seqs, False)
       │
       ├─ model_runner.call("run", seqs, False)
       │    └─ run(seqs, False)
       │         ├─ prepare_decode(seqs) → 读取 last_token，输入 [D]
       │         ├─ run_model() → 结合已有 KV cache 执行前向
       │         └─ rank 0 采样 → 返回 [E]
       │
       ├─ scheduler.postprocess(seqs, [E])
       │    └─ seq.append_token(E)
       │         token_ids = [A, B, C, D, E]，last_token = E
       │
       └─ 重复，直到满足停止条件
            postprocess 标记 FINISHED、释放缓存块、移出 running
            step 提取 completion_token_ids
            generate 汇总结果，用 tokenizer.decode() 得到生成文本
```

进入 `running` 队列不等于本轮已经是 decode；本轮阶段由 `schedule()` 返回的 `is_prefill` 决定。多请求情况下，后续 step 也可能先处理其他请求的 prefill，上图表示该请求正常推进的顺序。

### 三处关键代码：首次输入从哪里取，覆盖在哪里发生，decode 又读什么？

**① Prefill 从完整 token 列表中取尚未缓存的部分，不读取 `seq.last_token`。**

```python
# model_runner.py：prepare_prefill
token_ids = seq.token_ids
num_cached_tokens = seq.num_cached_tokens
input_ids.extend(token_ids[num_cached_tokens:])
```

本例没有前缀缓存，`num_cached_tokens = 0`，所以输入为 `[A, B, C]`。若命中前缀缓存，则只输入未缓存的后缀。

**② Prefill 采样结果也会进入 `postprocess()`，最终重新赋值 `last_token`。**

```python
# llm_engine.py：step，prefill 和 decode 都会走这里
outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
if outputs is not None:
    outputs = outputs.cpu().tolist()
self.scheduler.postprocess(scheduled_sequences, outputs)

# scheduler.py：postprocess
for seq, token_id in zip(seqs, token_ids):
    seq.append_token(token_id)

# sequence.py：append_token
def append_token(self, token_id):
    self.token_ids.append(token_id)
    self.last_token = token_id  # 传入 D 时，字段值从 C 变成 D
    self.num_tokens += 1
```

覆盖的是 `last_token` 字段；原来的 `C` 仍在 `token_ids` 中。模型刚返回 `[D]` 时字段仍为 `C`，执行上述后处理才变为 `D`。

**③ Decode 读取更新后的 `last_token`。**

```python
# model_runner.py：prepare_decode
input_ids.append(seq.last_token)  # D
context_lens.append(len(seq))    # 4，即 A、B、C、D 的序列长度
```

| 时刻 | 主进程的 `token_ids` | `last_token` | 模型输入/缓存状态 |
|---|---|---|---|
| 初始化 | `[A, B, C]` | `C` | 尚未推理 |
| Prefill 执行 | `[A, B, C]` | `C` | 输入 `[A, B, C]`，计算并缓存它们的 K/V |
| 采样并追加 D 后 | `[A, B, C, D]` | `D` | D 尚未经过模型，没有自己的 K/V |
| 第一次 decode 执行 | `[A, B, C, D]` | `D` | 输入 `[D]`，计算 D 的 K/V，结合历史缓存预测 E |
| 采样并追加 E 后 | `[A, B, C, D, E]` | `E` | 下一次 decode 输入 E |

### 注意：LM Head 中的 `last_token` 是另一个变量

```python
# layers/embedding_head.py：ParallelLMHead.forward
if context.is_prefill:
    last_token = context.cu_seqlens_q[1:] - 1
    x = x[last_token].contiguous()
logits = torch.nn.functional.linear(x, self.weight)
```

这里的局部变量 `last_token` 是**每条序列最后一个查询位置的索引张量**，不是 `Sequence.last_token` 中保存的 token ID。本例选择 C 所在位置的隐藏状态，用它计算预测 D 的 logits；它没有读取 `seq.last_token`。

### 多进程传输：worker 下一轮如何拿到 D？

rank 0 持有调度器管理的序列，并在采样后调用 `postprocess()` 更新它。worker 没有自己的调度器，也不负责采样；每轮从共享内存反序列化出本轮需要的序列状态。

```text
rank 0：call("run", seqs, is_prefill)
  → write_shm() → pickle.dumps() → Sequence.__getstate__()
  → 写共享内存 → event.set()

worker：loop() → read_shm()
  → event.wait() → pickle.loads() → Sequence.__setstate__()
  → event.clear() → call("run", ...) → run(...)
```

序列化时，除长度、缓存计数、block table 等元数据外，token 数据由以下表达式决定：

```python
# sequence.py：__getstate__
self.token_ids if self.num_completion_tokens == 0 else self.last_token
```

- 首次 prefill 前：没有生成 token，发送完整 `[A, B, C]`；worker 恢复 `token_ids = [A, B, C]`、`last_token = C`。
- 首次 prefill 后：rank 0 已执行 `append_token(D)`；下一次正常 decode 发送的是 D，worker 恢复 `token_ids = [D]`、`last_token = D`，而 `num_tokens` 仍为传来的完整长度 4。
- worker 的 `token_ids = [D]` 是本轮计算所需的精简状态，历史上下文依靠各 rank 的 KV cache 和传来的 block table 使用；主进程仍保留完整 token 历史。

这里的判断条件严格说是“是否已经生成 token”，并非直接检查 `is_prefill`。上述对应关系适用于首次 prefill → 正常 decode；不能据此断言有生成历史的抢占重算也会发送完整列表。

### 因果关系：为什么必须传，为什么后续只传一个？

**需要把新 token 传给 worker，是因为本实现只在主进程采样，而张量并行的各 rank 下一轮需要处理相同的输入。**

```python
# model_runner.py：run
token_ids = None
if self.rank == 0:
    token_ids = self.sampler(logits, self.prepare_sample(seqs))
return token_ids
```

所有 rank 都参与模型前向，但只有 rank 0 执行采样，得到新 token ID D。worker 此时没有这个采样结果；直到下一轮 rank 0 通过共享内存发送 D，worker 才获得它。这里的“得到 token”指采样得到 token ID；最终把 ID 转成文本的 `tokenizer.decode()` 也在主进程执行。

**正常 decode 只需传最后一个 token，是因为各 rank 已经保存了历史 token 对应的 KV Cache。** 历史上下文仍然参与注意力计算，只是不必再次作为完整 token 列表传入并重新计算。

```text
首次 prefill（无前缀缓存命中）：
rank 0 发送完整 prompt [A, B, C]
→ 各 rank 协同前向，保存各自负责部分的 KV Cache
→ rank 0 采样得到 D，worker 此时没有 D

下一轮正常 decode：
rank 0 发送 D + 序列长度、缓存块表等必要元数据
→ 各 rank 以 D 为输入，结合本地历史 KV Cache 协同前向
→ rank 0 采样得到 E
→ 下一轮再发送 E，重复此过程
```

三个层次应区分：

| 层次 | 原因或作用 |
|---|---|
| 张量并行的输入一致性要求 | 各 rank 协同处理同一批序列，需要获得一致的本轮输入 |
| 本实现的采样与传输策略 | 只在 rank 0 采样，因此由 rank 0 将新 token 分发给 worker；这不是所有张量并行实现唯一可用的策略 |
| KV Cache 带来的增量计算 | 正常 decode 只需新 token，所以序列化时可省去完整历史 token 列表 |

共享内存 + Event 是这个项目分发指令和输入的具体机制；模型前向中的 NCCL 张量通信属于另一层，仍然需要执行。“只传最后一个 token”说的是 token 列表的精简，不是说整轮通信只有一个整数，也不是说历史上下文被丢弃。

### 回到最初的问题

代码可以证明：**正常首次 prefill 构造输入时不读取初始化保存的 C；生成 D 后，该字段被覆盖为 D，后续 decode 才读取它。**

因此，初始化时保存 prompt 最后一个 token 并不是这条正常推理路径的计算必需条件。让 `last_token` 从对象创建起就与序列末尾一致，是合理的设计解释；但源码没有明确说明作者动机，不能把这一解释当作作者的原话。列表末尾访问本身也是 O(1)，单独设置字段并不改变该操作的时间复杂度。

源码定位（行号可能随后续修改移动）：

| 文件 | 方法及本次核对行号 |
|---|---|
| `src/myvllm/engine/llm_engine.py` | `step` L69、`add_prompt` L89、`generate` L96 |
| `src/myvllm/engine/scheduler.py` | `add_sequence` L21、`schedule` L35、`postprocess` L104 |
| `src/myvllm/engine/sequence.py` | `__init__` L17、`append_token` L83、`__getstate__` L88、`__setstate__` L97 |
| `src/myvllm/engine/model_runner.py` | `read_shm` L125、`write_shm` L134、`loop` L162、`call` L174、`prepare_prefill` L265、`prepare_decode` L318、`run_model` L354、`run` L386 |
| `src/myvllm/layers/embedding_head.py` | `ParallelLMHead.forward` L70 |

---

## 14. 前缀缓存命中：完整块、未满块与引用计数

### `num_cached_tokens` 统计什么？

```python
# sequence.py：初始化
self.num_cached_tokens = 0

# block_manager.py：allocate，每命中一个完整块
seq.num_cached_tokens += self.block_size

# model_runner.py：prepare_prefill，跳过命中的前缀
input_ids.extend(token_ids[num_cached_tokens:])
```

它表示**当前序列命中并可复用的前缀缓存 token 数量**，不是该请求在 KV Cache 中已有的全部 token 数量，也不会随着每轮 decode 自动加一。当前实现按完整块累加，因此其值是 `block_size` 的非负整数倍：块大小为 8 时，只会是 `0、8、16、24……`。

### 示例：请求 A 有 10 个 token，请求 B 有 11 个 token

假设 `block_size = 8`，A 的首次 prefill 没有命中缓存；B 的前 10 个 token 与 A 相同。下面数字表示 token 的位置，B 的第 11 个 token 是额外输入。

```text
A 完成 prefill 后：
第一块：[1 2 3 4 5 6 7 8] 的 K/V
第二块：[9 10 _ _ _ _ _ _] 的 K/V

A 后续 decode：
使用前面全部 10 个 token 的 K/V，包括第二块中的 2 个。
新生成的 token 下一轮经过模型后，其 K/V 继续写入第二块空位。

B 到来时，按完整块匹配：
[1 2 3 4 5 6 7 8] [9 10 11]
 └─ 第一块可命中     └─ 不满一块，3 个 token 都需要计算

预期命中结果：num_cached_tokens = 8
预期 prefill 输入：B.token_ids[8:]，即第 9、10、11 个 token
```

即使第 9、10 个 token 也相同，也不能只复用半块。`allocate()` 仅对完整块计算匹配哈希，未满块使用 `-1`，不登记到哈希映射中：

```python
h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) \
    if len(token_ids) == self.block_size else -1

# 分配新块后
if h != -1:
    self.hash_to_block_id[h] = block.block_id
```

**未满块不能作为前缀命中，不代表它不能保存和使用 K/V。** `Attention.forward()` 会通过 `store_kvcache()` 写入当前 token 的 K/V；decode 再通过 `paged_attention_decode()`，结合 `block_tables` 和 `context_lens` 读取已有缓存。因此，同一请求可以使用自己的未满块，不必等待它填满。

### 当前代码命中旧块的前提：旧块仍被引用

`ref_count` 表示当前有多少个请求引用该块。请求释放缓存时，对所引用的块逐一减计数；归零后调用 `_deallocate_block()`：

```python
# block_manager.py：deallocate
block.ref_count -= 1
if block.ref_count == 0:
    self._deallocate_block(block_id)

# _deallocate_block 中的关键操作
block.token_ids = []
self.used_block_ids.remove(block_id)
self.free_block_ids.append(block_id)
```

清空 `token_ids` 后，后来的请求无法通过匹配检查：

```python
self.blocks[block_id].token_ids != token_ids
# [] != [相同前缀的 8 个 token]，判定未命中
```

因此，在当前实现中：

- 第一块仍被 A 或其他请求引用：匹配信息保留，B 有可能命中它；还需要前缀一致、对应哈希映射存在等条件。
- 第一块引用数归零并释放：匹配信息被清除，后来的 B 需要重新计算这部分。

这里清除的是匹配元数据，**不一定立即清零 GPU 中的 K/V**。物理块进入空闲队列，之后可以被重新分配和覆盖。即使 A 结束，只要其他请求仍引用该块，它也不会因 A 的结束而立即释放。

### 实现边界：匹配命中不等于跨请求复用已经正确运行

上面的 B 命中 8 个、重新计算 3 个，描述的是**缓存匹配结果和预期计算方式**。当前代码尚未完整支持这条跨请求复用路径：

- `allocate()` 保留了匹配仍在使用的完整块、增加引用计数的分支。
- 但当前 prefill 的注意力计算使用本轮计算出的 K/V，尚未完整接入命中块的历史 K/V；源码注释还指出 Qwen3 的位置偏移需要相应处理。
- `_deallocate_block()` 清空匹配信息，使已释放的块无法被后续请求命中；这不等于在用块匹配路径已经得到完整支持。

所以应分别记住：**同一请求使用自己的未满块已实现；完整块匹配逻辑存在；跨请求复用的 prefill 计算链路仍不完整。**

### 与 vLLM 的关系

vLLM 的标准自动前缀缓存（APC）也按完整块复用，官方设计文档明确说明只缓存完整块用于前缀复用；当前请求自身的未满块仍可保存 K/V 供 decode 使用。[vLLM 官方设计文档](https://docs.vllm.ai/en/stable/design/prefix_caching/)

但“引用数归零就清除匹配信息”是当前 MinivLLM 的处理，不能当作 vLLM 的通用规则。vLLM 可以保留空闲块的缓存信息，在块尚未被驱逐或覆盖时供新请求命中；没有活跃请求引用，不等于缓存立即失效。上述官方文档的 Free、Eviction 和 Block Allocation 部分描述了这一机制。

---

## 15. `block_table`：逻辑块到物理缓存块的映射

```python
# sequence.py：Sequence.__init__
self.block_table = []
```

`block_table` 记录**当前序列的每个逻辑块，对应 KV Cache 中哪个物理块**。列表下标是逻辑块编号，元素是物理块 ID；它保存的是映射信息，实际 K/V 张量存放在各层的缓存中。初始化为空，表示请求尚未分配缓存块。

### 示例：10 个 token，块大小为 8

这个请求需要两个逻辑块，分配到的物理块可以不连续：

```python
seq.block_table = [5, 2]
```

| 逻辑块编号（列表下标） | 对应 token | 物理块 ID（列表元素） |
|---|---|---|
| 0 | 第 1～8 个 token | 5 |
| 1 | 第 9～10 个 token | 2 |

```text
序列逻辑顺序： [第 1～8 个 token] → [第 9～10 个 token]
                     ↓                     ↓
实际 KV 存储：    物理块 5               物理块 2
```

**逻辑顺序连续，不要求物理块连续。** 因而无需为整个序列申请一大段连续缓存空间，可以从块池中分配可用块，再通过这张表按序访问。

### 如何定位某个 token 的 K/V？

对于从 0 开始的 token 下标 `t`：

```python
logical_block = t // block_size
physical_block = seq.block_table[logical_block]
offset = t % block_size
```

例如第 10 个 token 的下标为 `9`：

```python
logical_block = 9 // 8              # 1
physical_block = seq.block_table[1] # 2
offset = 9 % 8                     # 1
```

即读取**物理块 2、块内下标 1**处的 K/V。实际访问张量时还会结合层、KV head、维度等索引；这里只展示 token 到缓存位置的映射。

### 在当前代码中的生命周期

1. **创建请求**：`Sequence.__init__()` 将表初始化为 `[]`。
2. **分配或匹配缓存块**：`BlockManager.allocate()` 按逻辑顺序执行 `seq.block_table.append(block.block_id)`。
3. **生成过程中扩容**：`BlockManager.append()` 在新 token 需要一个新块时，将新分配的物理块 ID 追加到表尾；原有末尾块还有空间时，继续使用原块。
4. **写入新 K/V**：`prepare_prefill()` 和 `prepare_decode()` 根据块表构造 `slot_mapping`，供 `store_kvcache()` 定位写入位置。
5. **Decode 读取历史 K/V**：各序列的块表被补齐并组成批量 `block_tables` 张量，传给 `paged_attention_decode()`；内核通过逻辑块编号查出物理块 ID，并结合 `context_lens` 只访问有效 token。
6. **释放请求缓存**：`BlockManager.deallocate()` 遍历块表减少各块引用计数，再将该序列的 `block_table` 清空。物理块是否回到空闲队列，取决于引用计数是否归零。

例如 decode 写入最新 token 的位置由以下代码确定：

```python
# model_runner.py：prepare_decode
slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
```

其中 `block_table[-1]` 找到末尾物理块，`last_block_num_tokens - 1` 找到最新 token 在块内的下标。

### 与其他字段的区别

| 字段 | 保存的内容 |
|---|---|
| `token_ids` | token ID 列表 |
| `last_token` | 当前序列最后一个 token ID |
| `num_cached_tokens` | 命中的前缀缓存 token 数量 |
| `block_table` | 当前序列逻辑块到物理缓存块 ID 的映射，包含新分配的块和命中的块 |

多进程时，`Sequence.__getstate__()` 也会传递 `block_table`，让 worker 定位自己所在 rank 的 KV Cache。**传递块表是传递索引元数据，并不是把整份 K/V 张量通过共享内存复制过去。**

建议注释：

```python
# 当前序列的逻辑块到 KV Cache 物理块的映射：
# 下标为逻辑块编号，元素为物理块 ID；分配缓存块时填充。
self.block_table = []
```

源码位置：`src/myvllm/engine/sequence.py`、`src/myvllm/engine/block_manager.py`、`src/myvllm/engine/model_runner.py`、`src/myvllm/layers/attention.py`。

---

## 16. pickle 与 `Sequence` 的序列化、状态恢复

### pickle 是什么，负责什么？

`pickle` 是 Python 自带的对象序列化模块，把 Python 对象转换成字节，也能根据字节恢复对象：

```python
import pickle

data = pickle.dumps(obj)  # 序列化：Python 对象 → bytes
obj = pickle.loads(data)  # 反序列化：bytes → Python 对象
```

不同进程不能直接使用对方的普通 Python 对象。本项目先把方法名和参数序列化，再通过共享内存传给 worker：

```python
# model_runner.py：主进程 write_shm
data = pickle.dumps((method_name, *args))

# worker 的 read_shm
method_name, *args = pickle.loads(self.shm.buf[4:n+4])
```

以 `call("run", seqs, is_prefill)` 为例，打包的数据包含方法名 `"run"`、序列列表和阶段标志。pickle 遍历序列列表中的 `Sequence` 对象时，会使用它们自定义的状态方法。

| 组件 | 职责 |
|---|---|
| `pickle` | 将对象编码成字节、从字节恢复对象 |
| 共享内存 | 存放这些字节，供另一进程读取 |
| `Event` | 通知 worker 数据已经准备好 |
| `__getstate__()` | 定义对象序列化时提取哪些状态 |
| `__setstate__(state)` | 定义反序列化时如何根据这些状态恢复对象 |

**pickle 本身不负责通信。** worker 恢复出来的是本进程中的对象，并不是与主进程共享同一个 `Sequence` 实例。

### 两个状态方法何时被调用？

```text
主进程的 Sequence
  → pickle.dumps(...)
  → 自动调用 Sequence.__getstate__()
  → 将状态元组编码成字节
  → 写共享内存，Event.set() 通知 worker

worker 读取共享内存
  → pickle.loads(...)
  → 创建待恢复的 Sequence 对象
  → 自动调用 Sequence.__setstate__(state)
  → 得到本轮推理需要的序列状态
```

本项目这种常规反序列化不会重新调用 `Sequence.__init__()`，因此接收端需要在 `__setstate__()` 中设置所需字段。`state` 是 `__getstate__()` 返回、经 pickle 编解码后得到的状态元组。

### `__getstate__()`：提取需要传输的状态

```python
def __getstate__(self):
    # pickle 序列化时自动调用，提取需要传给 worker 的序列状态
    return (
        self.num_tokens,
        self.num_prompt_tokens,
        self.num_cached_tokens,
        self.block_table,
        # 尚未生成 token 时返回完整 token_ids，否则只返回最新 token
        self.token_ids if self.num_completion_tokens == 0 else self.last_token
    )
```

这里的条件严格说是**是否已经生成 token**，并非直接判断 `is_prefill`。正常首次 prefill 尚无生成 token，所以发送完整列表；正常 decode 已有生成 token，所以只发送最新 token。被抢占后重新 prefill 的序列可能已有生成历史，不能将这个条件简单注释为“是否为 prefill”。

### `__setstate__()`：恢复接收端的本轮状态

```python
def __setstate__(self, state):
    # pickle 反序列化时自动调用，恢复主进程传来的本轮推理状态
    (
        self.num_tokens,
        self.num_prompt_tokens,
        self.num_cached_tokens,
        self.block_table,
        last_token_or_ids
    ) = state

    num_completion_tokens = self.num_tokens - self.num_prompt_tokens
    if num_completion_tokens == 0:
        # 尚未生成 token：恢复完整列表，供首次 prefill 使用
        self.token_ids = last_token_or_ids
    else:
        # 已有生成 token：只恢复最新 token 的列表，供正常 decode 使用
        self.token_ids = [last_token_or_ids]

    self.last_token = self.token_ids[-1] if self.token_ids else None
```

“恢复状态”不等于“让 worker 的全部字段与主进程完全一样”。例如 prompt 为 `[A, B, C]`，主进程生成并追加 D 后，下一轮正常 decode 的状态是：

| 字段 | 主进程 | worker 反序列化后 |
|---|---|---|
| `token_ids` | `[A, B, C, D]` | `[D]` |
| `last_token` | `D` | `D` |
| `num_tokens` | `4` | `4` |
| `num_prompt_tokens` | `3` | `3` |
| `block_table` | 当前序列的块映射 | 传来的相同块 ID 映射，用于本 rank 的缓存 |

worker 不保留完整 token 历史列表，但保留完整长度等元数据，并结合本地 KV Cache 执行 decode。因此不能用 worker 的 `len(token_ids)` 代替完整序列长度。

### 与 `append_token()` 的区别，以及注释应如何表述

| 方法 | 触发时机 | 作用 |
|---|---|---|
| `append_token()` | 主进程采样得到新 token 后，由 `postprocess()` 调用 | 追加 token、更新 `last_token`、增加序列长度 |
| `__getstate__()` | pickle 序列化对象时自动调用 | 提取待传输状态 |
| `__setstate__()` | pickle 反序列化对象时自动调用 | 恢复接收端对象的状态，不主动生成 token |

注释应表达“pickle 自动调用”“恢复本轮推理状态”“是否已有生成 token”，避免写成“子进程主动更新完整序列”或“与主进程所有字段保持一致”。`num_tokens - num_prompt_tokens` 的直接含义是“已生成的 token 数量”，随后才用它判断是否已有生成 token。

---

## 17. `SequenceStatus` 与 Scheduler 的设计思路

### 请求状态与推理阶段是两个维度

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
```

`Enum` 将固定状态组织成有名字的常量，`auto()` 自动分配不同的值。调度代码比较状态名称，不依赖手写的数值。

```text
创建请求 → WAITING
             ↓ 被调度器选中，分配缓存块
           RUNNING
             ├─ 满足结束条件 → FINISHED
             └─ 缓存不足，被抢占 → WAITING，等待重新 prefill
```

`RUNNING` 不等于 decode：请求首次被选中执行 prefill 时就已进入运行队列；本轮阶段由 `schedule()` 返回的 `is_prefill` 决定。

### 三个对象如何配合？

| 对象 | 职责 |
|---|---|
| `Sequence` | 描述请求的 token 历史、长度、状态及块映射 |
| `BlockManager` | 管理空闲块、在用块、分配、引用计数和回收 |
| `Scheduler` | 结合请求和资源状态，选择本轮执行者与推理阶段 |

块管理不是调度前一次性完成的准备工作，而是嵌入每一轮调度：查看候选序列 → 检查 token 预算和缓存空间 → 准备缓存块 → 加入本轮 batch → 推理。

### 调度循环

```text
add_sequence(seq)
  → 检查序列所需块数是否超过块池总容量
  → 加入 waiting

schedule()
  → 优先从 waiting 队首选取 prefill 请求
     检查 can_allocate、token 预算、batch 序列数
     分配缓存块，设为 RUNNING，加入 running
     只要选中 prefill 请求，就返回 (seqs, True)
  → 没选到 prefill 时，从 running 选择 decode 请求
     检查 can_append；必要时抢占其他请求
     调用 BlockManager.append 准备缓存位置
     每个请求计入 1 个输入 token 的预算
     将选中请求按原顺序放回 running，返回 (seqs, False)

ModelRunner.run()
  → 执行模型前向，rank 0 采样

postprocess()
  → Sequence.append_token：追加采样 token，更新 last_token 和长度
  → 检查 EOS、max_tokens、max_model_length
  → 完成则标记 FINISHED、释放缓存引用、移出 running
```

当前策略是 prefill 优先，同一轮不混合 prefill 和 decode。waiting 队首放不下时直接停止选取，不会继续找后面更短的请求；`max_num_sequences` 限制的是本轮 batch 的请求数。Prefill 按 `len(seq)` 计预算，没有扣除前缀命中数，也没有实现 chunked prefill。

Decode 缓存不足时，优先抢占 running 队尾的其他请求，再重试队首；没有其他请求可抢占时，抢占当前请求。`preempt()` 释放缓存引用、保留 token 历史并将请求放回 waiting 队首，表达的是重算策略，不是 CPU/GPU 缓存换入换出。前述多进程重算及前缀复用限制仍然适用。

### `is_finished()`：什么时候可以结束生成循环？

```python
def is_finished(self):
    return len(self.waiting) == 0 and len(self.running) == 0
```

`LLMEngine.generate()` 用 `while not self.scheduler.is_finished()` 持续执行 `step()`。必须两个队列都为空才能结束：waiting 为空时仍可能有请求正在生成；running 为空时仍可能有请求等待首次或重新 prefill。它判断的是全部请求是否完成，`Sequence.is_finished` 则判断单条请求的状态。

### 每轮调度的三个局部变量

| 变量 | 含义 |
|---|---|
| `scheduled_sequences` | 本轮选中、准备交给模型执行的序列，不等于所有 running 请求 |
| `current_scheduled_tokens` | 本轮输入 token 的累计预算；prefill 每条计 `len(seq)`，decode 每条计 1 |
| `preempted` | 本次调度是否发生过抢占，用于判断空 batch 是否允许重试 |

三个变量每次调用 `schedule()` 都重新初始化，不跨轮累计。

### Decode 达到预算上限后，为什么把序列放回？

```python
seq = self.running.popleft()
# ……检查缓存空间……
if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
    self.running.appendleft(seq)
    break
```

`popleft()` 是取出并删除队首元素。若本轮输入 token 数或序列数已达上限，当前 seq 尚未选入 batch，就要放回队首，避免请求丢失。`break` 只停止本轮选取，不结束整个推理。Decode 每条序列输入一个 token，因此在这一阶段两个上限共同限制本轮可选的序列数。

### 为什么用 `extendleft(reversed(...))`？

选入 batch 的序列也已被 `popleft()` 从 running 暂时取出，但还未完成，必须恢复到运行队列，后续才能继续调度：

```text
原 running：                  [A, B, C, D]
取出 A、B，选入本轮 batch：    scheduled_sequences = [A, B]
剩余 running：                [C, D]

extendleft([A, B])：           [B, A, C, D]  ← 逐个左插会反转顺序
extendleft(reversed([A, B]))： [A, B, C, D]  ← 保持原顺序
```

放回队首保留了原有调度优先级；如果改为 `extend(scheduled_sequences)`，就会放到队尾，变成 `[C, D, A, B]` 的轮转顺序。当前代码在模型执行前恢复 running，执行后的 `postprocess()` 再移除真正完成的请求。

### 空 batch：什么时候重试，什么时候报错？

```python
if scheduled_sequences:
    self.running.extendleft(reversed(scheduled_sequences))
elif not preempted and (self.waiting or self.running):
    raise RuntimeError(...)
```

进入 `elif` 隐含了本轮选中列表为空，再加上没有抢占、还有未完成请求，说明本轮没有推进。当前同步执行流程中，原样再调度仍会遇到相同条件，因此报错以避免无限空转。

例如 waiting 只有长度为 100 的请求，token 预算为 64，running 为空：该请求无法整体进入 prefill，又没有 decode 可执行，也没有发生抢占。如果没有这个检查，引擎会一直重复返回空 batch。报错文字提到的 KV Cache 容量不足和缓存块泄漏并未覆盖所有原因，token 预算等限制也可能导致无进展。

若本轮发生抢占，队列和缓存引用状态已有变化，即使没有选出 batch，也允许下一轮重试。但 `preempted=True` 只记录发生过抢占，不保证一定释放了可用物理块（共享块可能仍有引用），也不保证后续一定成功。两个队列都空时是正常完成，无需报错。

### 动态 batch、阶段分轮与 PD 分离

当前调度器每轮重新选择 batch，完成请求可退出，等待请求可在后续轮次加入，这是 continuous batching 的调度方式。`max_num_sequences` 和 `max_num_batched_tokens` 是每轮组装 batch 的上限，本身不等于动态 batching 的全部机制。

当前实现优先选择 prefill；只要选中了 prefill 请求，就返回，不再为本轮选择 decode。**没选到 prefill 才进入 decode 分支，并不要求 waiting 本身为空。** 两个阶段分轮执行，共用模型执行资源和缓存管理。

这与通常所说的 PD 分离部署不同：后者将 Prefill 和 Decode 放在不同 GPU 或实例上执行，并在两端间传递 KV Cache。本项目这里实现的是阶段分轮调度，没有实现这套分离部署。动态 batch 关注每轮选择哪些请求，PD 分离关注两个阶段分别使用哪些执行资源。

### `postprocess()`：先追加 token，再检查结束条件

`zip(seqs, token_ids)` 将本轮序列与各自新生成的 token 配对。本轮推理完成不代表整条请求完成：先执行 `seq.append_token(token_id)`，更新 token 历史、`last_token` 和总长度，再判断是否停止。

| 条件或参数 | 含义 |
|---|---|
| `not seq.ignore_eos and token_id == self.eos` | 未忽略 EOS，且本轮生成 EOS，停止生成 |
| `seq.max_tokens` | 生成 token 数的上限，不包含 prompt |
| `seq.max_model_length` | 可选的序列总长度上限，包含 prompt 和已生成 token |
| `max_num_batched_tokens` | 本轮 batch 输入 token 的预算，不是某条请求的总长度上限 |

例如总长度上限为 4096，prompt 有 3000 个 token，则到达该阈值还可生成 1096 个 token；也可能因 EOS 或 `max_tokens` 提前结束。Decode 虽然只输入最新的一个 token，历史上下文仍通过 KV Cache 参与计算。

当前代码在追加 token 后检查 `>=`，满足任意条件就标记 FINISHED、归还缓存块引用并移出 running；只有引用归零的块才回到空闲池，不是释放 GPU 内存池。`ignore_eos=True` 只忽略 EOS，不取消两个长度停止条件。

注意，这里的 `max_model_length` 是生成后的停止阈值，不是自动截断操作，也不等于在本函数中验证了模型支持的上下文长度。若初始 prompt 已达到或超过阈值，仅靠这里的后处理无法阻止首次前向或追加一个新 token；输入边界需要在生成前另行校验。

---

## 18. Block 与 BlockManager：元数据、哈希和物理块生命周期

### Block 保存的是管理元数据

| 字段 | 含义 |
|---|---|
| `block_id` | 物理缓存块编号 |
| `hash` | 当前块及其前缀的哈希指纹，`-1` 表示尚无有效哈希 |
| `ref_count` | 当前引用该块的请求数量 |
| `token_ids` | 块对应的 token ID 元数据，用于哈希和匹配校验 |

实际 K/V 张量保存在 GPU 缓存中，不在 `Block.token_ids` 中。`Sequence.token_ids` 在主进程保存整个请求的历史，`Block.token_ids` 只对应一个块；后者初次分配时设置，decode 时在块填满后更新，不保证每生成一个 token 都同步更新。因此不能始终根据它的长度判断 GPU 块中已写入多少个 K/V。

块大小和块总数来自 `block_size`、`num_blocks`，不是固定为 256 和 1024。`free_block_ids` 保存空闲块 ID 的队列，数量是 `len(free_block_ids)`；`used_block_ids` 是在用块 ID 的集合。

### 为什么哈希要包含前一个块？

相同 token 在不同前文中，其 K/V 可能不同。例如每块两个 token：

```text
请求 A：[我, 喜欢] [吃, 苹果]
请求 B：[他, 讨厌] [吃, 苹果]
```

第二块内容相同，前缀却不同，不能直接共用。当前实现用链式哈希编码这一差别：

```text
第一块哈希 = hash(第一块 token 的字节)
第二块哈希 = hash(第一块哈希的字节 + 第二块 token 的字节)
第三块哈希 = hash(第二块哈希的字节 + 第三块 token 的字节)
```

前块哈希已包含更早的前缀信息，因此不用每次重新传入整段历史。哈希并非绝对无碰撞的唯一标识；当前实现查表后还比较本块 `token_ids`，但这并不能提供对所有前缀哈希碰撞的绝对保证。

### 两部分如何合起来计算？

```python
h = xxhash.xxh64()
if prefix_hash_value != -1:
    h.update(prefix_hash_value.to_bytes(8, 'little'))
h.update(np.array(token_ids, dtype=np.int32).tobytes())
return h.intdigest()
```

前块哈希编码为 8 字节，本块每个 token ID 编码为 4 字节的 `int32`。连续调用同一个哈希对象的 `update()` 是继续加入字节，不是覆盖之前的输入，效果等价于对两段字节拼接后计算 xxHash64。`intdigest()` 返回最终的 64 位整数结果。第一块传入前缀占位值 `-1`，跳过前块哈希，只输入本块 token 字节。

当前策略只给完整块建立哈希索引；未满块不是数学上不能算哈希，而是实现选择不将它作为可复用完整块登记。

### 分配与回收方法各管哪一层？

| 方法 | 操作范围与作用 |
|---|---|
| `can_allocate(seq)` | 检查空闲块数是否至少等于序列所需块数，不实际分配；未扣除潜在可复用块，检查偏保守 |
| `allocate(seq)` | 逐逻辑块匹配或分配，维护引用计数、命中数量及 `seq.block_table` |
| `_allocate_block(block_id)` | 将一个空闲物理块分配出去，更新块元数据及空闲/在用集合 |
| `deallocate(seq)` | 释放整条序列持有的缓存块引用 |
| `_deallocate_block(block_id)` | 将一个引用数已归零的块归还空闲池 |

`_allocate_block()` 要求 `ref_count == 0`，然后调用 `reset()`、移出空闲队列、加入在用集合。`reset()` 将哈希设为 `-1`、清空 token 元数据，并将引用数设为 **1**：这次调用发生在新请求获得该块时，第一个引用必须被记录。若错误地设为 0，释放时减一将变成 -1，无法触发引用数归零的回收条件。

```text
空闲：ref_count = 0
→ 分配给 A：reset()，ref_count = 1
→ 其他请求匹配并引用：ref_count += 1
→ 某请求释放：ref_count -= 1
→ 归零才回到空闲池
```

这些方法管理预先准备的物理块的使用权，不现场申请新 GPU 显存，也不计算 K/V。

`deallocate(seq)` 遍历块表减引用数，只回收归零的块，最后清空 `seq.block_table` 并将 `seq.num_cached_tokens` 归零，保留 token 历史。它由 Scheduler 在两种情况下调用：请求标记 FINISHED 后，以及请求因资源不足被抢占时。

### 为什么查到哈希后，还要比较 `token_ids`？

```python
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    no_cache_found = True
```

哈希表负责快速定位候选块，token 列表比较负责进一步确认当前块的内容。`block_id == -1` 表示没有候选块；此时 Python 的 `or` 短路，不会继续访问 `self.blocks[-1]`。

内容不匹配可能有两种不同原因：

| 原因 | 发生了什么 | 是否需要不同内容产生相同哈希 |
|---|---|---|
| 哈希碰撞 | 不同输入 A、B 得到相同哈希，查到了内容不符的候选块 | 是 |
| 哈希映射过时 | 原来 `hash(A) → 块 5`，块被清空或重新使用后，旧键仍指向块 5 | 否 |

两者表面上都是“查到了 ID，但内容不匹配”，本质却不同。第二种是**索引记录与块当前内容不一致**，不能归为哈希碰撞。

当前代码中，第二种情况确实可能发生，完整过程如下（哈希值用 12345 示意）：

```text
① 原来缓存有效：
   hash_to_block_id[12345] = 5
   blocks[5].token_ids = [1, 2, 3, 4, 5, 6, 7, 8]

② 引用数归零，调用 _deallocate_block(5)：
   blocks[5].token_ids = []
   块 5 回到空闲队列
   但 hash_to_block_id[12345] 仍然是 5

③ 相同前缀的新请求到来：
   相同输入正常得到相同哈希 12345
   查表仍得到 block_id = 5

④ 比较块的当前元数据：
   [] != [1, 2, 3, 4, 5, 6, 7, 8]
   → 判定缓存未命中，不复用这块旧缓存
```

这个例子完全不需要哈希碰撞，甚至不需要块 5 已经被新内容覆盖：释放时清空匹配元数据，就足以让旧映射过时。GPU 中旧 K/V 可能仍在，并不意味着该块仍是有效的匹配结果。

这次比较仅校验本块 token，不能彻底排除“前缀不同但前缀哈希碰撞、本块 token 又相同”的情况，因此也不能把它理解为绝对无碰撞的保证。

### 为什么会有“缓存命中，但块不在 used_block_ids”这个分支？

一般的保留空闲缓存策略允许：请求 A 结束 → 块引用数归零，进入空闲队列，但保留有效 K/V 和匹配信息 → 尚未被覆盖时，请求 B 前缀相同 → 命中这个无人使用的块。此时应重新占用它：移出空闲队列、加入在用集合、引用数设为 1，同时保留可复用内容。

但**当前实现不具备这条完整路径**。回收会清空 `block.token_ids`，虽然没有删除 `hash_to_block_id` 的旧键，也没有立即清零 GPU K/V，但后面的 token 比较会失败：

```text
旧哈希键仍在 → 查到候选物理块 ID
→ block.token_ids 已是 []，与请求 token 列表不匹配
→ 判定未命中
→ 不会进入“命中且未使用”的分支
```

所以，物理块里可能有旧数据、旧哈希键可能还在，都不足以证明缓存有效。选中分支当前是不可达的预留路径。另外，它调用的 `_allocate_block()` 会 `reset()` 掉哈希和 token 元数据，未来启用空闲缓存复用时，还需区分“分配块给新内容”和“重新激活旧缓存”，不能只删掉回收处的清空操作。前缀缓存的 prefill 读取与位置处理也仍需完善。

---

## 19. Decode 扩容：`can_append()` 与 `append()`

### 先理解调用时机

```text
上一轮前向完成 → 采样出新 token
→ Sequence.append_token() 将它加入历史，num_tokens 已加一
→ 下一轮 Scheduler 调用 can_append() 检查资源
→ BlockManager.append() 准备块并维护元数据
→ 模型前向计算新 token 的 K/V，写入对应物理块
```

最新 token 已在序列中，并不表示它的 K/V 已计算或写入。上一轮输入 A，计算并缓存 A 的 K/V，再采样得到 B；下一轮将 B 作为输入，才计算 B 的 K/V 并采样得到 C。

`BlockManager.append()` 不只是做登记后让前向决定分配：它在需要新块时就从预先申请好的 KV Cache 内存池中领取物理块，并更新块表。随后前向计算 K/V，按已确定的映射写入对应位置。顺序是**先分配位置、维护元数据，再计算并填入数据**。

### `can_append()` 返回的是“空间够不够”，不是“需不需要新块”

```python
if seq.num_tokens % self.block_size == 1:
    return len(self.free_block_ids) > 0
return True
```

以 `block_size = 8` 为例：

| 已包含最新 token 的总长度 | 最新 token 所在位置 | 是否需要新块 | 检查结果 |
|---|---|---|---|
| 8 | 第一块第 8 个位置 | 否 | True |
| 9 | 第二块第 1 个位置 | 是 | 取决于有没有空闲块 |
| 10 | 第二块第 2 个位置 | 否 | True |
| 16 | 第二块第 8 个位置 | 否 | True |
| 17 | 第三块第 1 个位置 | 是 | 取决于有没有空闲块 |

余数为 1 表示最新 token 跨入新块；余数为 0 表示它恰好填满已有块。其余位置也位于已有块内，所以即使空闲池为空仍可返回 True。这里是在合法块状态、前一轮缓存已安排好的前提下检查单步增量容量，不是重新验证整个块表。

### `append()` 的三个分支

| 条件（块大小为 8 的示例） | 操作 |
|---|---|
| 余数 0，例如长度 16 | 本逻辑块刚满，计算前缀哈希，更新块的哈希与 token 元数据，建立或更新哈希映射；不分配新块 |
| 余数 1，例如长度 17 | 最新 token 开始下一块，确认原末尾块已有哈希，分配新块并追加其 ID |
| 其余，例如长度 10 | 使用已有未满块，只检查在用状态和哈希占位值，不分配、不计算完整块哈希 |

完整块哈希在模型前向之前更新，所以“元数据表明完整块”不等于“最新 token 的 K/V 此时已就绪”。不要把第一条注释写成“每生成一个 token 都计算哈希”。

### 为什么修改 `block_tables` 不需要再赋值给序列？

```python
block_tables = seq.block_table
block_tables.append(block.block_id)
```

赋值没有复制列表，两个引用指向同一个列表对象；`append()` 原地修改它，因此 `seq.block_table` 同步变化。

```python
seq.block_table = [5, 2]
block_tables = seq.block_table
block_tables.append(7)
# seq.block_table 现在也是 [5, 2, 7]
```

若改成 `block_tables = block_tables + [7]`，则创建了新列表，仅改变局部变量的指向，需要再赋值给 `seq.block_table` 才会更新序列。

---

## 20. 核心要点速记

1. `spawn` 启动子进程 = 全新解释器，不继承父进程 CUDA 状态（fork 会崩）。
2. `ctx.Process()` 只是创建进程对象，`process.start()` 才真正启动；spawn 下参数走 pickle 序列化传递。
3. 调度只在 rank 0（driver），指令必须广播给所有 worker（executor），保证集合通信对齐。
4. 共享内存传数据（低延迟），Event 传信号（唤醒而非轮询，省 CPU）。
5. 进程句柄必须保存并在退出时 `join()`，避免僵尸进程；事件必须保存才能广播通知，否则 worker 死锁。
6. `init_process_group` 是隐式汇合屏障（TCPStore + store-based barrier）：worker 先到先等，rank 0 到齐后所有 rank 同时放行。
7. 通信对等、控制集中：NCCL 集合操作人人平等，但调度控制权只在 rank 0——"领导者"体现在控制权而非初始化。
8. worker 入口必须改 stdout 为行缓冲：多进程下 stdout 落入块缓冲，print 会静默积压，屏障卡住/死循环期间无法实时观察日志。
9. worker 的角色分岔点在 `loop()`：进循环 = 被动执行者；不进循环（rank 0）= 主动驱动者。
10. `max_num_batched_tokens` 是事前准入的 token 预算（组装时卡住，超标 batch 组不出来）；preempt 是事后显存救济（KV cache 不足踢序列回 waiting 重算）——两套机制独立。
11. prefill 后 decode 的 token 数断崖下降（每序列每步 1 token），不存在"batch token 越推越多需要动态调整"的问题。
12. `atexit.register(self.exit)` 给正常退出上强制保险：退出前自动广播 exit 指令、唤醒卡在 `event.wait()` 的 worker、销毁通信组/共享内存、join 回收子进程；它管不了硬崩溃，硬崩溃的残留靠下次启动时 `unlink` 共享内存兜底——两层防御缺一不可。
13. exit 善后与日常 run 走同一广播链路（共享内存 + event），worker 无需特殊代码——收到 `exit` 指令执行后 break 出死循环自然退出。
14. `last_token` 初始化为 prompt 末尾 token，但正常首次 prefill 从 `token_ids` 取输入；采样后经 `postprocess → append_token` 覆盖为新 token，后续 decode 才读取它。多进程正常 decode 发送更新后的末尾 token，worker 结合本地 KV cache 执行计算。

15. 请求状态与推理阶段不同：RUNNING 请求也可能正在首次 prefill；调度依赖 Sequence 状态、BlockManager 资源和本轮预算。
16. 块哈希结合前块哈希与本块 token 字节，保留前缀信息；哈希表查到 ID 后仍需进行匹配校验。
17. reset 在分配空闲块时将引用数设为 1；deallocate 释放序列引用，只有归零块才回到空闲池，回收不等于清零 GPU K/V。
18. can_append 检查是否有空间容纳最新 token 的 K/V；append 负责准备块及元数据，实际 K/V 写入由模型完成。
19. 当前空闲块匹配元数据被清除，“命中但未使用”的分支不可达；旧哈希键残留不代表可复用缓存仍有效。
