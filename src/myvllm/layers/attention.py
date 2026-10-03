"""
推理注意力：先将本轮 K/V 写入分页缓存，再按 prefill/decode 阶段选择计算内核。

本脚本不包含 QKV 线性投影、Q/K RMSNorm、RoPE 或最终 o_proj；调用方已准备好 Q/K/V。
这里的头数都是当前 rank 的本地头数，不再除以张量并行进程数，各 rank 独立计算本地头。
主流程：Attention.forward -> 可选 store_kvcache -> prefill 或 decode -> 展平本地头输出。

统一维度符号和例子（数值仅用于说明布局，不是实际模型配置）：
    B=2：本轮请求数；A/B 两条请求。
    T=5：prefill 本轮 token 总数，A 有 2 个、B 有 3 个，行顺序 [A0,A1,B0,B1,B2]。
    Hq=4：本地 Q 头数；Hkv=2：本地 K 头数，也是本地 V 头数。
    D=32：每个头的特征宽度；一个 token 的完整本地 Q 有 4*32=128 个特征。
    S=4：block_size，每个物理缓存块能容纳 4 个 token。
    C=6：num_blocks，单层单 rank 的缓存池有 6 个物理块，编号为 0..5。
    M=2：max_num_blocks，decode 块表的列数，每条请求最多列出 2 个逻辑块。
    BLOCK_M/BLOCK_N：内核一次处理的 query/key token 数，是计算分块尺寸，不是缓存块容量 S。
    Triton grid 的一个坐标启动一个 program，不能将它等同于单个 CUDA thread。
    tl.constexpr 参数用于编译期决定维度等信息；Triton 的编译/运行还要求合适的设备和数据类型。

张量形状中每一维的含义：
    prefill Q [T,Hq,D]=[5,4,32]：本轮 token、本地 Q 头、头内特征。
    prefill K/V 各 [T,Hkv,D]=[5,2,32]：本轮 token、本地 KV 头、头内特征。
    K/V cache 各 [C,S,Hkv,D]=[6,4,2,32]：物理块、块内 token 槽、KV 头、头内特征。
    cu_seqlens_q [B+1]=[3]：请求长度前缀和；值为 [0,2,5]，不是长度为 5 的张量。
    slot_mapping [T]=[5]：每个新 token 写入的物理槽位；值为 [8,9,20,21,22]。
    decode Q [B,Hq,D]=[2,4,32]：每条请求的一个当前 token、本地 Q 头、头内特征。
    decode block_tables [B,M]=[2,2]：请求编号、请求内逻辑块编号。
    decode context_lens [B]=[2]：每条请求已有的有效 token 数，包含本轮当前 token。

缓存地址例子：
    A 的前两个 token 放在物理块 2 的槽 0/1，slot=2*S+[0,1]=[8,9]。
    B 的前三个 token 放在物理块 5 的槽 0/1/2，slot=[20,21,22]。
    经过后续生成，另取一个 decode 时刻：A 长度为 3、B 长度为 7。
    context_lens=[3,7]，block_tables=[[2,-1],[5,1]]。
    A 的逻辑块 0 -> 物理块 2；B 的逻辑块 0/1 -> 物理块 5/1。
    块表中的 -1 是无效填充，不是一个可读物理块；物理块编号可以不连续。
    本轮 A2/B6 的写入 slot_mapping=[10,6]，分别为 2*4+2、1*4+2。

GQA：要求 Hkv>0 且 Hq%Hkv==0；本例每组有 4/2=2 个 Q 头。
    Q0/Q1 读取 K0/V0，Q2/Q3 读取 K1/V1；共享 KV 不会合并 Q 的输出。
    每个 Q 头的计算为 softmax(Q @ K.T * s) @ V，s 是已经包含 1/sqrt(D) 的最终缩放。
    prefill 每个 query 只关注本请求中不晚于自己的 token；decode 关注本请求的有效缓存历史。
    输出在内核中分别为 [T,Hq,D] / [B,Hq,D]，Attention 返回前展平为 [T,Hq*D] / [B,Hq*D]。

在线 softmax 共用原理：每个 query 保存最大得分 m、指数和 l、未归一化加权向量 acc。
    新块得分为 scores，更新 m_new=max(m,max(scores))，alpha=exp(m-m_new)，p=exp(scores-m_new)。
    然后 acc=acc*alpha+p@V，l=l*alpha+sum(p)，m=m_new；最后 output=acc/l。
    p 此时没有归一化，不能直接叫最终注意力概率；alpha 将旧累积量换到新的最大值基准。
    无需保存整个序列的得分矩阵，仍能在精确算术下得到完整 softmax 的加权结果。
    手算例子只展示 V 的前两维，其余 30 维设为 0；得分已经缩放并完成有效位置筛选：
        scores=[0,ln(2),ln(3)]，V 的前两维=[[1,0],[0,1],[1,1]]。
        完整概率=[1/6,2/6,3/6]，输出前两维=[2/3,5/6]。
        为说明递推，手算将前三项分成“前两项、最后一项”两块，实际 BLOCK_N 选择不变：
        第一块 m=ln(2)，p=[1/2,1]，l=3/2，acc=[1/2,1]。
        第二块 m_new=ln(3)，alpha=2/3，p=[1]；
        l=(3/2)*(2/3)+1=2，acc=[1/2,1]*(2/3)+[1,1]=[4/3,5/3]。
        acc/l=[2/3,5/6]，与一次性 softmax 的结果一致。

实现范围：底层内核按连续三维 Q/K/V 和连续四维缓存寻址；调用者应满足形状、头数和地址约定。
当前 prefill 直接读取本轮 K/V，不通过块表补读命中的历史前缀；decode 才从分页缓存读取历史。
Attention 中的四维 K/V 展平仅用于缓存写入，没有把后续注意力计算完整适配为四维批输入。
"""

import triton 
import triton.language as tl
from myvllm.utils import get_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr, # 本轮 K [T,Hkv,D] 的首地址，下面按连续张量的元素偏移寻址。
    value_ptr,
    k_cache_ptr, # 缓存 K [C,S,Hkv,D] 的首地址；V 使用相同布局和槽位映射。
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    """
    将一个 token 的一个 KV 头写入其物理缓存槽位；一个 program 处理 D 个特征。
    启动网格 [T,Hkv]，例 [5,2]，总计 10 个 program，不是仅启动 10 个 GPU thread。
    key/value [T,Hkv,D]=[5,2,32]；k_cache/v_cache [C,S,Hkv,D]=[6,4,2,32]。
    slot_mapping [T]=[5]，例值 [8,9,20,21,22]；同一 token 的所有 KV 头写入同一 token 槽。
    """
    # 第 0 个 grid 维度选择本轮 token 行号 t；t=3 对应拼接输入中的 B1，不是位置编码值。
    token_idx = tl.program_id(0) # 每个 program 负责一个 (token,KV头)，不是一个标量特征。
    # 从 slot_mapping 读该 token 的物理槽位，t=3 时 slot_idx=21。
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    # -1 表示这个 token 不写缓存，例如运行器某些预热输入的占位槽；直接跳过该 program。
    if slot_idx == -1:
        return
    
    # 将全池 slot 拆成物理块与块内槽位：21//4=5，21%4=1，即 cache[5,1,...]。
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    # 第 1 个 grid 维度选择 KV 头 h，范围 0..Hkv-1；Q 头不参与 K/V 缓存写入。
    head_idx = tl.program_id(1)
    
    # 生成当前头的 D 个特征编号 [0,...,D-1]，本例形状 [32]。
    head_offsets = tl.arange(0, head_dim)
    # 连续输入 [T,Hkv,D] 的元素偏移为 t*(Hkv*D)+h*D+d。
    # t=3、h=1 时为 3*64+32+[0..31]=[224..255]，正是 key[3,1,:]。
    # 这些是元素偏移，指针类型会处理元素字节数，不需要在公式中另乘 dtype 的字节宽度。
    input_offset = (token_idx * num_kv_heads * head_dim + # skip previous tokens
                    head_idx * head_dim + # skip previous heads
                    head_offsets)

    # 连续缓存 [C,S,Hkv,D] 的元素偏移为 block*(S*Hkv*D)+slot_in_block*(Hkv*D)+h*D+d。
    # 上例写入 cache[5,1,1,:]：5*256+1*64+1*32+[0..31]=[1376..1407]。
    # 一个物理块占 4*2*32=256 个元素，一个 token 槽占 2*32=64 个元素。
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + # skip previous blocks
                   block_offset * num_kv_heads * head_dim + # skip previous positions in block
                   head_idx * head_dim + # skip previous heads
                   head_offsets) 
    
    # 从本轮输入加载当前头的 K/V 向量，各为 [D]=[32]；无需加载或保存 Q。
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)
    
    # 按相同偏移写入 K/V 各自缓存，供当前及后续 decode 查询；本操作不计算注意力。
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    block_size: int
):
    """
    缓存写入的 Python 包装器：整理连续布局、检查部分条件并启动 Triton 内核。

    key/value [T,Hkv,D]：T 个新 token，每个有 Hkv 个头，每头 D 个特征；prefill 例 [5,2,32]。
    k_cache/v_cache [C,S,Hkv,D]：C 个物理块，每块 S 个 token 槽；例 [6,4,2,32]。
    slot_mapping [T]：本轮每行 token 应写的全池槽位，prefill 例 [8,9,20,21,22]。
    block_size=S=4；必须与缓存的第二维一致，不能传计算分块参数 BLOCK_N。
    decode 同一接口可写两条请求的新 token：K/V [2,2,32]，slot_mapping=[10,6]。
    """
    # 从三维 K 读取 T/Hkv/D；缓存的第一维 C 不等于本轮 token 数 T。
    num_tokens, num_kv_heads, head_dim = key.shape
    
    # 手写指针公式没有传入 stride，因此源 K/V 必须连续；必要时 contiguous 会复制数据。
    # 目标缓存同样应有约定的连续布局，由运行器分配；这里没有将目标缓存重新复制一份。
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    # 仅检查缓存形状一致、槽位数量与 token 数一致；头宽、槽位范围等仍由调用方保证。
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"
    
    # grid=[T,Hkv]，每个 program 向量化处理一个 token/头的 D 个特征。
    grid = (num_tokens, num_kv_heads)
    store_kvcache_kernel[grid](
        key, # Triton 将张量参数作为其底层数据指针传入内核。
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    变长 prefill 内核：一个 program 负责某请求、某 Q 头的一块 query 行。

    Q/O [T,Hq,D]，K/V [T,Hkv,D]，cu_seqlens_q [B+1]；只读取当前请求的 token 范围。
    本例 Q/O [5,4,32]，K/V [5,2,32]，请求前缀和 [0,2,5]。
    BLOCK_M=64 表示一个 program 处理最多 64 个 query，BLOCK_N=64 表示一次加载最多 64 个 key。
    实际序列只有 2/3 个 token，超出有效范围的 tile 位置由 mask 屏蔽，不是创建了额外真实 token。

    tile 内部维度：q [BLOCK_M,D]，k [D,BLOCK_N]，qk/p [BLOCK_M,BLOCK_N]，
    v [BLOCK_N,D]，acc [BLOCK_M,D]，m_i/l_i/alpha [BLOCK_M]。
    qk 的行对应 query token，列对应 key token；它只有一个选定 Q 头，不包含额外的头维度。
    在线 softmax 在 key 分块上累积，最终对每个有效 query 输出一个 D 维向量。
    """
    # grid 三个轴依次为 query 分块编号、Q 头编号、请求编号；没有“物理缓存块”轴。
    start_m = tl.program_id(0) # query tile 编号，tile 起始行是 start_m*BLOCK_M。
    off_h = tl.program_id(1) # head index
    seq_idx = tl.program_id(2) # sequence index

    # GQA 均匀分组：group_size=Hq//Hkv；本例为 2，Q 头 0/1 -> KV0，2/3 -> KV1。
    # 调用者须保证 Hkv>0、Hq%Hkv==0，否则整数除法会产生错误映射甚至超出 KV 头范围。
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # 从前缀和取得本请求的拼接区间 [seq_start,seq_end)，请求内长度为二者之差。
    # seq_idx=1 对应 B：start=2、end=5、len=3，所以只能读拼接行 B0/B1/B2。
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    # 网格按最长请求分配 tile，较短请求可能拿到多余 program，此时直接退出。
    if start_m * BLOCK_M >= seq_len:
        return
    
    # offs_m [BLOCK_M] 是当前 tile 的请求内 query 行号；不是全局行号，也不是物理 slot。
    # 例如首 tile 为 [0..63]，B 只有前三项有效；全局行号需加 seq_start=2。
    # offs_d [D]=[32] 是头内特征编号，与 token 行号属于不同维度。
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    # 连续 Q 的地址公式为 global_token*(Hq*D)+q_head*D+feature。
    # offs_m[:,None] [BLOCK_M,1] 与 offs_d[None,:] [1,D] 广播出 [BLOCK_M,D] 地址。
    # B0、Q头3 的起点为 2*4*32+3*32=352，读取偏移 [352..383]。
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # mask_m [BLOCK_M] 屏蔽请求末尾以外的 query 行；广播为 [BLOCK_M,1] 后覆盖其全部 D 维。
    # 加载结果 q [64,32]，无效行填 0，最后也不会写回这些无效行。
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # 每个 query 行独立维护在线 softmax 状态，不把多个 query 的概率混在一起。
    # l_i [BLOCK_M]：截至已处理 key 块的指数和；初始为 0。
    # m_i [BLOCK_M]：截至已处理 key 块的最大得分；-1e10 是大负初始化值，并非真正 -inf。
    # acc [BLOCK_M,D]：同一最大值基准下的未归一化 V 加权和，使用 float32 累积。
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # 此局部变量 num_blocks 是本请求的 key 计算 tile 数 ceil(seq_len/BLOCK_N)，
    # 不是全局缓存池的物理块数 C，也不是请求块表列数 M；本例 ceil(3/64)=1。
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    # 逐 key tile 扫描该请求所有 K/V；query tile q 在循环期间保持不变。
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # offs_n [BLOCK_N] 是本轮 key 的请求内位置；mask_n 标记小于 seq_len 的位置。
        mask_n = offs_n < seq_len
        
        # 从 K [T,Hkv,D] 中读取当前 KV 头，并在指针布局上组织成 [D,BLOCK_N]。
        # offs_d[:,None] [D,1] 是特征行，offs_n[None,:] [1,BLOCK_N] 是 key 列。
        # 因此变量 k 已按 K.T 的计算方向排列，不需要再显式执行一次 transpose。
        # B0、KV头1 的起点为 2*2*32+1*32=160，读取 [160..191]。
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        
        # k [32,64]；超出本请求 key 范围的列填 0，后面还需屏蔽其 attention 得分。
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # tl.dot 做矩阵乘法：[64,32] @ [32,64] -> qk [64,64]。
        # 每个元素表示一个 query 与一个 key 在 D 维上的点积，随后乘最终缩放 s。
        # 包装器参数 scale 已包含 1/sqrt(D)，内核不能再额外除一次 sqrt(D)。
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # 因果条件 query位置>=key位置，形状 [BLOCK_M,BLOCK_N]，只允许关注当前/过去。
        # 同一个 seq_start 加到两边不改变比较结果，B1（请求内位置1）只允许 key B0/B1。
        # B 的有效 3x3 区域为 [[允许,禁止,禁止],[允许,允许,禁止],[允许,允许,允许]]。
        # 请求边界已经由 seq_start/seq_len 隔离，因此这里不会读到 A 的 token。
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        # 无效/未来 key 的得分设为大负数，正常得分下其指数权重近似为 0。
        # 此处用的是有限值 -1e10，不是 -inf；无效 query 行最终由 mask_m 禁止写回。
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # 当前 key tile 的逐行最大值得到 m_ij [BLOCK_M]，axis=1 归约的是 key 列。
        # m_i_new 合并旧/新最大值；alpha=exp(旧最大值-新最大值) 将旧累积量换算到新基准。
        # p [BLOCK_M,BLOCK_N] 是当前 tile 的未归一化指数权重，并非最终概率。
        # 递推的完整数值例子见文件开头 scores=[0,ln(2),ln(3)] 的两块手算。
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # acc [BLOCK_M,D] 乘 alpha[:,None] [BLOCK_M,1]，每个 query 的 D 维共用其缩放系数。
        acc = acc * alpha[:, None]
        
        # V 从与 K 相同的 token/KV头读取，但指针布局为 [BLOCK_N,D]，用于 p@V。
        # K 提供匹配得分，V 提供汇总内容，两者地址布局相同而数据内容通常不同。
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # p [64,64] @ v [64,32] -> 新的未归一化输出贡献 [64,32]，累加进 acc。
        # p 转为 V 的 dtype 以进行 tl.dot，acc 保持 float32；实际浮点计算可能有舍入误差。
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # 旧指数和也需乘同一 alpha，再加当前块逐行指数和；l_i/m_i 都为 [BLOCK_M]。
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # 所有 key tile 完成后按行除指数和：[BLOCK_M,D]/[BLOCK_M,1]。
    # 此时才得到 softmax(scores)@V 的归一化结果，每个有效 query 一份 D 维向量。
    acc = acc / l_i[:, None]
    
    # 写回 O [T,Hq,D]，地址与 Q 相同；只写 mask_m 标记的有效 query 行。
    # 转换回输出 dtype，保留当前头和 token 的布局，头展平由 Attention.forward 完成。
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    prefill 包装器：根据最长请求选择网格，一次启动所有请求/本地 Q 头的 query tile。

    q [T,Hq,D]=[5,4,32]，k/v [T,Hkv,D]=[5,2,32]，输出 [T,Hq,D]=[5,4,32]。
    cu_seqlens [B+1]=[3]，值 [0,2,5] 标记 A 的区间 [0:2]、B 的区间 [2:5]。
    scale 为最终得分乘数 s；Attention.forward 传入 self.scale/sqrt(D)。
    num_heads=Hq=4、num_kv_heads=Hkv=2、head_dim=D=32，均为当前 rank 的值。
    输出没有减少 token 数或 Q 头数，GQA 只是减少输入 K/V 的头数。
    本函数将 Q/K/V 视为同一批拼接 token，不支持仅传新 token K/V 却省略历史前缀的完整注意力。
    """
    # 内核使用固定的连续寻址公式，必要时复制为连续布局；不会在这里展开/重复 GQA 的 K/V 头。
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # 输出沿用 q 的形状、设备和 dtype；empty_like 未初始化，但有效输出位置由内核写入。
    output = torch.empty_like(q)
    
    # 按头宽选择计算 tile：头越宽，使用较小 tile 以控制中间张量和 GPU 资源需求。
    # 分支是启发式选择，真实共享内存/寄存器占用取决于编译结果，并非这里精确计算的大小。
    # D=32 选择 BLOCK_M=BLOCK_N=64；注意 S=4 的物理缓存块容量与这两个 tile 尺寸无关。
    
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    # 前缀和有 B+1 项，所以请求数 B=3-1=2，不是 cu_seqlens[-1] 所表示的 token 总数 5。
    num_seqs = cu_seqlens.shape[0] - 1
    
    # 相邻前缀和之差得到每条请求长度：[2,3]，最长为 3。
    # 当前实现把边界张量搬到 CPU 再读取最大值，会产生主机读取/设备同步开销。
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # grid=[ceil(最长请求长度/BLOCK_M),Hq,B]，本例 [1,4,2]，共 8 个 program。
    # 每个 program 处理一个请求的一个 Q 头和一个 query tile，而不是一个物理缓存块。
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    # 一次 kernel launch 覆盖整个 grid；每个 program 在内核内部遍历该请求的 key tile。
    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    # 此处仍为三维 [5,4,32]；调用方 Attention 会展平为二维 [5,128]。
    return output


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    decode 分页注意力内核：一个 program 计算一条请求的一个当前 Q 头的完整输出。

    query/output [B,Hq,D]=[2,4,32]；K/V cache [C,S,Hkv,D]=[6,4,2,32]。
    block_tables [B,M]=[2,2]，值 [[2,-1],[5,1]]；context_lens [B]=[2]，值 [3,7]。
    每条请求只有一个当前 query，所有历史 K/V 通过其逻辑块表从物理缓存中读取。
    一次 key chunk 的大小 BLOCK_N=64，不要求等于 S=4，也不要求只能读一个缓存块。

    请求 B 的历史位置 t=0..6 映射为：
        t=0/1/2/3 -> 逻辑块 0 -> 物理块 5，块内偏移 0/1/2/3。
        t=4/5/6   -> 逻辑块 1 -> 物理块 1，块内偏移 0/1/2。
    每个 key 位置独立查块表，不能只用 chunk 第一个 token 的物理块寻址其余 token。

    内部形状：q/acc/output [D]，offs_n/score/p/weight/physical_block [BLOCK_N]，
    k/v/kv_offset [D,BLOCK_N]；m_i/l_i/alpha 为标量，因为本 program 只有一个 query。
    weighted V 沿 key 维归约后为 [D]，最后输出所有历史 token 汇总成的一个头向量。
    """
    # grid=[B,Hq]；batch_idx 是请求行号，不是请求内 token 位置或物理缓存块编号。
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # 每组 Hq//Hkv 个 Q 头共用一个 K/V 头；本例 Q0/Q1 -> KV0，Q2/Q3 -> KV1。
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # 读当前请求的有效上下文长度，包含本轮当前 token；请求 B 的长度为 7，对应缓存位置 0..6。
    # 该长度不等于 query.shape[0]：后者是请求数 2，当前请求历史长度却是 7。
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # 从 query [B,Hq,D] 只读取当前请求的当前 Q 头，得到 q [D]=[32]。
    # B 请求行号 1、Q头3 时起点=1*4*32+3*32=224，读取 [224..255]。
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # 这里只有一个 query，所以最大值/指数和是标量，而加权内容 acc 是 [D] 向量。
    # acc 用 float32 累积；每个 (请求,Q头) program 的状态互相独立。
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    # 请求块表最多列 M 个逻辑块，容量上限 M*S；这不是缓存池 C*S 的总容量。
    # max_chunks=ceil(M*S/BLOCK_N)，本例 ceil(2*4/64)=1。
    # 所有请求采用这个相同循环上限，短请求在下面按自身 context_len 跳过无效 chunk。
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    
    # 分块扫描当前请求的所有有效历史位置，逐块进行点积和在线 softmax。
    for chunk_idx in range(max_chunks):
        # token_start 是请求内历史位置的 chunk 起点，不是多请求拼接行号或全池 slot。
        token_start = chunk_idx * BLOCK_N
        
        # 只有起点小于当前请求长度才读这个 chunk；它不是“只读最后一个缓存块”。
        if token_start < context_len:
            # offs_n [BLOCK_N] 是 chunk 内每一 lane 负责的请求内历史位置，首 chunk 为 [0..63]。
            # B 中前 7 项有效，A 中前 3 项有效，其余 lane 在下面被屏蔽。
            offs_n = token_start + tl.arange(0, BLOCK_N)
            logical_block = offs_n // block_size
            # 每项分别计算逻辑块号 t//S 和块内偏移 t%S；二者均为 [BLOCK_N]。
            # t=6 时逻辑块=1、块内偏移=2；不能将全局请求内位置 6 直接当作块内偏移。
            offs_in_block = offs_n % block_size

            # 同时检查历史位置有效、逻辑块列不越界；这些条件用于安全加载块表。
            in_range = (offs_n < context_len) & (logical_block < max_num_blocks)

            # 块表是连续 [B,M]，条目地址为 batch_idx*M+logical_block。
            # B 的 t=6 读 block_tables[1,1]=1，说明该 token 实际位于物理块 1。
            # 同一 chunk 可从物理块 5 跳到物理块 1，每个 lane 各自查表，不能假设物理连续。
            physical_block = tl.load(
                block_tables_ptr + batch_idx * max_num_blocks + logical_block,
                mask=in_range, other=-1)
            # -1 块表填充不代表可读缓存，必须再次加入 valid 掩码。
            valid = in_range & (physical_block != -1)
            # 无效 lane 仍会参与后续地址表达式，先将其物理块替换成 0，避免由 -1 计算负地址。
            # 真正 load 仍使用 valid 屏蔽，不会把物理块 0 的无关内容纳入有效注意力。
            # 转 int64 后再算缓存偏移，适合较大的物理缓存池地址计算。
            physical_block = tl.where(valid, physical_block, 0).to(tl.int64)

            # 缓存 [C,S,Hkv,D] 的地址=physical_block*S*Hkv*D+offset_in_block*Hkv*D+kv_head*D+d。
            # physical_block[None,:]/offs_in_block[None,:] 为 [1,BLOCK_N]，offs_d[:,None] 为 [D,1]，
            # 广播后 kv_offset 为 [D,BLOCK_N]：每一列对应一个历史 token，每一行对应一个头内特征。
            # B 的 t=6、Q头3 -> KV头1 时地址=1*256+2*64+1*32+[0..31]=[416..447]。
            kv_offset = (physical_block[None, :] * (block_size * num_kv_heads * head_dim)
                         + offs_in_block[None, :] * (num_kv_heads * head_dim)
                         + kv_head_idx * head_dim
                         + offs_d[:, None])

            # 加载 k [D,BLOCK_N]，无效历史位置填 0，再将 K 转 float32 参与逐元素点积。
            # q[:,None] [D,1] 与 k [D,BLOCK_N] 相乘，沿 axis=0（D 个特征）求和，
            # 得到 score [BLOCK_N]，每项是当前 Q 与一个历史 K 的点积乘最终缩放 s。
            k = tl.load(k_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            k = tl.cast(k, tl.float32)
            score = tl.sum(q[:, None] * k, axis=0) * scale
            # 只需要屏蔽无效缓存位置；当前 query 位于 context_len-1，范围内没有未来 token，
            # 因此 decode 无需像 prefill 那样构造一个 query/key 二维因果掩码。
            qk = tl.where(valid, score, -1e10)

            # 当前 chunk 的最大值 m_ij、旧最大值 m_i、新最大值 m_i_new 都为标量。
            # p [BLOCK_N] 是以 m_i_new 为基准的未归一化指数值，alpha 为旧/新基准的转换系数。
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            # 一个 query 的整个 D 维加权和与标量指数和乘同一 alpha，保持二者的基准一致。
            acc = acc * alpha
            l_i = l_i * alpha

            # V 与 K 使用相同物理块/槽/头地址，加载 v [D,BLOCK_N]，数值来自独立的 V cache。
            # valid 为 False 的 lane 将 weight 显式置 0，避免无效 lane 贡献指数和。
            v = tl.load(v_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)
            v = tl.cast(v, tl.float32)
            weight = tl.where(valid, p, 0.0)
            # weight[None,:] [1,BLOCK_N] 广播到 [D,BLOCK_N]，沿 axis=1（历史 key 位置）求和，
            # 得到本 chunk 的加权内容 [D]，与 acc 累加；l_i 累加本 chunk 有效权重的标量和。
            acc = acc + tl.sum(weight[None, :] * v, axis=1)
            l_i = l_i + tl.sum(weight)

            m_i = m_i_new
    
    # 遍历完成后用标量指数和归一化 [D] 向量；至少应有一个有效缓存 token，才能避免 l_i=0。
    # 文件开头的手算例子：acc 前两维 [4/3,5/3]、l_i=2 -> output 前两维 [2/3,5/6]。
    output = acc / l_i
    
    # 按请求/Q头/特征写回 output [B,Hq,D]，每个 program 写一个 [D] 头向量。
    # 所有 program 完成后输出 [2,4,32]，Attention 再将其展平为 [2,128]。
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    decode 包装器：每条请求一个当前 Q，通过块表读取该请求的所有有效历史 K/V。

    query [B,Hq,D]=[2,4,32]：请求数、本地 Q 头数、头内特征；这里 B 也等于本轮新 token 数。
    k_cache/v_cache [C,S,Hkv,D]=[6,4,2,32]：物理块数、块容量、本地 KV 头数、头内特征。
    block_tables [B,M]=[2,2]：每条请求的逻辑块到物理块映射，例 [[2,-1],[5,1]]。
    context_lens [B]=[2]：有效上下文长度，值 [3,7]，不是两个请求都只有 2 个历史 token。
    scale=s 已包含 1/sqrt(D)，num_heads=Hq、num_kv_heads=Hkv、head_dim=D、block_size=S。
    返回 output [B,Hq,D]=[2,4,32]；不返回历史每个 token 的输出，只返回当前 query 的结果。
    """
    # B 是本轮请求数，M 是每条请求块表经填充后的最大列数，不是物理池的块数 C。
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    # query 内核寻址按连续 [B,Hq,D] 计算，必要时复制；缓存与块表也应由调用方提供连续布局。
    query = query.contiguous()
    
    # 每个当前 Q 头输出一个相同宽度 D 的向量，沿用 query 的形状、设备和 dtype。
    output = torch.empty_like(query)
    
    # 每个 chunk 最多扫描 BLOCK_N 个历史 token；D=32 时为 64，可跨越多个 S=4 的缓存块。
    BLOCK_N = 64 if head_dim <= 128 else 32
    
    # grid=[B,Hq]=[2,4]，共 8 个 program，每个 program 在内部遍历整个请求历史。
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    
    # 内核返回三维 [2,4,32]；头维度在 Attention.forward 中再展平成 [2,128]。
    return output


class Attention(nn.Module):
    """
    对当前 rank 的 Q/K/V 执行注意力，管理缓存写入与 prefill/decode 路径选择。

    输入已由外部线性层投影、按头 reshape，并由调用方完成需要的 Q/K Norm 与 RoPE。
    本类没有注意力投影的可学习参数；其 k_cache/v_cache 存放历史 token 的内容。
    prefill 输入 Q [T,Hq,D]、K/V [T,Hkv,D]，返回 [T,Hq*D]，例 [5,128]。
    decode 输入 Q [B,Hq,D]、本轮 K/V [B,Hkv,D]，返回 [B,Hq*D]，例 [2,128]。
    最后的头展平只是 reshape，不会减半维度、跨 rank 收集或执行输出投影。
    Qwen3Attention 随后用 RowParallelLinear 将本地头结果投影并归约到完整 hidden_size。

    Context 由运行器 set_context 提供，本类通过 get_context 读取，不从张量形状自动推断阶段：
        is_prefill=True：需要 cu_seqlens_q [B+1]，本例 [0,2,5]，用于请求边界与因果注意力。
        is_prefill=False：需要 block_tables [B,M] 和 context_lens [B]，用于读取历史缓存。
        slot_mapping：本轮新 K/V 写入位置，prefill 长度 T、decode 长度 B。
    序列中 token 的位置编码已由调用方处理，这些缓存地址/请求边界不是 RoPE 的 positions。
    """

    def __init__(
        self,
        num_heads: int,  # 本地 Q 头数 Hq，统一例子为 4，不是完整模型未分片的全局头数。
        head_dim: int,  # 单个头的特征宽度 D，例 32，不是 Hq*D=128 的展平宽度。
        scale: float = 1.0,  # 额外得分乘数，forward 中还会除以 sqrt(D)。
        num_kv_heads: int = None,  # 本地 K/V 头数 Hkv，例 2；None 时与 Q 头数相同。
        block_size: int = 16,  # 物理缓存块容量 S，统一例子显式传 4；不是计算 tile 的 BLOCK_N。
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        # 默认 Hkv=Hq 是普通 MHA，较少 KV 头是 GQA；当前内核要求 Hkv>0 且 Hq%Hkv==0。
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        # 空张量仅表示尚未分配缓存，不是实际 [C,S,Hkv,D] 的缓存池。
        # 两个属性最初引用同一个空占位张量，运行器之后会分别赋予独立 K/V 缓存切片。
        # 它们是普通属性，未注册为 Parameter/buffer；本类的 .cuda() 不负责分配或搬运它们。
        # ModelRunner 直接在相应 GPU 上分配缓存，然后赋到每层 Attention 的这两个属性。
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # 主路径 q [T,Hq,D]、k/v [T,Hkv,D]；decode 时 T=B，因为每条请求一个当前 token。
        # prefill 的 T=5 是所有请求本轮 token 总数，decode 的 T=2 是两条请求各一个新 token。
        context = get_context()
        # 读取当前层的本地缓存引用，每层/每个 rank 各有自己的 K/V 内容。
        k_cache, v_cache = self.k_cache, self.v_cache

        # 缓存存在且提供 slot_mapping 时，先写本轮 K/V；decode 后续读取即可包含当前 token。
        # 若缓存仍为空（例如分配前的 prefill 预热），跳过写入，但不自动禁止后面的 decode 读取。
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            if k.dim() == 4:
                # 仅为缓存写入将四维 K/V [B,N,Hkv,D] 展平为 [B*N,Hkv,D]。
                # slot_mapping 必须与这个 B*N 行顺序对应；例如请求0的 N 行在请求1之前。
                # 注意重新生成的 k_to_store/v_to_store 没有替换原始 q/k/v，
                # 后面的注意力内核仍要求三维输入，因此此分支不代表完整四维批处理支持。
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # 主路径已经是三维 [T,Hkv,D]，只保证连续布局，token 顺序和头数不改变。
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            
            # prefill 例 K/V [5,2,32] 写 slot [8,9,20,21,22]；decode 例 [2,2,32] 写 [10,6]。
            # 不写 Q：历史查询不需要重新执行，历史 K/V 才是未来 query 会读取的内容。
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        # 最终得分缩放 s=self.scale/sqrt(D)；D=32、self.scale=1 时约为 0.176777。
        # 两个底层内核都直接使用这个最终值，不在内部重复除以 sqrt(D)。
        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # prefill 一条请求可以有多个当前 query；按前缀和将拼接 token 分成独立序列。
            # 本例 cu_seqlens_q=[0,2,5]，分别计算 A0/A1 与 B0/B1/B2 的因果注意力。
            cu_seqlens = context.cu_seqlens_q
            # 仅有 q.shape[0]=5 无法推断这 5 行属于几条请求，缺少边界时直接报错。
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            
            # 此分支直接读取本轮 k/v；即便 Context 有 cu_seqlens_k 或 block_tables，
            # 当前函数也没有用它们补读已缓存前缀，因此不能据此认为支持完整的前缀缓存 prefill。
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # o [T,Hq,D]=[5,4,32] -> [T,Hq*D]=[5,128]：把同一 token 的四个头按顺序展开。
            # 保持所有 5 个 token，不跨 token 相加，也不在此处对不同 rank 的头做归约。
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            # decode 一条请求一个当前 query，本例 q [2,4,32]；历史长度可以为 3/7。
            # 缓存中历史 K/V 不再经过本轮投影，通过各请求的块表直接读取。
            # 调用方需提供已分配且内容有效的缓存、块表和正的上下文长度，本层不会自动创建。
            o = paged_attention_decode(
                q, 
                k_cache, 
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # o [B,Hq,D]=[2,4,32] -> [B,Hq*D]=[2,128]，每条请求返回当前 token 的本地头结果。
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # 以下保留原有旧示例的执行逻辑，仅注释，不将它作为当前接口已验证可直接运行的测试。
    # 旧示例的 Q/K/V 将头维展平为 [B,N,Hq*D]，缓存也是三维，且没有调用 set_context；
    # 与上面的三维 Q/K/V [T,Hq,D]、四维缓存 [C,S,Hkv,D] 和阶段上下文约定不匹配。
    # 仅创建 slot_mapping 局部变量不会让 get_context() 自动获得它。
    #
    # 与统一例子一致的 prefill 调用方式可写为（下面仅为注释示例，需 CUDA/Triton 环境）：
    #   from myvllm.utils.context import set_context
    #   device = 'cuda'
    #   layer = Attention(num_heads=4, head_dim=32, num_kv_heads=2, block_size=4)
    #   q = torch.randn(5, 4, 32, device=device, dtype=torch.float16)
    #   k = torch.randn(5, 2, 32, device=device, dtype=torch.float16)
    #   v = torch.randn(5, 2, 32, device=device, dtype=torch.float16)
    #   layer.k_cache = torch.zeros(6, 4, 2, 32, device=device, dtype=torch.float16)
    #   layer.v_cache = torch.zeros_like(layer.k_cache)
    #   set_context(
    #       is_prefill=True,
    #       cu_seqlens_q=torch.tensor([0, 2, 5], device=device, dtype=torch.int32),
    #       slot_mapping=torch.tensor([8, 9, 20, 21, 22], device=device, dtype=torch.long),
    #   )
    #   output = layer(q, k, v)  # [5,128]，同时将本轮 K/V 写到对应物理缓存槽。
    #
    # 旧示例里的参数：Q 头数 8，每头宽度 64；未传 num_kv_heads，所以 K/V 头数也为 8。
    layer = Attention(num_heads=8, head_dim=64).cuda()
    # 此处 D=512 是 8*64 的展平宽度，不同于文件其余注释中 D 所指的每头宽度。
    # B=4 表示请求数，N=1024 表示每请求 token 数；正确展平后 T 应为 4096。
    B, N, D = 4, 1024, 512
    q = torch.randn(B, N, D).cuda()
    k = torch.randn(B, N, D).cuda()
    v = torch.randn(B, N, D).cuda()
    # 旧缓存 [4,1024,512] 未拆为 [物理块,块内槽,KV头,头内维]，也没有对应请求块表。
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()
    # 这里只创建长度 1024 的张量，未写入 Context，也未与 4096 个真实 token 行逐一对应。
    slot_mapping = torch.arange(N).cuda()

    # 若先适配正确输入与上下文，预热可消化首次内核编译等开销；旧逻辑在适配前可能已报错。
    for _ in range(10):  # Warm-up iterations
        _ = layer(q, k, v)

    import time
    # 下方计时方式仅在上述调用已正确准备后有意义；CUDA kernel 默认异步执行。
    times = []
    for _ in range(100):  # Timing iterations
        # 等待之前 GPU 工作完成，再记录本次开始时间，避免将排队任务混入计时。
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        # 等待本次 GPU 工作完成再读结束时间，避免只测到 Python 提交 kernel 的时间。
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    # 平均 100 次耗时，time.time 单位为秒，最后乘 1000 输出毫秒。
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
