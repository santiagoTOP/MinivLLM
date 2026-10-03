import torch.nn as nn
import torch 

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    将查表得到的 cos/sin 应用到一个 Q 或 K 张量，执行实际的二维旋转。

    本实现采用前后半维度配对：d=4 时，x=[a,b,c,d] 拆为 x1=[a,b]、x2=[c,d]，
    旋转的是 (a,c)、(b,d) 两对，而不是相邻的 (a,b)、(c,d)。
    每对的旋转公式为 (u,v) -> (u*cos-v*sin, u*sin+v*cos)，角度取决于位置和该对的频率。
    旋转之后按 [所有 out1 | 所有 out2] 恢复原布局，最后一维不会减半。

    下面的数值例子使用 RotaryEmbedding 的普通频率：base=100、d=4、position=1。
    两对的角度为 [1,0.1] 弧度；对向量 [1,2,3,4]：
        out1=[1*cos(1)-3*sin(1), 2*cos(0.1)-4*sin(0.1)]。
        out2=[1*sin(1)+3*cos(1), 2*sin(0.1)+4*cos(0.1)]。
        拼回后的结果约为 [-1.9841,1.5907,2.4624,4.1797]，仍为 4 维。
    """
    if x.dim() == 3:
        # 主推理路径：x [T,h,d]，T 为本轮 token 数，h 为当前 Q 或 K 的本地头数。
        total_tokens, num_heads, head_dim = x.shape
        # 查表后的 cos/sin 都是 [T,d/2]，每个 token 有自己的一组角度。
        # 插入头维度得到 [T,1,d/2]，在 h 个头之间广播，不把不同 token 的角度混在一起。
        # 例：Q [5,2,2] 的 cos [5,1] -> [5,1,1]，同一 token 的两个 Q 头共用旋转角。
        # K 可为 [5,1,2]，使用同一组位置角度，不要求 Q/K 头数相等，也不需要复制 K 头。
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 沿每个头的特征维度拆为前后两半，各为 [T,h,d/2]；chunk 返回视图。
        # d=4 时，最后一维 [1,2,3,4] -> x1=[1,2]、x2=[3,4]。
        x1, x2 = x.chunk(2, dim=-1)

        # 每个位置/头/维度对独立旋转，cos/sin 在头维度上广播。
        # 两个乘法都是逐元素运算，不是 Q@K.T 注意力得分计算。
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # 拼回 [T,h,d]，保持原始头顺序和维度顺序；不原地覆盖输入张量。
        # 与 SiluAndMul 不同，这里拆半后再拼接，因此输出宽度与输入相同。
        return torch.cat([out1, out2], dim=-1)
    else:
        # 此分支按四维 x [B,N,h,d] 处理，B 是 batch，N 是每条序列的长度。
        # 代码没有显式校验维数，调用方应提供该分支所约定的四维张量。
        B = x.size(0)
        seq_len = x.size(1)
        num_heads = x.size(2)
        head_dim = x.size(-1)

        # 当前四维分支约定 positions [N]，因此 cos/sin 为 [N,d/2]。
        # 扩展成 [1,N,1,d/2]，同时在 batch 和头维度上广播。
        # 这表示所有 batch 共用这组 N 个位置，不支持直接传 [B,N] 的不同 batch 位置表。
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # 前后两半各为 [B,N,h,d/2]，配对方式与三维主路径一致。
        x1, x2 = x.chunk(2, dim=-1)

        # 每个 batch 中相同位置、不同头，使用相同频率的 cos/sin 旋转对应维度对。
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # 恢复 [B,N,h,d]；这个辅助函数处理四维张量，并不表示后续注意力内核也支持四维输入。
        return torch.cat([out1, out2], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    RoPE（旋转位置编码）：根据 token 的位置旋转 Q/K 向量中的二维特征对。

    它不把位置向量加到隐藏向量上，而是对投影后的 Q/K 执行旋转；调用方的 V 不传入本层。
    本层负责预计算频率与 cos/sin 位置表，实际旋转由 apply_rotary_pos_emb 完成。
    每个维度对有一个角频率 omega_j，位置 p 的旋转角为 theta[p,j]=p*omega_j。
    在精确算术下，二维旋转保留该对的长度，但会改变向量方向以及不同位置之间的点积。
    若写成列向量 Q'_m=R(m)Q_m、K'_n=R(n)K_n，则点积为：
        Q'_m.T @ K'_n = Q_m.T @ R(n-m) @ K_n。
    所以注意力得分中出现相对位置 n-m；得分仍取决于 Q/K 内容，不是只由距离决定。

    主路径形状：positions [T]、query [T,Hq,d]、key [T,Hkv,d]。
    返回 query_rotated/key_rotated，各自形状不变；GQA 的 Hq/Hkv 可以不同。
    本实现旋转整个头，正确使用要求 rotary_embedding 等于 Q/K 最后一维 d，且 d 为正偶数。
    当前没有实现“只旋转头的前几维、剩余维度保持原值”的部分旋转逻辑。

    普通 RoPE 数值例子：base=100、rotary_embedding=4、max_position=6、is_llama3=False。
        两个维度对的编号 j 为 0/1，角频率为 [1,0.1]，不是 [1,1]。
        位置表的角度行为：p=0 -> [0,0]，p=1 -> [1,0.1]，p=2 -> [2,0.2]。
        cache[p]=[cos(p),cos(0.1*p),sin(p),sin(0.1*p)]，每行有 4 个数。
        对 [1,2,3,4]，位置 0 的输出仍为 [1,2,3,4]；位置 1 的输出约为
        [-1.9841,1.5907,2.4624,4.1797]，详细配对计算见上面的辅助函数。

    与之前 Qwen3Attention 的例子对应：d=2 时只有一个维度对，普通角频率为 [1]。
    positions=[0,1,0,1,2] 与 A0/A1/B0/B1/B2 对齐，查表不会把 B0 的位置误当成 2。
    输入 Q [5,2,2]、K [5,1,2]，查得 cos/sin 各 [5,1]，分别广播到 Q/K 的本地头上。
    输出仍为 Q [5,2,2]、K [5,1,2]；本层不跨 rank 通信，也不决定请求间的注意力边界。

    cos_sin_cache 是按位置预计算的常量表，不是保存历史 token 内容的 KV cache。
    解码时只需用当前 token 的绝对位置查表；已经旋转过的历史 K 由 Attention 的 KV cache 保存。
    """

    def __init__(
        self, 
        base:int,  # 普通频率公式的底数；常用值大于 1，较后面的维度对旋转得更慢。
        rotary_embedding: int,  # 要旋转的维度数 d，本实现要求与实际头宽度相同，且为正偶数。
        max_position: int = 2048,  # 缓存长度 L，预计算合法非负位置 0 到 L-1，不会动态扩容。
        is_llama3: bool = False,  # 是否在普通频率基础上执行下面的 Llama 3 频率缩放规则。
        # 以下四个参数仅在 is_llama3=True 时用于频率缩放。
        llama3_rope_factor: float = 32.0,  # 低频部分的缩放倍数；除以 32 表示角频率变为原来的 1/32。
        llama3_rope_high_freq_factor: float = 4.0,  # 用 L0/该值划分短波长（高频）区域。
        llama3_rope_low_freq_factor: float = 1.0,  # 用 L0/该值划分长波长（低频）区域。
        llama3_rope_original_max_position_embeddings: int = 8192,  # 原始参考长度 L0，用于频率分区。
    ):
        # 初始化 Module 后才能注册后面的 cos_sin_cache buffer。
        super().__init__()
        self.base = base
        # 保存头的旋转宽度，频率数量只有 d/2，因为每个频率对应一对特征。
        self.rotary_embedding = rotary_embedding
        # max_position 决定实际查表上限；Llama 的频率缩放不会自动把此长度乘以 factor。
        self.max_position = max_position
        # omega_j = 1 / base**(2*j/d)，j=0,...,d/2-1；变量名 inv_freq 表示倒数幂频率。
        # arange(0,d,2) 生成 [0,2,...,d-2]，除以 d 后得到各维度对对应的指数。
        # base=100、d=4 时：[0,2]/4=[0,0.5] -> base 幂 [1,10] -> inv_freq=[1,0.1]。
        # omega 的单位可理解为弧度/token；位置每增加 1，第一对转 1 弧度，第二对转 0.1 弧度。
        # self.inv_freq 是普通张量属性，不是可学习 Parameter，也没有注册成 buffer。
        # 它只用于初始化下面的位置表；forward 使用的可迁移设备的 buffer 是 cos_sin_cache。
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))

        if is_llama3:
            # 在普通频率基础上，按波长区分高频、低频和过渡区，部分降低旋转频率。
            # 本项目 LlamaAttn 调用时启用此分支；Qwen3Attention 默认不启用。
            import math
            inv_freq = self.inv_freq
            # 完成一次 2*pi 旋转所需的 token 距离，即波长 lambda=2*pi/omega。
            # 频率越高，波长越短；频率越低，波长越长。
            # 默认 L0=8192、high=4、low=1，对应短波长阈值 2048、长波长阈值 8192。
            wave_len = 2 * math.pi / inv_freq
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                # 两个阈值相同时没有过渡区，使用一次条件选择，避免后面 delta=0 的除法。
                # 比阈值短的波长保留原频率，其余频率除以 factor；torch.where 按元素选择。
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                # 正常使用时 high>low>0，delta 是两个频率分区参数的差。
                # smooth=(L0/lambda-low)/(high-low)，再夹到 [0,1]：
                #   lambda<=L0/high -> smooth=1，保留短波长/高频部分。
                #   lambda>=L0/low  -> smooth=0，长波长/低频部分除以 factor。
                #   两阈值之间 -> smooth 在 0/1 之间，使频率缩放平滑过渡。
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                # 最终乘数是 smooth*1+(1-smooth)/factor，并非对所有频率统一除以 factor。
                # 默认参数下，lambda=4096 时 smooth=(8192/4096-1)/(4-1)=1/3，
                # 乘数=1/3+(2/3)/32=17/48≈0.354167。
                # 因此新频率是原频率的约 0.354167 倍，旋转变慢，波长相应变长。
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            # 更新为缩放后的频率，后面将按此频率构造所有位置的 cos/sin 表。
            self.inv_freq = inv_freq

        # 创建所有位置索引并转为浮点以计算角度；L=6 时得到 [0.,1.,2.,3.,4.,5.]。
        positions = torch.arange(self.max_position).float()
        # einsum("i,j -> ij") 是外积：freqs[p,j]=positions[p]*inv_freq[j]。
        # 结果 [L,d/2]；这里变量名 freqs 实际保存“各位置的角度”，不是新的角频率。
        # 本例 [6] 与 [2] 外积得到 [6,2]，前三行为 [[0,0],[1,0.1],[2,0.2]]。
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)

        # 对每个位置/维度对的角度取 cos/sin，两个张量都为 [L,d/2]。
        # 位置 0 的 cos 全为 1、sin 全为 0，所以该位置的旋转保持输入不变。
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # 沿最后一维拼成 [cos_all_pairs | sin_all_pairs]，形状 [L,d]。
        # d=4 的位置 1 行约为 [0.540302,0.995004,0.841471,0.099833]。
        # 这里缓存的是 cos/sin，布局不同于辅助函数所处理的向量前后两半。
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        # 注册为非参数 buffer：不参与学习，但随 Module.to()/cuda() 移动，默认进入 state_dict。
        # 一次预计算之后，forward 只查所需行，避免每步重复计算所有位置的三角函数。
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    # 编译前向张量操作，编译器可能优化查表/逐元素旋转；首次调用通常有编译开销。
    @torch.compile
    def forward(self, positions, query, key):
        """
        根据 token 位置查 cos/sin 表，再分别旋转 Q/K，返回两个新张量。

        主路径：positions [T]，query [T,Hq,d]，key [T,Hkv,d]。
        positions 应为有效整数位置索引，满足 0<=position<max_position，通常使用 torch.long。
        四维路径：query/key [B,N,heads,d]，positions [N]，各 batch 共用这组位置。
        输入头宽度必须与初始化的 rotary_embedding 一致，设备也应与相关张量匹配。
        """
        # 主路径从 [L,d] 取 T 行，得到 [T,d]；positions 可以不连续，也可以重复。
        # 例如 [0,1,0,1,2] -> [cache[0],cache[1],cache[0],cache[1],cache[2]]。
        # 同位置的不同请求查相同角度，这是正常的；请求隔离由 Attention 的上下文负责。
        # 若缓存长度至少为 7，decode 两条请求长度为 [3,7] 时查位置 [2,6]，不是本轮行号 [0,1]。
        # 当前方法没有额外的合法位置校验或缓存扩容，调用方应保证索引处于有效非负范围。
        cos_sin = self.cos_sin_cache[positions]  # 主路径 [T,d]；四维路径查表得到 [N,d]。
        # 按前后半段取出 cos 与 sin，各为 [T,d/2]；不是把每头的旋转维度永久减半。
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 使用同一组位置角度分别旋转 Q/K，辅助函数会按各自头数广播；V 不在此函数参数中。
        # 返回顺序为 (旋转后的 Q,旋转后的 K)，调用方可写 q,k=rotary_emb(positions,q,k)。
        # 本层不保存 token 的 Q/K 内容；需要缓存的历史 K 由后续 Attention 单独写入 KV cache。
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


if __name__ == "__main__":
    base = 5
    # how many dimensions to apply rotary embedding
    rotary_dim = 16
    # maximum position that the long context can reach
    max_position = 100
    print(torch.arange(0, rotary_dim, 2))
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    print(inv_freq)

    t = torch.arange(max_position).float()

    freqs = torch.einsum("i,j -> ij", t, inv_freq)

    print(freqs.size())

    print(freqs[2])
