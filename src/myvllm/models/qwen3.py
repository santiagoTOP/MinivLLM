from myvllm.layers import *
import torch 
import torch.nn as nn

class Qwen3Attention(nn.Module):
    """
    自注意力子层：隐藏向量 -> 合并 QKV 投影 -> Q/K 归一化 -> RoPE -> 注意力 -> 输出投影。

    Q（Query）表示当前 token 用于查询的向量，K（Key）用于匹配查询，V（Value）提供被汇总的信息。
    对某个 Q 头，注意力的概念公式为：
        scores = (scale / sqrt(head_dim)) * Q @ K.T + causal_mask
        head_output = softmax(scores, dim=key_token_dim) @ V
    causal_mask 屏蔽未来 token；不同请求的注意力也必须隔离，不能跨请求互相读取。
    实际计算由 Attention 的分块内核完成，不在本类中显式构造完整 scores 矩阵。
    当 Q 头数大于 KV 头数时采用 GQA：一组 Q 头共用同一 K/V 头，减少 K/V 投影和缓存量。

    张量并行沿头/投影输出特征分配工作，不切分本轮待计算的 token：
        每个 rank 收到相同完整隐藏向量，只计算自己负责的 Q/K/V 头。
        注意力在本地头上计算；o_proj 用行并行计算输出贡献，再求和得到完整隐藏向量。
    本类不加残差，也不做进入注意力前的 hidden_size 维归一化，这些由 DecoderLayer 处理。

    以下用同一组配置逐步解释形状：
        hidden_size=8：每个 token 的隐藏向量长度为 8。
        总 Q 头数=4、总 KV 头数=2：Q 有 4 个头，K 和 V 各有 2 个头。
        head_dim=2：每个 Q/K/V 头的向量长度都是 2。
        tp_size=2：使用两个并行进程，通常分别对应两张 GPU。
    这里需要区分“token 数”和“头数”：两个 rank 都处理全部 token，只负责不同的头。

    1. prefill 输入为什么是 [5,8]？
        请求 A 有 A0/A1 两个 token，请求 B 有 B0/B1/B2 三个 token。
        拼接后的行顺序是 [A0,A1,B0,B1,B2]，T=2+3=5。
        x=[h_A0,h_A1,h_B0,h_B1,h_B2]，其中每个 h 都是一个 8 维隐藏向量。
        因此 x 的形状 [5,8] 表示 [token 数,每个 token 的隐藏宽度]；一行不是一条请求。

    2. 完整 QKV 的宽度与本地 QKV 的宽度有什么区别？
        完整 Q：4 个头*2 维=8 维；完整 K：2*2=4 维；完整 V：2*2=4 维。
        完整合并投影宽度=8+4+4=16，所以未分片时 x [5,8] -> qkv [5,16]。
        分给两个 rank 后，各自负责的头如下，编号均表示全局头编号：
            rank 0：Q0/Q1、K0、V0。
            rank 1：Q2/Q3、K1、V1。
        每个 rank 的 Q 宽度=2*2=4，K/V 宽度各为 1*2=2，本地合计 4+2+2=8。
        所以每个 rank 都从完整 x [5,8] 得到自己的 qkv [5,8]，两份 qkv 内容不同。
        输入与本地 qkv 恰好同宽只是此配置的巧合，不能理解为投影没有改变数据。

    3. split/view 分别改变什么？
        split([4,2,2]) 把本地 qkv [5,8] 拆为 q [5,4]、k [5,2]、v [5,2]。
        view 将最后一维中挤在一起的多个头展开：
            q [5,4] -> [5,2,2]，三个数字依次是 token 数、Q 头数、每头宽度。
            k/v 各 [5,2] -> [5,1,2]，依次是 token 数、KV 头数、每头宽度。
        数据量和 token 顺序不变；Q/K RMSNorm、RoPE 同样不改变这些形状。

    4. 共用 KV 为什么仍然有两个头的输出？
        rank 0 的 Q0 使用 K0 匹配、V0 汇总；Q1 也使用 K0 匹配、V0 汇总。
        Q0/Q1 的查询向量不同，注意力权重和结果通常也不同；共用 KV 不会合并 Q 头。
        本地两个 Q 头各输出 2 维，得到 [5,2,2]，Attention 返回前展平为 [5,4]。
        rank 1 同样返回 [5,4]，但对应 Q2/Q3 的结果。
        o_proj 将两份本地头结果分别映射为 [5,8] 的输出贡献，再通过 all_reduce 求和。
        最后每个 rank 都持有完整输出 [5,8]，其隐藏宽度与输入一致。

    5. positions 与 cu_seqlens_q 分别表示什么？
        拼接行号    token    所属请求内的位置
            0        A0          0
            1        A1          1
            2        B0          0
            3        B1          1
            4        B2          2
        positions=[0,1,0,1,2] 给 RoPE 使用；B0 是另一条请求的开头，所以从位置 0 重新计数。
        cu_seqlens_q=[0,2,5] 是长度前缀和，标记请求 A 的范围 x[0:2]、请求 B 的范围 x[2:5]。
        底层注意力结合请求边界和因果掩码，使 A1 只关注 A0/A1，B2 只关注 B0/B1/B2。
        物理上拼接 token 方便批量计算，不会让两个请求互相读取信息。

    6. decode 为什么只有 [2,8] 输入，也能关注很长的历史？
        若两条请求本轮各处理一个新 token，输入仅包含它们的当前隐藏向量，所以 T=2。
        两条请求当前长度若为 [3,7]，当前 token 的位置为 [2,6]，最终输出仍为 [2,8]。
        当前 token 产生新的 Q/K/V，K/V 写入缓存；当前 Q 再查询当前及历史 token 的 K/V。
        历史长度决定要读多少缓存，不决定本轮要重新投影多少 token。
        因此不需要把全部历史 token 的隐藏向量再次传入本层；历史 K/V 由缓存提供。
    """

    def __init__(
        self,
        hidden_size: int,  # 每个 token 的完整隐藏宽度 H，也是本层最终输出宽度。
        num_heads: int,  # 完整模型的 Q 头数，不是当前 rank 的本地头数。
        head_dim: int,  # 每个 Q/K/V 头的特征宽度 d；当前 RoPE 要求偶数维度。
        scale: float = 1.0,  # 注意力得分的额外乘数，底层还会除以 sqrt(d)。
        num_kv_heads: int | None = None,  # 完整 K/V 头数；None 表示与 Q 头数相同。
        rms_norm_epsilon: float = 1e-5,  # 当前实现接收此参数，但创建 Q/K Norm 时尚未使用。
        qkv_bias: bool = False,  # QKV 投影是否加偏置；本实现还以它决定是否执行 Q/K Norm。
        base: int = 10000,  # RoPE 频率的底数，决定不同旋转维度对的频率。
        max_position: int = 16384,  # RoPE 预计算位置表长度，合法非负位置索引应小于此值。
        block_size: int = 256,  # 每个物理 KV cache 块可保存的 token 数，不是注意力头维度。
    ):
        # 注册投影、归一化、RoPE 和 Attention 等子模块前，先初始化 nn.Module。
        super().__init__()
        # 使用默认分布式进程组；构造前需要初始化，通常本项目一个进程使用一张 GPU。
        self.tp_size = dist.get_world_size()

        # 同时保留全局 Q 头数和本地 Q 头数；示例分别为 4 和 4//2=2。
        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size

        # K 和 V 使用相同头数；None 时退化为 Q/K/V 头数相同的多头注意力（MHA）。
        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        # 示例全局 KV 头数为 2，每个 rank 负责 2//2=1 个 KV 头。
        # 正确配置要求 Q/KV 头数都能整除 tp_size，且本地 Q 头数能整除本地 KV 头数。
        # 当前这里使用整数除法，没有显式检查这些条件，也不支持 KV 头少于进程数时的复制策略。
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        # 理论上 d 可由 H/总 Q 头数推导；但本构造函数后面仍直接使用传入的 head_dim，
        # 所以这里的回退赋值并不能让 head_dim=None 完整可用，实际应传入有效正整数。
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        # Attention 内部的最终缩放系数为 scale/sqrt(d)，scale=1 才是标准的 1/sqrt(d)。
        self.scale = scale

        # 将三个具有相同输入宽度 H 的投影合成一次线性运算。
        # 示例完整 Q/K/V 投影权重分别为 [8,8]、[4,8]、[4,8]；
        # 各 rank 只保存 Q 的 4 行以及 K/V 各 2 行，共 8 行，每行仍读取全部 8 个隐藏特征。
        # 完整权重形状为 [(总 Q 头数+2*总 KV 头数)*d,H]；本地形状为 [q_size+2*kv_size,H]。
        # 示例完整权重 [16,8]，每个 rank 的本地权重 [8,8]。
        # 本地参数/输出顺序必须是 [Q_local | K_local | V_local]，各投影分别按头分片加载。
        # 对应 QKVColumnParallelLinear.weight_loader 的投影 ID 为 'q'、'k'、'v'。
        # 层提供分片加载方法，但外部 checkpoint 加载器仍需显式调用；直接复制完整权重不会自动分片。
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,  # 投影读取完整隐藏向量，不要求 H 一定等于总 Q 头数*d。
            head_size=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 记录单个 rank 的展平特征宽度，而不是全局宽度；示例 q_size=4、kv_size=2。
        # 后续 split 依据这两个宽度分离合并输出。
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        # 本项目虽然将类命名为 LayerNorm，实际计算的是 RMSNorm：
        #   norm(t) = t / sqrt(mean(t**2, dim=-1)+eps) * gamma
        # 这里最后一维是 d，所以每个 token 的每个头独立计算 RMS，不混合 token 或头。
        # Q/K 各有一份可学习 gamma [d]，在其所有 token 和头之间共享，初始值为全 1。
        # 归一化调整 Q/K 的尺度，影响点积得分，但不保证数值绝对有界或输出均值为 0。
        # 当前未传 eps=rms_norm_epsilon，实际使用 LayerNorm 的默认 eps=1e-5。
        # 两个模块始终创建，是否调用则由 forward 中的 qkv_bias 条件决定；V 不做此归一化。
        self.q_norm = LayerNorm(torch.ones(head_dim))
        self.k_norm = LayerNorm(torch.ones(head_dim))

        # RoPE 在每个头的全部 d 维上旋转 Q/K，将 token 位置编码进匹配关系，不改变形状。
        # RotaryEmbedding 预先生成 [max_position,d] 的 cos/sin buffer，运行时按 positions 索引。
        # 本实现将每个头分成前后两半配对旋转：
        #   out1=t1*cos-t2*sin，out2=t1*sin+t2*cos；不同头共享同一位置的旋转角。
        # base 控制 inv_freq，旋转角由 position*inv_freq 得到；位置 0 时 cos=1、sin=0。
        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position
        )

        # 底层 Attention 使用本地头数进行 GQA，不需要收集其他 rank 的 Q/K/V。
        # 内核按 q_head // (本地 Q 头数/本地 KV 头数) 选择对应 KV 头。
        # 示例 rank 0 的全局 Q 头 0/1 共用 KV 头 0，rank 1 的 Q 头 2/3 共用 KV 头 1。
        # “共用 KV”指各 Q 头复用同一组 K 向量和同一组 V 向量，不表示 K 和 V 数值相同。
        # 它通过 get_context() 获取 prefill/decode 阶段、请求边界、缓存槽位和块表等信息。
        # KV cache 由 ModelRunner 分配并赋给该模块，每层每个 rank 分别保存自己的 KV 头。
        # 单份 K 或 V cache 的形状为 [num_blocks,block_size,本地 KV 头数,d]。
        self.attention = Attention(
            self.num_heads,
            head_dim,
            scale,
            self.num_kv_heads,
            block_size
        )

        # 多头注意力的全局输出宽度为 总 Q 头数*d；各 rank 只持有其中 本地 Q 头数*d。
        # RowParallelLinear 沿输入特征切分权重，直接消费本地头的展平结果，无需先 all_gather。
        # 本地投影贡献均为 [...,H]，再 all_reduce(SUM) 得到各 rank 相同的完整输出。
        # 示例完整 o_proj 权重 [8,8]，本地权重 [8,4]；bias=False 避免归约前偏置重复累加。
        self.o_proj = RowParallelLinear(
            input_size=head_dim * self.total_num_heads,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self, 
        x: torch.Tensor,  # 主推理路径为 [T,H]，T 是本轮各请求待处理 token 的总数。
        positions: torch.Tensor,  # 主路径为整数位置索引 [T]，与 x 的每一行一一对应。
    ) -> torch.Tensor:
        # 各 rank 接收相同 token 顺序和完整隐藏特征；投影负责输出特征分片，本层不广播输入。
        # 示例 x 的 5 行依次对应 A0/A1/B0/B1/B2，每行有 8 个隐藏特征。
        # 两个 rank 都处理这 5 行，不会把请求 A 分给 rank 0、请求 B 分给 rank 1。
        # 主路径 x=[T,H]，示例 [5,8]；得到本地 qkv=[T,q_size+2*kv_size]，示例 [5,8]。
        # 注意示例的输入/合并输出恰好等宽只是配置巧合，二者通常可以不同。
        qkv = self.qkv_projection(x)

        # 按实际本地投影宽度拆分，不是平均三等分：GQA 下 Q 宽度可能大于 K/V。
        # 示例 [5,8] -> q [5,4]、k [5,2]、v [5,2]；split 返回源张量的视图。
        # 用 rank 0 的某个 token 举例，假设其投影结果为 [1,2,3,4,5,6,7,8]：
        #   q=[1,2,3,4]，前两个数属于 Q0，后两个数属于 Q1。
        #   k=[5,6] 属于 K0，v=[7,8] 属于 V0；这些数字仅示意投影后的布局。
        # rank 1 同样按 [4,2,2] 拆分，但得到的是自己的 Q2/Q3、K1、V1。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if q.dim() == 2:
            # 将展平的头特征恢复为“头数、每头维度”，token 顺序保持不变。
            # -1 推导 T：q [T,本地 Q 头数*d] -> [T,本地 Q 头数,d]，示例 [5,2,2]。
            # k/v 使用本地 KV 头数，示例各为 [5,1,2]；Q/K/V 头数无需相同。
            # 上面的单 token q=[1,2,3,4] 会被解释成 [[1,2],[3,4]]，两个子向量分别是 Q0/Q1。
            # 单 token k=[5,6] -> [[5,6]]，v=[7,8] -> [[7,8]]，各保留一个 KV 头。
            # view 只改变视图形状，不进行注意力计算，也不复制或重排头的内容。
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            # 此分支按三维 x=[B,N,H] 处理，将投影结果恢复为 [B,N,本地头数,d]。
            # 对应当前 RoPE 的广播规则，positions 应为 [N]，各 batch 共用这组位置。
            # 但这里没有校验输入维数，也不能据此认为整个注意力已完整支持三维批输入：
            # 下游 varlen/paged 内核仍按 [T,heads,d] 读取张量，未在注意力计算前统一展平 B/N。
            # 本项目实际推理应使用上面的 [T,H] 主路径；此分支仅提供 reshape 层面的处理。
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # 当前实现仅在 qkv_bias 为 False 时执行 Q/K RMSNorm；True 时两个 Norm 都被跳过。
        # 这是本代码的分支规则，并不是“有偏置就不能归一化”的数学限制。
        # 每个头沿最后一维归一化，形状不变：示例 q [5,2,2]、k [5,1,2]；V 保持投影结果。
        if self.qkv_bias is False:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 在归一化之后按位置旋转 Q/K，V 不旋转；输出形状和各 rank 的头分片保持不变。
        # prefill 的拼接位置例为 [0,1,0,1,2]，不同请求各自从 0 开始，不能直接视作一条长序列。
        # 行号 2 的 token 是 B0，虽然它在拼接张量中排第 3 行，RoPE 位置仍为 0。
        # cu_seqlens_q=[0,2,5] 则标记 A 的行区间 [0:2]、B 的行区间 [2:5]，
        # 描述的是拼接后的行边界，与 positions 的请求内位置含义不同。
        # decode 通常取 context_lens-1：上下文长度 [3,7] 对应当前新 token 的位置 [2,6]。
        # positions 只用于 RoPE，注意力请求边界还需要运行上下文，不能由位置索引代替。
        q, k = self.rotary_emb(positions, q, k)

        # Attention 在 cache 已分配且 slot_mapping 存在时，先把本轮 K/V 写入各自缓存槽位。
        # 写入的是上述处理后的 K（已执行 RoPE），以及未旋转的 V；本类不直接管理物理缓存地址。
        # prefill：使用本轮 q/k/v 和 cu_seqlens_q，逐请求执行带因果掩码的 varlen 注意力。
        # 示例 A1 可以关注 A0/A1；B2 可以关注 B0/B1/B2，但不会关注 A0/A1。
        # 当前 prefill 内核直接读取本轮 k/v，不通过块表补读历史缓存前缀。
        # decode：使用当前 Q，通过 block_tables/context_lens 从缓存读取当前及历史 K/V。
        # 两条请求各一个新 token 时，输入 x=[2,8]、本地 q=[2,2,2]、k/v 各 [2,1,2]。
        # 即使它们的历史长度不同，当前 Q 的 token 数仍为 2；历史 K/V 不需要重新从隐藏向量投影。
        # 注意力内部每头输出 [T,本地 Q 头数,d]，返回前已展平为 [T,本地 Q 头数*d]。
        # 因此这里的 o 是二维本地特征张量，示例 [5,4]，不是仍带独立头维度的 [5,2,2]。
        # 共用 KV 的两个 Q 头依然各自输出 2 维：Q0/Q1 的注意力权重通常不同，输出也通常不同。
        # 两个头展平后每个 token 有 2*2=4 维；decode 时相应输出为 [2,4]。
        o = self.attention(q, k, v)

        # 行并行输出投影：各 rank 的本地头先贡献到全部 H 个输出特征，再跨 rank 求和。
        # 示例输入 [5,4] -> 本地贡献 [5,8] -> all_reduce 后每个 rank 都有完整 [5,8]。
        # rank 0 的 [5,4] 来自 Q0/Q1，rank 1 的 [5,4] 来自 Q2/Q3；
        # 各自先通过对应权重映射到相同的 8 个隐藏输出特征，再将这些贡献相加。
        # decode 的 token 数改为 2，同样经过 [2,4] -> [2,8] -> 求和后的完整 [2,8]。
        # 求和发生在投影贡献之间，不是把不同注意力头的向量直接逐元素相加。
        # 不需要在这里手动展平 o 或收集完整多头向量，Attention/RowParallelLinear 已分别处理。
        o = self.o_proj(o)

        # 返回注意力子层的完整隐藏向量；残差相加仍由外部 DecoderLayer 的流程负责。
        return o

# Qwen3MLP
# gate_up
# activateion
# gate_down
class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        self.activation = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x


# Qwen3DecoderLayer
# input_layernorm, also consider residual
# self_attn
# layer_norm post attention
# mlp
# 一个 DecoderLayer 包含两组“先归一化、再计算子层”的操作：RMSNorm -> Attention，RMSNorm -> MLP。
# 这里的 LayerNorm 实际是 RMSNorm；本层处理隐藏向量，不直接接收 token ID，也不计算词表 logits。
# 常规公式：a = Attention(RMSNorm(h))；r = h + a；m = MLP(RMSNorm(r))；完整层输出为 r + m。
# 本实现将 r 和 m 分开返回，在下一层入口或模型末尾的 RMSNorm 中再相加，保留残差流。
# 形状例子（单进程，仅用于说明）：hidden_size=4，num_heads=2，head_dim=2，num_kv_heads=1，intermediate_size=8。
# prefill 中 A、B 两条请求分别有 2、3 个待处理 token，拼接后输入 x 为 [5, 4]；decode 输入为 [2, 4]。
# 数值例子只跟踪其中一个 token：首层输入 h=[1, 2, 3, 4]，两处 RMSNorm 的 gamma 都取全 1。
# Attention 和 MLP 的数值输出在下面人为假定以说明残差运算，不是仅凭 h 就能推导出的实际模型输出。
class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        block_size: int = 256,
    ):
        # 注册后续子模块，使其参数可以被统一加载、移动设备和管理。
        super().__init__()
        # 每一层的输入归一化层
        # hidden_size 是每个 token 的完整隐藏维度，本例 gamma=[1, 1, 1, 1]，形状 [4]。
        # gamma 用于初始化可学习的缩放参数，加载 checkpoint 后通常会替换成训练所得的值。
        gamma = torch.ones(hidden_size)
        # 当前调用没有传 eps，所以使用 LayerNorm 的默认值 1e-5；rms_norm_epsilon 没有用于这两处归一化。
        self.input_layernorm = LayerNorm(gamma)

        # 每一层的自注意力层
        # hidden_size=4 决定输入/输出宽度；num_heads=2、head_dim=2 决定 Q 的两个注意力头。
        # num_kv_heads=1 表示两组 Q 头共用一组 K/V 头（GQA）；None 时使用与 Q 相同数量的 K/V 头。
        # scale 是注意力 QK 得分的乘数；此代码直接使用传入值，默认值为 1.0。
        # qkv_bias 控制 QKV 线性层偏置；当前 Qwen3Attention 还以它是否为 False 决定是否做 Q/K RMSNorm。
        # base、max_position 配置 RoPE；block_size 是每个 KV cache 块能保存的 token 数量。
        # rms_norm_epsilon 被传给注意力构造函数，但当前 Qwen3Attention 内部创建 Q/K RMSNorm 时也未使用它。
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            block_size=block_size,
        )

        # 每一层的自注意力层后的归一化层
        # 它归一化的是“注意力输出 + 残差”，为后续 MLP 准备输入，并不是只归一化注意力输出。
        # 虽然传入相同的 gamma，LayerNorm 会 clone 并创建 Parameter，因此两处缩放权重独立、不共享。
        self.post_attention_layernorm = LayerNorm(gamma)

        # 每一层的MLP层，里面没有归一化层，对应的归一化层被提前到了post_attention_layernorm
        # intermediate_size=8 是 MLP 的中间宽度，ffn_bias 控制 gate/up 和 down 投影的偏置。
        # 单进程时：[5, 4] -> gate_up [5, 16] -> SiLU(gate)*up [5, 8] -> down_proj [5, 4]。
        # gate_up 合并两个各为 8 维的投影；MLP 对每个 token 的特征做变换，token 之间的信息交互由注意力完成。
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # x、residual 在本项目中均为 [N, hidden_size]；本例 prefill 是 [5, 4]，两者相加不改变形状。
        # 首层 x 是 embedding 输出且 residual=None；后续层 x 是上一层 MLP 输出，residual 是待累加的残差流。
        if residual is not None:
            # 将上一层尚未相加的 MLP 输出并入残差，返回 (RMSNorm(x+residual), x+residual)。
            # 接续下面的例子：x=[0.1, 0.2, 0.3, 0.4]，residual=[2, 3, 4, 5]。
            # 更新 residual=[2.1, 3.2, 4.3, 5.4]；归一化后的 x≈[0.532115, 0.810841, 1.089568, 1.368295]。
            x, residual = self.input_layernorm(x, residual)
        else:
            # 第一层的输入归一化层
            # 在归一化之前保存原始 h=[1, 2, 3, 4]，用于后续残差相加。
            # residual=x 是保存张量引用；下面 RMSNorm 返回新张量，所以 residual 仍是原始 h。
            residual = x  # Save BEFORE normalization
            # RMS 分母为 sqrt((1+4+9+16)/4 + 1e-5)；归一化后 x≈[0.365148, 0.730296, 1.095444, 1.460593]。
            x = self.input_layernorm(x)
        # Compute positions based on context (respecting sequence boundaries for batched prefill)
        # context 由 ModelRunner 准备，包含推理阶段、序列边界、上下文长度及 KV cache 映射。
        from myvllm.utils import get_context
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            # For batched prefill, create positions that restart at 0 for each sequence
            # 例：cu_seqlens_q=[0, 2, 5]；A 在 x[0:2]，B 在 x[2:5]。
            # 这里构建的是每个 token 的 RoPE 位置编号，不是 token ID 或 KV cache 物理槽位。
            positions = []
            # 转为 CPU 上的 Python 列表，供下方循环逐条请求计算本轮长度。
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                # 两次迭代得到 seq_len=2、3，分别追加 [0, 1] 和 [0, 1, 2]。
                seq_len = cu_seqlens[i+1] - cu_seqlens[i]
                positions.extend(range(seq_len))
            # positions=[0, 1, 0, 1, 2]，形状 [5]，与 x 的 5 行一一对应，并放到相同设备。
            # 上述编号对应无缓存前缀的例子；当前代码总是从 0 起，未加入缓存前缀的长度偏移。
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        elif context.is_prefill:
            # For single sequence prefill, use sequential positions
            # 进入此分支的条件：当前是 prefill，且上一个分支未满足，即 context.cu_seqlens_q 为 None。
            # 缺少边界信息时，这里的位置构造逻辑假设输入属于一条序列，从第 0 个位置连续编号。
            # 例：一条请求有 5 个 token，每个隐藏向量有 4 个特征，x 的形状为 [5, 4]。
            # x[0]～x[4] 分别是这 5 个 token 的隐藏向量；4 是 hidden_size，不是序列长度。
            # x.size(0) 取第 0 维的长度，得到 5；x.size(1) 才是隐藏维度 4。
            # torch.arange(5) 等价于生成整数区间 [0, 5)，起点默认为 0，步长默认为 1，不包含终点 5。
            # 因此 positions = tensor([0, 1, 2, 3, 4])，形状为 [5]，整数终点使这里默认生成 int64 张量。
            # 每一行与一个位置一一对应：x[0] -> 0，x[1] -> 1，x[2] -> 2，x[3] -> 3，x[4] -> 4。
            # positions 只有每个 token 的位置编号，不需要给该 token 的 4 个特征分别生成编号。
            # 它也不是 token ID：即使输入 token ID 是 [42, 7, 42, 9, 3]，对应的位置仍然是 [0, 1, 2, 3, 4]。
            # device=x.device 让 positions 直接创建在 x 所在的设备上；例如 x 在 cuda:1，就在 cuda:1 创建。
            # 这里只匹配设备，不匹配数据类型：x 是浮点隐藏向量，positions 是整数位置索引。
            # 后续注意力将 positions 传给 RoPE，按这些位置旋转 Q/K；此行本身不改变 x，也不执行注意力。
            # 与有边界的例子不同：若这 5 行实际来自长度为 2、3 的两条请求，位置应为 [0, 1, 0, 1, 2]。
            # arange 无法从隐藏向量推断请求边界，因此多请求拼接时需要前一分支的 cu_seqlens_q 信息。
            positions = torch.arange(x.size(0), device=x.device)
        else:
            # For decode, use context_lens - 1 as positions (current position for each sequence)
            # decode 中每条请求本轮只输入最后一个 token，之前 token 的 K/V 已在各层的 KV cache 中。
            # context_lens 由 ModelRunner.prepare_decode 中的 len(seq) 构建，包含历史 token 和当前输入 token。
            # 它记录的是每条请求目前的完整长度，不是本轮输入数量，也不是仅有的历史缓存长度。
            # 例：batch 有两条请求，A 的完整 token ID 为 [1, 4, 9]，B 为 [2, 3, 4, 5, 6, 7, 8]。
            # 本轮 input_ids 只取各请求末尾的 token，得到 [9, 8]；context_lens=tensor([3, 7])。
            # 这两个 token 经过 embedding 和前面层的处理，在这里对应 x 的两行；hidden_size=4 时 x 为 [2, 4]。
            # x[0] 是 A 当前 token 9 的隐藏向量，x[1] 是 B 当前 token 8 的隐藏向量。
            # 序列位置从 0 开始：A 的三个位置为 [0, 1, 2]，B 的七个位置为 [0, 1, 2, 3, 4, 5, 6]。
            # 因而最后一个 token 的位置等于完整长度减 1，逐元素计算 [3, 7] - 1 = [2, 6]。
            # positions 的值为 tensor([2, 6])，形状为 [2]：表示一维张量有两个元素，不是只包含数值 2。
            # 对应关系：x[0] -> A 的位置 2，x[1] -> B 的位置 6；二者不要求有相同的序列长度。
            # 不能改成 arange(2)=[0, 1]：那是当前 batch 的行编号，不是 token 在各自完整序列中的位置。
            # 当前 token ID [9, 8]、序列位置 [2, 6]、KV cache 物理槽位是三种不同的信息。
            # 后续 RoPE 使用 [2, 6] 旋转当前 Q/K，得到与它们在各自序列中所处位置相对应的表示。
            # 本层算出当前 K/V 后，注意力先按 slot_mapping 写入本层缓存，再读取各请求的完整 K/V。
            # A 的当前 Q 关注位置 0～2 的 K/V，B 的当前 Q 关注位置 0～6 的 K/V，包括当前 token 自己。
            # 历史 K/V 不需要重新投影；block_tables 指明缓存块位置，context_lens 指明每条请求的有效长度。
            # 当前输入 token 通常是上一轮刚生成并追加到请求中的 token；本轮用它的输出预测再下一个 token。
            # 本轮生成新 token 并追加后，下一轮完整长度变成 [4, 8]，对应的当前 token 位置就是 [3, 7]。
            # 这一行返回新的位置张量，不会把 context.context_lens 本身从 [3, 7] 改成 [2, 6]。
            positions = context.context_lens - 1

        x = self.self_attn(x, positions=positions)
        # Residual connection always on for attention output
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual

# Qwen3Model
# embedding
# layers stack
# final layer norm
class Qwen3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        block_size: int = 256,
    ):
        super().__init__()
        # 模型的词嵌入层，将输入的token ids 转换为隐藏状态
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,  # 词表大小
            embedding_dim = hidden_size # 每个词或者每个 token 的隐藏状态维度
        )
        # 模型的层栈，每个层包含一个自注意力层和一个MLP层
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_epsilon=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
            ) for _ in range(num_layers)
        ])
        # 模型的最终归一化层，将隐藏状态归一化到[-1, 1]之间
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x, _ = self.norm(x, residual)
        return x



# Qwen3ForCausalLM
# add lm_head on top of Qwen3Model
class Qwen3ForCausalLM(nn.Module):
    # 权重加载器使用的映射规则，当前未被使用
    packed_module_mapping = {
        "q_proj": ('q_proj', 'q'),
        "k_proj": ('k_proj', 'k'),
        "v_proj": ('v_proj', 'v'),
        "gate_up": ('gate_up_proj', '0'),
        "gate_down": ('gate_down_proj', '1'),
    }
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False,
        block_size: int = 256,
    ):
        super().__init__()
        head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        # 模型的主体部分
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size,
        )
        # 模型的语言头部分，主要是用来计算推理出的隐藏状态在词表中的概率分布
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        # 如果需要将词嵌入和语言头的权重共享，那么就将语言头的权重设置为词嵌入的权重
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 前向传播计算隐藏状态
        x = self.model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 计算隐藏状态在词表中的概率分布
        logits = self.lm_head(hidden_states)
        return logits

if __name__ == "__main__":
    model = Qwen3ForCausalLM(
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        num_layers=2,
    )
    input_ids = torch.randint(0, 50257, (2, 16)).cuda()
    output = model(input_ids)
