from myvllm.layers import *
import torch 
import torch.nn as nn

# Qwen3Attention: 
# qkv projection
# if not qkv_bias: then rms_norm
# apply rotary embedding to q, k
# attention
# output projection
class Qwen3Attention(nn.Module):
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
        block_size: int = 256,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        # self.num_kv_heads is per-GPU value (divided by tp_size)
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = scale

        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,  # Fixed: was head_dim * total_num_heads, should be hidden_size
            head_size=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        # Q and K norms as used in Qwen3
        self.q_norm = LayerNorm(torch.ones(head_dim))
        self.k_norm = LayerNorm(torch.ones(head_dim))

        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position
        )

        self.attention = Attention(
            self.num_heads,
            head_dim,
            scale,
            self.num_kv_heads,
            block_size
        )

        self.o_proj = RowParallelLinear(
            input_size=head_dim * self.total_num_heads,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self, 
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Input: x shape (B, N, hidden_size) - REPLICATED on all GPUs

        # ===== QKV Projection (Column Parallel - THIS IS WHERE SHARDING HAPPENS) =====
        # Output shape PER GPU: (B, N, head_dim * (num_heads + 2*num_kv_heads))
        # where num_heads = total_num_heads/tp_size
        #       num_kv_heads = total_num_kv_heads/tp_size
        qkv = self.qkv_projection(x)

        # ===== Split QKV =====
        # q_size = head_dim * num_heads           - Per-GPU size!
        # kv_size = head_dim * num_kv_heads       - Per-GPU size!
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Handle both batched (3D) and varlen (2D) inputs
        # Varlen: q shape: (total_tokens, q_size) where q_size = num_heads * head_dim
        # Batched: q shape: (B, N, q_size)
        if q.dim() == 2:
            # Varlen mode: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            # Batched mode: (B, N, q_size) -> (B, N, num_heads, head_dim)
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # Apply Q and K norms - these are used in Qwen3 to stabilize attention
        # Applied to q and k because they participate in attention_weight computation
        # Removes possibility of large numbers that cause softmax instability
        if self.qkv_bias is False:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)

        o = self.attention(q, k, v)
        # o shape: (B*N, num_heads, head_dim)     - Per-GPU, different heads per GPU

        # ===== Output Projection (Row Parallel - COMMUNICATION HAPPENS HERE by dist.all_reduce) =====
        o = self.o_proj(o)
        # Input: (B*N, num_heads * head_dim) sharded across GPUs
        # Output: (B*N, hidden_size) REPLICATED on all GPUs (after all_reduce)

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
