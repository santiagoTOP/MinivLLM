import torch.nn as nn 
import torch
import torch.distributed as dist
import os

class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim 
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""

# the simpliest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
class ReplicatedLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# columnsplit Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter
class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the output column
        full_data_output_size = loaded_weights.size(0)
        # dim size after sharding
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# an extension of ColumnParallelLinear by merging several matrices
# 将多个具有相同输入维度的线性投影合并，使每个进程用一次 F.linear 同时计算多个投影的本地输出。
# 它继承父类的 forward：y = x @ weight.T + bias；主要新增各投影的大小记录和分段加载规则。
# ColumnParallel 按输出特征分片：各进程接收相同的完整 x，输出不同特征，而不是切分 batch/token。
# PyTorch 存储 weight 的形状是 [输出特征数, 输入特征数]，因此输出维度分片对应 weight 的行分片。
# 例：input_size=2，output_sizes=[4, 4]（gate、up 各输出 4 维），tp_size=2，bias=False。
# 完整权重 W_gate=[[1, 0], [0, 1], [1, 1], [2, 1]]，W_up=[[10, 0], [0, 10], [10, 10], [20, 10]]。
# 本地权重布局是 [gate_local; up_local]，每个投影各占 4/2=2 行，整个 weight 形状为 [4, 2]。
# rank 0 保存 [G0, G1, U0, U1]；rank 1 保存 [G2, G3, U2, U3]，G/U 分别代表 gate/up 的权重行。
# 不能简单拼接完整 [G0,G1,G2,G3,U0,U1,U2,U3] 再均分，否则 rank 0 只有 gate，rank 1 只有 up。
# 实际 Qwen3MLP 用 output_sizes=[intermediate_size, intermediate_size] 合并 gate_proj 和 up_proj。
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, 
        input_size: int, 
        output_sizes: list[int], # e.g. merge QKV matrices to compute MM together and then split
        bias: bool = True,
    ):
        # input_size 是完整输入宽度；output_sizes 是各原始投影的完整输出宽度，不是本地分片宽度。
        # 这里保存 [4, 4]，顺序决定本地参数/输出的布局，也决定 loaded_weight_id=0、1 的含义。
        self.output_sizes = output_sizes
        # sum([4, 4])=8，父类再除以 tp_size=2，分配本地 weight [4, 2]。
        # bias=True 时还分配本地 bias [4]；它的顺序同样是 [gate_bias_local; up_bias_local]。
        # 父类将 self.weight_loader 挂到参数上；此时 self 是子类实例，因此绑定的是下面的分段加载方法。
        # 正确使用要求每个 output_sizes[i] 都能整除 tp_size；当前父类仅检查总和能否整除，不做补齐。
        super().__init__(input_size, sum(output_sizes), bias)

    # 没有重写 forward，所以 merged(x) 会调用 ColumnParallelLinear.forward，而不是在这里逐投影循环。
    # 本例各进程输入 x=[[1, 2]]，形状 [1, 2]；一次 F.linear 得到本地 [1, 4] 输出：
    #   rank 0：[[1, 2, 10, 20]]，前两列是 gate_local，后两列是 up_local。
    #   rank 1：[[3, 4, 30, 40]]，例如 gate 第 3 个特征为 1*1+2*1=3，up 对应特征为 1*10+2*10=30。
    # 对 N 个 token，输入/本地输出形状分别是 [N, 2]、[N, 4]；token 数不变，仅输出特征被分片。
    # 本层 forward 不执行 gather 或 all_reduce，两个进程各自保留不同的本地结果。
    # Qwen3MLP 的 SiluAndMul 将本地结果沿最后一维分成 gate/up 两半，再逐元素计算 SiLU(gate)*up。
    # rank 0 处理全局特征 0、1，rank 1 处理特征 2、3；之后 down_proj 的行并行运算再汇总贡献。
    # 如需还原完整输出，应先分别拼接各 rank 的 gate 和 up，再合并：[1,2,3,4,10,20,30,40]。
    # 直接按 rank 拼接会变成 [1,2,10,20,3,4,30,40]，顺序不等于 [gate_all, up_all]。

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        将一个原始投影的完整权重或偏置切分，并写入当前进程的合并参数对应片段。

        本例加载两个投影时，各进程分别调用：
            merged = MergedColumnParallelLinear(2, [4, 4], bias=False)
            merged.weight_loader(merged.weight, W_gate, loaded_weight_id=0)
            merged.weight_loader(merged.weight, W_up, loaded_weight_id=1)

        loaded_weights 是当前这一个投影的完整矩阵 [4, 2]，不是已合并的 [8, 2]。
        loaded_weight_id 是投影在 output_sizes 中的索引，与当前进程的 tp_rank 不同。
        """
        # 本地 param_data 的形状为 [4, 2]，但这次调用只更新其中 gate 或 up 的两行。
        # weight 初始由 torch.empty 分配，所以两个投影都要正确加载后才能使用。
        param_data = param.data
        # compute offset 
        # offset 是当前投影在“本地目标参数”中的起点，所有 rank 的本地布局相同：
        #   id=0（gate）：sum([])/2=0；id=1（up）：sum([4])/2=2。
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        # compute size
        # 当前投影分给每个 rank 的行数：无论 gate 还是 up，本例 shard_size=4/2=2。
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # find the correct slice to be loaded in the sharded parameter
        # narrow(0, offset, shard_size) 沿行维度取视图：gate 目标为 param[0:2]，up 目标为 param[2:4]。
        # 这是原参数的视图，不是独立副本；后续 copy_ 会直接更新原始 weight 的对应两行。
        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        # 源矩阵起点由 rank 决定：rank 0 从第 0 行取，rank 1 从第 2 行取。
        # 注意与 offset 区分：offset 取决于投影 ID，loaded_weights_start_index 取决于进程 rank。
        loaded_weights_start_index = self.tp_rank * shard_size
        # 若当前是 rank 1、id=1：从完整 W_up 中取第 2、3 行，shard_weights=[[10, 10], [20, 10]]。
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        # 将这两行写入 rank 1 的本地 weight[2:4]；gate 的 weight[0:2] 不受本次加载影响。
        # 两次加载完成后，rank 1 的 weight=[[1, 1], [2, 1], [10, 10], [20, 10]]。
        # 偏置也可用同一规则加载：传 param=merged.bias、完整 bias [4] 及对应投影 ID，目标视图为 [2]。
        # 外部加载器必须显式提供 loaded_weight_id；给参数附加 weight_loader 不会自动执行这些加载调用。
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # Calculate per-GPU output size
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # Pass TOTAL output size to parent (it will divide by tp_size)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    # load_weight_id: q, k, v
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        # batch_size * num_heads * num_token * head_size
        param_data = param.data
        # loaded_weights: batch_size * num_token * (head_size*num_heads)
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # compute offset
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the input row
        full_data_input_size = loaded_weights.size(1)
        # dim size after sharding
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result




if __name__ == "__main__":
    # how to run?
    # 1. cd src/myvllm/layers
    # 2. CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 linear.py # 4 GPUs

    def _init_dist(): 
        rank = int(os.environ["RANK"]) 
        world_size = int(os.environ["WORLD_SIZE"]) 
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) 
        backend = "nccl" if torch.cuda.is_available() else "gloo" 

        if torch.cuda.is_available(): 
            torch.cuda.set_device(local_rank) 
            device = torch.device("cuda", local_rank) 
            dist.init_process_group(
                backend=backend, 
                init_method="env://", 
                device_id=local_rank,
                ) 
        else: 
            device = torch.device("cpu") 
            dist.init_process_group( backend=backend, init_method="env://", ) 
        return rank, world_size, local_rank, device

    # Single linear layer and column parallel test
    @torch.no_grad()
    def test_column_parallel(device):
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()  # Number of parallel GPUs

        in_features = 1024 * tp_size
        out_features = 1024 * tp_size
        batch = 4

        # Ensure that each rank gets exactly the same full input/weight
        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_full = torch.randn(out_features, in_features, generator=g)
        b_full = torch.randn(out_features, generator=g)

        x_full = x_full.to(device)
        w_full = w_full.to(device)
        b_full = b_full.to(device)

        # reference (single GPU)
        single_layer = ReplicatedLinear(in_features, out_features, bias=True).to(device)
        single_layer.weight.weight_loader(single_layer.weight, w_full)
        single_layer.bias.weight_loader(single_layer.bias, b_full)
        y_single = single_layer(x_full)

        # TP layer (each rank stores out_features/tp)
        col_tp_layer = ColumnParallelLinear(in_features, out_features, bias=True).to(device)
        col_tp_layer.weight.weight_loader(col_tp_layer.weight, w_full)
        col_tp_layer.bias.weight_loader(col_tp_layer.bias, b_full)

        # forward
        y_col_tp = col_tp_layer(x_full)  # [batch, out_features/tp]

        # Restore full output: all_gather+concat
        y_parts = [torch.empty_like(y_col_tp) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_col_tp)
        y_full = torch.cat(y_parts, dim=-1)  # [batch, out_features]

        # Alignment check (print only at rank0)
        max_err = (y_full - y_single).abs().max().item()
        ok = torch.allclose(y_full, y_single, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[ColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # MergedColumnParallelLinear test
    @torch.no_grad()
    def test_merged_column_parallel(device):
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        # Make the dimension automatically adapt to tp_size to ensure divisibility
        in_features = 1024 * tp_size
        out_each = 512 * tp_size
        out_sizes = [out_each, out_each, out_each]  # Combination of Q, K and V matrices
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_q = torch.randn(out_sizes[0], in_features, generator=g)
        w_k = torch.randn(out_sizes[1], in_features, generator=g)
        w_v = torch.randn(out_sizes[2], in_features, generator=g)

        x_full = x_full.to(device)
        w_q = w_q.to(device)
        w_k = w_k.to(device)
        w_v = w_v.to(device)

        # Reference: single-card equivalent output (Q | K | V concat)
        y_ref = torch.cat(
            [
                nn.functional.linear(x_full, w_q, None),
                nn.functional.linear(x_full, w_k, None),
                nn.functional.linear(x_full, w_v, None),
            ],
            dim=-1,
        )

        # TP merged layer
        # NOTE: your MergedColumnParallelLinear defines weight_loader(param, loaded_weights, loaded_weight_id),
        # so bias loader signature doesn't match base (param, loaded_weights). Therefore bias=False here.
        merged = MergedColumnParallelLinear(in_features, out_sizes, bias=False).to(device)
        merged.weight_loader(merged.weight, w_q, 0)
        merged.weight_loader(merged.weight, w_k, 1)
        merged.weight_loader(merged.weight, w_v, 2)

        y_local = merged(x_full)  # [batch, sum(out_sizes)/tp], layout: [q_local, k_local, v_local]

        # all_gather then re-pack to [Q_all | K_all | V_all]
        y_parts = [torch.empty_like(y_local) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_local)

        ql = out_sizes[0] // tp_size
        kl = out_sizes[1] // tp_size
        vl = out_sizes[2] // tp_size

        q_full = torch.cat([p[:, :ql] for p in y_parts], dim=-1)
        k_full = torch.cat([p[:, ql : ql + kl] for p in y_parts], dim=-1)
        v_full = torch.cat([p[:, ql + kl : ql + kl + vl] for p in y_parts], dim=-1)
        y_full = torch.cat([q_full, k_full, v_full], dim=-1)

        max_err = (y_full - y_ref).abs().max().item()
        ok = torch.allclose(y_full, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[MergedColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # QKVColumnParallelLinear test
    @torch.no_grad()
    def test_qkv_column_parallel(device):
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        # make input dim divisible by tp_size
        input_size = 1024 * tp_size
        head_size = 16
        num_heads = 4 * tp_size
        num_kv_heads = 2 * tp_size
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, input_size, generator=g)
        w_q = torch.randn(head_size * num_heads, input_size, generator=g)
        w_k = torch.randn(head_size * num_kv_heads, input_size, generator=g)
        w_v = torch.randn(head_size * num_kv_heads, input_size, generator=g)

        x_full = x_full.to(device)
        w_q = w_q.to(device)
        w_k = w_k.to(device)
        w_v = w_v.to(device)

        # reference: full Q|K|V
        y_ref = torch.cat(
            [
                nn.functional.linear(x_full, w_q, None),
                nn.functional.linear(x_full, w_k, None),
                nn.functional.linear(x_full, w_v, None),
            ],
            dim=-1,
        )

        # TP QKV layer: rank output layout [q_local | k_local | v_local]
        qkv = QKVColumnParallelLinear(
            input_size=input_size,
            head_size=head_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=False,
        ).to(device)

        qkv.weight_loader(qkv.weight, w_q, "q")
        qkv.weight_loader(qkv.weight, w_k, "k")
        qkv.weight_loader(qkv.weight, w_v, "v")

        y_local = qkv(x_full)  # [batch, head_size*(local_h + 2*local_kv)]

        # all_gather then re-pack to [Q_all | K_all | V_all]
        y_parts = [torch.empty_like(y_local) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_local)

        ql = head_size * (num_heads // tp_size)
        kl = head_size * (num_kv_heads // tp_size)
        vl = head_size * (num_kv_heads // tp_size)

        q_full = torch.cat([p[:, :ql] for p in y_parts], dim=-1)
        k_full = torch.cat([p[:, ql : ql + kl] for p in y_parts], dim=-1)
        v_full = torch.cat([p[:, ql + kl : ql + kl + vl] for p in y_parts], dim=-1)
        y_full = torch.cat([q_full, k_full, v_full], dim=-1)

        max_err = (y_full - y_ref).abs().max().item()
        ok = torch.allclose(y_full, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[QKVColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # RowParallelLinear test
    @torch.no_grad()
    def test_row_parallel(device):
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        in_features = 128 * tp_size
        out_features = 256
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_full = torch.randn(out_features, in_features, generator=g)
        b_full = torch.randn(out_features, generator=g)

        x_full = x_full.to(device)
        w_full = w_full.to(device)
        b_full = b_full.to(device)

        # reference
        single = ReplicatedLinear(in_features, out_features, bias=True).to(device)
        single.weight.weight_loader(single.weight, w_full)
        single.bias.weight_loader(single.bias, b_full)
        y_ref = single(x_full)

        # RowParallel
        row_tp = RowParallelLinear(in_features, out_features, bias=True).to(device)
        row_tp.weight.weight_loader(row_tp.weight, w_full)

        if row_tp.bias is not None:
            row_tp.bias.data.copy_(b_full / tp_size)

        shard = in_features // tp_size
        start = tp_rank * shard
        x_part = x_full.narrow(-1, start, shard)

        y_row = row_tp(x_part)

        max_err = (y_row - y_ref).abs().max().item()
        ok = torch.allclose(y_row, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[RowParallel] allclose={ok}, max_abs_err={max_err:.6f}")

    rank, world_size, local_rank, device = _init_dist()
    if rank == 0:
        print(f"Running TP tests with world_size={world_size} on device={device}")

    # The test output 'allclose=True' means passed.
    test_column_parallel(device)
    test_merged_column_parallel(device)
    test_qkv_column_parallel(device)
    test_row_parallel(device)

    dist.barrier()
    dist.destroy_process_group()
