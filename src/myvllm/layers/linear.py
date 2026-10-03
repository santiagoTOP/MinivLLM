import torch.nn as nn 
import torch
import torch.distributed as dist
import os

class LinearBase(nn.Module):
    """
    线性层的公共基类：创建参数、记录张量并行信息，并约定加载/计算接口。

    继承 nn.Module 后，层可以作为模型的子模块，其注册参数可被
    named_parameters()、state_dict()、to()/cuda() 等 PyTorch 接口管理。
    基类不决定如何切分完整权重，也不实现矩阵乘法；这些工作由子类完成。

    注意：这里的 input_size/output_size 是当前进程实际保存的参数维度。
    是否已切分，取决于子类传入的值；基类不会根据 tp_dim 再除一次 tp_size。
    例：完整 Din=8、Dout=12，两个张量并行进程时，各子类传给基类的尺寸为：
        ReplicatedLinear：input_size=8，output_size=12 -> weight [12,8]。
        ColumnParallelLinear：input_size=8，output_size=6 -> weight [6,8]。
        RowParallelLinear：input_size=4，output_size=12 -> weight [12,4]。
    上述三种层都复用本类的参数创建逻辑，区别在于子类的维度选择和具体实现。
    """

    def __init__(
        self, 
        input_size: int,  # 本地权重的输入宽度，对应 weight 的第 1 维。
        output_size: int,  # 本地权重的输出宽度，对应 weight 的第 0 维。
        bias: bool = True,  # 是否创建本地输出偏置；False 时 self.bias 为 None。
        tp_dim: int | None = None  # 权重分片维度：None 不切分，0 切输出，1 切输入。
    ):
        # 初始化 nn.Module 的参数、子模块和 hook 等内部容器。
        # 必须先完成这一步，后续给 self.weight/self.bias 赋 nn.Parameter 才能注册参数。
        super().__init__()
        # 仅记录分片方式；保存 tp_dim 本身不会切分张量，也不会发起通信。
        # 例如 ColumnParallelLinear 传 0，RowParallelLinear 传 1，ReplicatedLinear 默认 None。
        self.tp_dim = tp_dim 
        # 查询默认分布式进程组：rank 是当前进程在组中的编号，size 是组内进程数。
        # rank 不是 GPU 编号；本项目运行器通过一个进程使用一张 GPU 将两者对应起来。
        # 子类的 weight_loader 会用 rank 选择源参数区间，用 size 计算分片大小。
        # 即使不切分权重或只运行一个进程，本基类也要求预先初始化进程组。
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # PyTorch 线性层保存 weight [输出宽度, 输入宽度]，计算时使用 x @ weight.T。
        # torch.empty 只分配存储，不填入有效初始权重；必须加载后才能用于计算。
        # 设备/dtype 遵循 torch 的默认设置，本类没有主动把参数放到特定 GPU 上。
        # nn.Parameter 被赋给 Module 属性时会自动注册；默认 requires_grad=True。
        # 注册后可随模型迁移设备或转换 dtype，也可被参数遍历/优化器发现。
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 为参数附加一个普通 Python 属性，供外部加载器发现并调用专用加载方法。
        # self 仍是具体子类实例，因此 self.weight_loader 会解析到子类重写的方法。
        # 得到的是已绑定 self 的方法，例如：
        #   layer.weight.weight_loader(layer.weight, checkpoint_weight)
        # 等价于 layer.weight_loader(layer.weight, checkpoint_weight)，无需额外传 self。
        # 这不是 PyTorch 自动执行的 hook，也不是 state_dict 中的张量条目；
        # 仅赋值不会加载数据，外部加载器必须显式调用这个属性。
        # Merged/QKV 子类的加载方法还需要投影 ID，应由外部加载器额外提供。
        self.weight.weight_loader = self.weight_loader

        if bias:
            # 每个本地输出特征对应一个偏置，形状 [output_size]，初始值全部为 0。
            # 列并行时它是输出偏置分片；行并行时输出宽度完整，偏置长度也完整。
            # 并行计算中怎样正确加载/加入偏置由子类和调用方负责，本基类不处理归约。
            self.bias = nn.Parameter(torch.zeros(output_size))
            # 偏置同样挂上专用加载方法，但方法能否处理一维偏置取决于子类实现。
            # 例如列并行按第 0 维切分的加载规则适用于权重和偏置；
            # 当前行并行加载方法按第 1 维切权重，不能直接用于一维偏置。
            self.bias.weight_loader = self.weight_loader 
        else:
            # 在 Module 参数表中登记 bias=None，让所有子类都能统一访问 self.bias。
            # 它不分配偏置张量，也不会出现在 named_parameters()/state_dict() 的参数条目中。
            # 子类调用 F.linear(x, self.weight, self.bias) 时，None 表示省略偏置加法。
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        参数加载接口：把 checkpoint 的源张量写入当前层的目标参数。

        param 是已分配的目标参数；loaded_weights 是外部读取的源权重/偏置。
        ReplicatedLinear 直接复制；Column/RowParallelLinear 按各自维度选取分片。
        基类不实现具体规则，调用本方法会抛出异常，要求子类提供对应实现。
        这里没有使用 abc.abstractmethod，因此不会在实例化 LinearBase 时强制检查覆盖。
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算接口：具体的线性运算和必要的进程间通信由子类实现。

        使用 layer(x) 时，nn.Module.__call__ 会调度到子类的 forward，并处理相关 hook。
        普通/列并行子类执行 F.linear；本项目的行并行子类还会归约各进程的输出贡献。
        直接使用本基类执行 layer(x) 会抛出异常，因为这里没有具体计算逻辑。
        """
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

class ColumnParallelLinear(LinearBase):
    """
    按输出特征切分的线性层：每个进程保存部分权重，用完整输入计算部分输出。

    设完整输入/输出维度为 Din/Dout，张量并行进程数为 P：
        普通线性层：X [..., Din] @ W.T [Din, Dout] + b [Dout]
        当前进程：X [..., Din] @ W_r.T [Din, Dout/P] + b_r [Dout/P]
    各进程处理相同的 token 和完整输入特征；切分的是输出特征，不是 batch/token。

    名称中的 Column 来源于数学写法 Y = X @ A，A 的形状是 [Din, Dout]。
    按 A 的列切分，即按输出特征切分。PyTorch 保存 W = A.T，形状 [Dout, Din]，
    因此代码实际沿 W 的第 0 维（行）切分，数学含义仍然是列并行。

    以下注释统一使用两进程示例：Din=2、Dout=4、P=2、bias=True。
    完整 W=[[1,0], [0,1], [1,1], [2,1]]，完整 b=[10,20,30,40]。
    rank 0 保存前两行/前两个偏置，rank 1 保存后两行/后两个偏置。
    两个进程都输入 X=[[1,2]]，分别输出 [[11,22]] 和 [[33,44]]。
    """

    def __init__(
        self, 
        input_size: int,  # 完整输入宽度 Din；每个进程都需要全部输入特征。
        output_size: int,  # 完整输出宽度 Dout；不是当前进程的本地输出宽度。
        bias: bool = True,
    ):
        # 查询默认分布式进程组的进程数，而不是机器上的 GPU 总数。
        # 本项目通常一个进程对应一张 GPU，并将整个默认组用于张量并行。
        # 构造前必须调用 dist.init_process_group；即使 P=1，本实现也需要初始化。
        tp_size = dist.get_world_size()
        # 输出特征必须能平均分配；本层不切分输入，所以不要求 input_size 能整除 P。
        # 示例 Dout=4、P=2，每个进程负责 2 个输出特征。
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        # 将本地输出宽度传给 LinearBase，输入宽度保持完整。
        # 基类据此创建本地 weight [Dout/P, Din]、bias [Dout/P]；示例为 [2,2] 和 [2]。
        # 基类还记录 tp_rank（当前进程编号）和 tp_size（并行进程数）。
        # tp_dim=0 记录权重按输出维度切分；本类加载时直接使用 narrow(0, ...)，
        # 并没有读取 tp_dim 来自动切分参数。
        # 基类用 torch.empty 创建 weight，没有初始化有效权重，使用前必须完成加载。
        # 基类的 self.weight.weight_loader = self.weight_loader 会绑定当前实例的加载方法：
        # 直接创建本类时绑定下面的方法；创建子类时则可绑定子类重写的方法。
        # 附加这个属性不会自动加载；外部调用者仍需显式调用它。
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        从完整 checkpoint 参数中取出当前 rank 的分片，写入本地参数。

        param 是已分配的本地目标；loaded_weights 是当前调用者提供的完整源参数。
        权重形状：[Dout/P, Din] <- [Dout, Din]；偏置形状：[Dout/P] <- [Dout]。
        本方法不执行通信，也不从其他进程接收参数。

        示例调用（在各进程中分别执行，full_weight/full_bias 为完整参数）：
            layer = ColumnParallelLinear(2, 4, bias=True)
            layer.weight.weight_loader(layer.weight, full_weight)
            layer.bias.weight_loader(layer.bias, full_bias)

        参数上的 weight_loader 已绑定 layer，所以调用时不用额外传 self。
        当前 utils/loader.py 的 load_weights_from_checkpoint 直接使用 copy_，
        没有调用这个自定义方法；附加方法本身不能让外部加载器自动支持分片。
        """
        # 取得目标参数的数据张量；后面的 copy_ 原地写入参数，属于权重加载操作。
        param_data = param.data 
        # 权重的第 0 维是完整输出特征数；偏置的一维长度也是完整输出特征数。
        # 示例完整 weight [4,2]、bias [4]，两种情况下此处都得到 4。
        full_data_output_size = loaded_weights.size(0)
        # 每个进程需要的输出特征数：示例 4 // 2 = 2。
        shard_size = full_data_output_size // self.tp_size
        # 检查源参数的分片行数与已分配的本地参数行数是否一致。
        # 这里检查的是第 0 维；正常加载仍要求源权重的输入维度也匹配。
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # 当前 rank 在完整源参数中的起点；rank 0 为 0，rank 1 为 2。
        # 每个 rank 负责连续区间 [rank * shard_size, (rank + 1) * shard_size)。
        start_index = self.tp_rank * shard_size
        # narrow(dim, start, length)：沿第 0 维取 shard_size 行，保留全部输入列。
        # 对二维权重等价于 loaded_weights[start_index:start_index + shard_size, :]。
        # rank 0 取 [[1,0], [0,1]]；rank 1 取 [[1,1], [2,1]]。
        # 对偏置等价于 loaded_weights[start_index:start_index + shard_size]，
        # 两个 rank 分别取 [10,20]、[30,40]，与各自权重的输出特征对应。
        # narrow 返回源张量的视图；此处还没有把数据写入本地参数。
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        # 将源分片复制到目标参数自己的存储中，完成当前进程的参数加载。
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear 执行 x @ self.weight.T + self.bias（bias=None 时省略加法）。
        # 并行来自各进程保存不同的权重分片，计算本身仍是普通的线性运算。
        # 调用者应保证各进程拿到相同的完整 x；本方法不会广播或切分输入。
        # 输入 [..., Din] -> 本地输出 [..., Dout/P]，前面的 batch/token 维度不变。
        # 示例 X=[[1,2]]：rank 0 输出 [[11,22]]；rank 1 输出 [[33,44]]。
        # 每个结果都是对应输出特征的完整值，所以恢复完整输出需要拼接，不能求和：
        #   parts = [torch.empty_like(y_local) for _ in range(self.tp_size)]
        #   dist.all_gather(parts, y_local)
        #   y_full = torch.cat(parts, dim=-1)  # 示例 [[11,22,33,44]]。
        # 上面的收集需由调用者执行，本层 forward 本身没有 all_gather/all_reduce。
        # 若后续算子能处理本地特征，可继续保留分片；例如逐元素激活后接
        # RowParallelLinear，各进程计算最终输出的部分贡献，再用 all_reduce 求和。
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
    """
    合并 Q/K/V 三个投影，并让每个进程保存各投影的一部分输出头。

    三个投影都读取同一个完整隐藏向量 x，但各有独立权重 W_q/W_k/W_v。
    本地合并权重为 [W_q_local; W_k_local; W_v_local]，上下拼接输出特征对应的权重行。
    继承父类 forward，执行一次 F.linear 就得到 [Q_local | K_local | V_local]。
    本层只做投影，不进行 Q/K 归一化、RoPE、softmax 注意力或 KV cache 写入。

    使用与 Qwen3Attention 相同的形状例子：
        input_size=8、head_size=2、总 Q 头数=4、总 KV 头数=2、tp_size=2。
        每个头对应 head_size=2 个输出特征，因此在权重中对应连续的 2 行。
        完整 W_q [8,8]、W_k [4,8]、W_v [4,8]，完整合并权重 [16,8]。
        每个 rank 有 2 个 Q 头和 1 个 K/V 头，本地 Q 宽度=4，K/V 宽度各=2。
        本地合并权重 [8,8]，各 rank 的内容与行布局为：
            本地行区间     rank 0 的全局头     rank 1 的全局头
               [0:4]           Q0/Q1               Q2/Q3
               [4:6]           K0                  K1
               [6:8]           V0                  V1
        注意“1 个 KV 头”表示 K 有一个头、V 也有一个头，不是 K/V 合起来只占一个头。

    为什么不能先拼接完整权重再按普通列并行均分？
        完整行布局是 [Q0,Q1,Q2,Q3,K0,K1,V0,V1]，每个头占两行。
        直接按前/后 8 行均分会让 rank 0 只有所有 Q，rank 1 只有所有 K/V。
        本层却需要每个 rank 都有对应的 Q/K/V，所以必须分别切分三个源投影，
        再将各自分片写入本地合并参数中。这正是下面 weight_loader 的工作。

    两个加载起点的含义不同：
        offset：当前投影在本地目标参数中的起点，由 'q'/'k'/'v' 决定。
        loaded_weights_start_index：当前 rank 在该投影完整源参数中的起点，由 rank 决定。
    示例加载 rank 1 的 K：完整 W_k [4,8] 的行 [2:4] -> 本地 weight [8,8] 的行 [4:6]。
    前者找到源 K1，后者找到本地 K 区域，不应混用两个起点。

    完整输入 x [T,8] 在两个 rank 上相同，输出都是 [T,8]，但分别包含不同头。
    T 是本轮 token 数，未按进程切分；例如 A/B 请求共 5 个 token 时，两边的 T 都是 5。
    Qwen3Attention 随后按 [4,2,2] 拆出 Q/K/V，再恢复头维度。

    数值例子（bias=False，仅跟踪一个 token）：
        令 x=[[1,0,0,0,0,0,0,0]]，每行权重形如 [a,0,0,0,0,0,0,0]。
        完整 W_q 各行的 a 为 [1,2,3,4,5,6,7,8]，W_k 为 [10,20,30,40]，
        W_v 为 [100,200,300,400]，于是两个 rank 各自输出：
            rank 0：[[1,2,3,4,10,20,100,200]]。
            rank 1：[[5,6,7,8,30,40,300,400]]。
        输出的每个数是对应特征的完整值；本层不会跨 rank 求和或收集输出。
        若要还原完整 [Q_all | K_all | V_all]，应分别拼接各 rank 的 Q、K、V，
        再合并三部分，得到 [[1,2,3,4,5,6,7,8,10,20,30,40,100,200,300,400]]。
        直接按 rank 拼接会得到 [Q_rank0,K_rank0,V_rank0,Q_rank1,K_rank1,V_rank1]，
        顺序与完整的 [Q_all,K_all,V_all] 不同。实际注意力可直接消费本地头，无需先收集。
    """

    def __init__(
        self,
        input_size: int,  # 完整输入隐藏宽度 H；示例为 8，各 rank 都读取全部 H 个特征。
        head_size: int,  # 单个头的宽度 d，对应 Qwen3Attention 的 head_dim；示例为 2。
        num_heads: int,  # 完整模型的 Q 头数；示例为 4，尚未除以 tp_size。
        num_kv_heads: int | None = None,  # 完整模型的 K/V 头数；示例为 2。
        bias: bool = False,  # 是否为合并投影创建本地偏置，其顺序也为 [Q_local,K_local,V_local]。
    ):
        # 查询默认分布式进程组，构造前需初始化；P=1 时本实现同样需要进程组。
        self.tp_size = dist.get_world_size()
        # None 时 K/V 头数与 Q 相同，对应 MHA；此处使用 or，因此传入 0 也会被替换。
        # GQA 可传入较少的 K/V 头，K 和 V 各自使用同一个 num_kv_heads 数量。
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        # 属性保存的是本地头数，函数参数 num_heads/num_kv_heads 仍保持全局头数。
        # 示例本地 Q 头数=4//2=2，本地 K/V 头数=2//2=1。
        # 正确配置要求两种全局头数分别能整除 P，且本地头数为正；这里没有逐项断言。
        # 当前只做均分，不支持 num_kv_heads<P 时将同一个 KV 头复制到多个 rank。
        # 若用于 GQA 注意力，还需保证本地 Q 头数是本地 KV 头数的整数倍。
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # 本地输出宽度=d*(本地 Q 头数+本地 K 头数+本地 V 头数)。
        # 示例 2*(2+2*1)=8；其中乘数 2 表示 K/V 两个投影，不是把单个 KV 头加宽。
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # 传给 ColumnParallelLinear 的必须是全局输出宽度，因为父类还会除以 P。
        # 示例全局宽度=2*(4+2*2)=16，父类分配本地 weight [16/2,8]=[8,8]。
        # 如果误传 self.output_size=8，父类会再次除以 2，错误地分配成 [4,8]。
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        # 父类仅检查总输出宽度能否整除 P，不代替上面的逐投影/逐头可整除要求。
        # 基类还记录 tp_rank，并将本类重写的 weight_loader 绑定到 weight 和可选 bias 上。
        # 权重由 torch.empty 创建，三份投影都要加载完成后才能执行 forward。
        super().__init__(input_size, total_output_size, bias=bias)

    # 本类不重写 forward，因此 qkv_layer(x) 调用 ColumnParallelLinear.forward。
    # 一次 F.linear 的本地输出宽度就是 self.output_size，布局由下面的分段加载规则保证。
    # 三维输入 [B,N,H] 也可做线性投影，得到 [B,N,self.output_size]；本层不负责下游注意力适配。

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        """
        加载一个原始投影的完整权重/偏置，更新本地合并参数中该投影对应的区域。

        param：当前 rank 的合并目标权重 [q_local+2*kv_local,H]，或合并偏置 [q_local+2*kv_local]。
        loaded_weights：这一次提供的完整 W_q、W_k、W_v 中的一份，而非已经拼接的完整 QKV。
        load_weight_id：'q'/'k'/'v'，表示这次加载哪个投影；它与进程编号 tp_rank 不同。
        权重张量没有 batch/token 维度，第一维是投影输出特征，第二维才是输入隐藏特征。

        示例调用（每个 rank 都分别执行这三次，源参数为完整 checkpoint 权重）：
            layer = QKVColumnParallelLinear(8, 2, 4, num_kv_heads=2, bias=False)
            layer.weight.weight_loader(layer.weight, W_q, 'q')  # W_q [8,8]。
            layer.weight.weight_loader(layer.weight, W_k, 'k')  # W_k [4,8]。
            layer.weight.weight_loader(layer.weight, W_v, 'v')  # W_v [4,8]。
        也可直接调用 layer.weight_loader(layer.weight, W_q, 'q')，效果相同。
        bias=True 时可用同样规则分别加载完整 b_q [8]、b_k [4]、b_v [4] 到 layer.bias。

        外部调用者必须显式传入投影 ID；参数附加了 weight_loader 并不意味着它会自动执行。
        当前 utils/loader.py 的 QKV 分支直接拼接并 copy_，没有调用本方法，不能自动实现多进程分片。
        本方法只在当前进程切片并复制，不读取文件，也不执行分布式通信。
        """
        # 取得本地合并目标数据；示例整个权重为 [8,8]，本次仅更新其中一个投影区域。
        param_data = param.data
        # 校验投影名称，防止把来源张量写入错误区域；本方法未额外逐项校验源张量的完整形状。
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # offset 是“本地目标”起点；shard_size 是当前投影在本地占用的行数/偏置元素数。
        # 示例的布局固定为 Q 占 4 行、K 占 2 行、V 占 2 行，所有 rank 的本地偏移相同。
        if load_weight_id == 'q':
            # Q 写入目标 [0:4]；每个 rank 各保存 2 个 Q 头，每头 2 行。
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            # K 紧接本地 Q，写入目标 [4:6]；不能使用全局 Q 宽度 8 作为本地起点。
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            # V 紧接本地 Q/K，写入目标 [6:8]。
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        # 沿第 0 维选中目标片段，保留所有输入列；返回原始 param.data 的视图。
        # 例如加载 K 时目标视图为 param.data[4:6,:]，形状 [2,8]。
        # 若是偏置则取 param.data[4:6]，形状 [2]；维度规则与权重的输出特征一致。
        # 此处重新绑定局部变量 param_data，但没有改变 layer.weight 本身的整体形状。
        param_data = param_data.narrow(0, offset, shard_size)
        # 再定位“该投影完整源张量”中的当前 rank 分片，源起点=rank*该投影本地宽度。
        # 示例：Q 的 shard_size=4，rank 0/1 分别取完整 W_q 的 [0:4]/[4:8]；
        # K/V 的 shard_size=2，rank 0/1 分别取各自完整 W_k/W_v 的 [0:2]/[2:4]。
        # 这里的源起点不是完整拼接 QKV 中的行号，不能加上目标 offset。
        loaded_weights_start_index = self.tp_rank * shard_size
        # 源 narrow 同样返回视图，权重分片仍保留全部 H 个输入特征。
        # 若当前 rank=1、id='k'：loaded_weights=W_k [4,8]，取源 [2:4,:]，得到 K1 的两行。
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        # 把源分片原地写入目标视图，从而更新原始合并参数的对应区域，其余 Q/K/V 区域保持原值。
        # 上例实际执行的是 local_weight[4:6,:].copy_(W_k[2:4,:])。
        # 三次调用完成后，rank 1 的本地参数依次存放全局 Q2/Q3、K1、V1 对应的权重行。
        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):
    """
    按输入特征切分的线性层：每个进程计算全部输出特征的部分贡献，再求和。

    设完整输入/输出宽度为 Din/Dout，张量并行进程数为 P：
        完整输入 X [..., Din] 按最后一维分为 X_r [..., Din/P]。
        完整权重 W [Dout, Din] 按第 1 维分为 W_r [Dout, Din/P]。
        各进程先计算 Z_r = X_r @ W_r.T，形状均为 [..., Dout]。
        完整线性结果为 sum_r(Z_r)，因为各 rank 分别累加了不同输入特征的贡献。
    与列并行的区别：列并行计算不同输出特征的完整值，恢复时拼接；本层需要求和。

    名称 Row 基于数学写法 Y=X @ A，其中 A [Din,Dout] 沿输入维度按行切分。
    PyTorch 保存 W=A.T，所以代码对应 W 的列切分，即第 1 维，而不是第 0 维。

    以下注释使用 Din=4、Dout=2、P=2，先以 bias=False 展示计算：
        完整 X=[[1,2,3,4]]，W=[[1,0,1,0], [0,1,0,1]]。
        rank 0：X_0=[[1,2]]，W_0=[[1,0],[0,1]] -> Z_0=[[1,2]]。
        rank 1：X_1=[[3,4]]，W_1=[[1,0],[0,1]] -> Z_1=[[3,4]]。
        all_reduce(SUM) 后，两个进程都得到完整输出 [[4,6]]。
    本层需要调用者传入已分片的输入，且偏置必须避免在归约时重复累加，详见 forward。
    """

    def __init__(
        self,
        input_size: int,  # 完整输入宽度 Din，不是当前 rank 的本地输入宽度。
        output_size: int,  # 完整输出宽度 Dout；每个进程都保留全部输出特征。
        bias: bool = True,
    ):
        # 查询默认分布式进程组的大小；构造前必须初始化进程组，即使只运行一个进程。
        tp_size = dist.get_world_size()
        # 输入特征必须能平均分给 P 个进程；输出不切分，不要求 output_size 能整除 P。
        # 示例 Din=4、P=2，每个进程使用 2 个输入特征。
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        # 仅缩小传给基类的输入宽度，输出宽度保持完整。
        # 基类创建本地 weight [Dout,Din/P]，示例 [2,2]；bias=True 时创建 bias [Dout]。
        # tp_dim=1 记录权重沿输入维度切分；实际加载直接使用下面的 narrow(1, ...)。
        # 基类同时记录 rank/size，并将当前实例的 weight_loader 附加到参数上。
        # weight 使用 torch.empty 分配，使用前需要加载；附加加载方法不会自动执行加载。
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """
        从完整二维权重 [Dout,Din] 中取当前 rank 的列分片，写入本地 [Dout,Din/P]。

        param 是已分配的本地目标，loaded_weights 是调用者提供的完整源权重。
        调用示例：layer.weight.weight_loader(layer.weight, full_weight)。
        本方法不执行通信，也不负责读取 checkpoint 文件。

        此实现只适用于二维权重：会访问 size(1)，不能直接用于一维 bias [Dout]。
        虽然基类也把该方法挂到 bias 上，偏置仍需要调用者单独加载和处理重复累加。
        """
        # 取得目标参数的数据张量，后续 copy_ 将原地填入对应权重分片。
        param_data = param.data 
        # 权重第 1 维是完整输入特征数；示例 loaded_weights [2,4]，这里得到 4。
        full_data_input_size = loaded_weights.size(1)
        # 每个进程需要的列数：示例 4 // 2 = 2。
        shard_size = full_data_input_size // self.tp_size
        # 检查源分片列数是否与本地权重列数一致；完整输出行数也应与目标匹配。
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        # rank 决定在完整权重中取哪一段输入列；示例 rank 0 从 0 开始，rank 1 从 2 开始。
        start_index = self.tp_rank * shard_size
        # narrow(dim, start, length)：沿第 1 维取 shard_size 列，保留全部输出行。
        # 等价于 loaded_weights[:, start_index:start_index + shard_size]。
        # 此列区间必须与 forward 输入 x 所包含的全局输入特征区间一致。
        # narrow 返回源张量的视图；尚未把数据复制到目标参数。
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        # 将当前 rank 的完整输出行、部分输入列复制到其本地权重存储中。
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 应为本地输入 [...,Din/P]；本层不会自动从完整输入中切分或分发数据。
        # 若已有完整 x_full，调用者可取：
        #   width = x_full.size(-1) // self.tp_size
        #   x = x_full.narrow(-1, self.tp_rank * width, width)
        # 若前层为列并行投影（或其后接本地激活），可直接传入其对应的输出分片。
        # F.linear 计算 x @ weight.T + bias，得到 [...,Dout]：
        # 输出宽度虽然完整，但在归约前只是本地输入特征对每个输出的贡献。
        # 无偏置示例：rank 0 得 [[1,2]]，rank 1 得 [[3,4]]。
        # 偏置为什么会重复：本实现在 all_reduce 之前加 bias，各 rank 的结果已包含偏置。
        # 令 Z_r=X_r @ W_r.T，bias_r 是 rank r 保存的整个 [Dout] 偏置向量，
        # 不是偏置的第 r 个元素，也不一定等于 checkpoint 的完整偏置 b。
        #   本地结果：result_r = Z_r + bias_r
        #   求和结果：sum_r(result_r) = sum_r(Z_r) + sum_r(bias_r)
        #   期望结果：X @ W.T + b = sum_r(Z_r) + b
        # 所以正确加载/加入偏置必须满足 sum_r(bias_r)=b。
        # all_reduce 只对整个张量按对应位置求和，不能区分矩阵乘法贡献与偏置。
        # 若所有 P 个 rank 都加同一完整偏置 b，最终得到 X @ W.T + P*b，本层不会自动修正。
        # 两进程错误例：Z_0=[[1,2]]、Z_1=[[3,4]]，完整 b=[10,20]。
        #   rank 0：[[1,2]]+[10,20]=[[11,22]]
        #   rank 1：[[3,4]]+[10,20]=[[13,24]]
        #   求和后：[[24,46]]，而正确结果应为 [[4,6]]+[10,20]=[[14,26]]。
        #
        # 处理方式 1（当前 test_row_parallel 使用）：每个 rank 加载 b/P。
        #   row_tp.bias.data.copy_(b_full / tp_size)
        # 偏置长度仍为 Dout，只把数值除以 P，并不是把偏置切成更短的向量。
        # 本例各 rank 保存 [5,10]，局部结果 [[6,12]]、[[8,14]]，求和得到 [[14,26]]。
        #
        # 处理方式 2：仅一个 rank 加完整 b，其余 rank 的偏置保持为 0。
        # 本例 bias_0=[10,20]、bias_1=[0,0]，局部结果 [[11,22]]、[[3,4]]，
        # 求和同样得到 [[14,26]]。方式 1/2 都可配合当前 forward，不必改变归约顺序。
        #
        # 处理方式 3（另一种实现方式，下面的实际代码尚未采用）：先归约，再加完整偏置。
        #   result = nn.functional.linear(x, self.weight, bias=None)
        #   if self.tp_size > 1:
        #       dist.all_reduce(result, op=dist.ReduceOp.SUM)
        #   if self.bias is not None:
        #       result = result + self.bias
        # 此时每个 rank 都可保存相同完整偏置 b；归约后各自只在完整结果上加一次 b，
        # 加完后不再对结果求和，因此不会变成 P*b。若改成此方式，加载时也不应再除以 P。
        # bias=False、全零偏置或 P=1 时，不存在多个进程重复累加非零偏置的问题。
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            # SUM 按相同位置将所有 rank 的贡献相加，并原地更新每个 rank 的 result。
            # 每个参与进程都得到 [...,Dout] 的完整输出，无需再沿特征维度拼接。
            # 各 rank 必须按相同顺序参与集合通信，并保证结果形状和 token 对应关系一致。
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        # P=1 时本地输入/权重已经完整，直接返回；P>1 时返回归约后的完整结果。
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
