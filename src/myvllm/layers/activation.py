import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    对合并投影的 gate 部分施加 SiLU，再与 up 部分逐元素相乘。

    本层实现 SwiGLU 中的激活/门控步骤，不包含前后的线性投影：
        gate_up = MergedColumnParallelLinear(h)  # 输出顺序为 [gate | up]。
        z = SiLU(gate) * up                      # 本层负责的步骤。
        output = RowParallelLinear(z)            # 后续 down_proj。
    gate/up 是同一输入 h 的两个不同线性投影；若投影层有偏置，偏置已在投影时加入。

    SiLU(t) = t * sigmoid(t) = t / (1 + exp(-t))。
    因此 z_i = gate_i * sigmoid(gate_i) * up_i，乘法按对应特征逐元素执行。
    up 保留其投影值，不在这里额外施加 SiLU；gate 的值也不是限制在 [0,1] 的概率。
    与 ReLU 不同，SiLU 对负输入可以给出负输出，不是简单地把所有负值置零。

    设每个投影的本地输出宽度为 I_local：
        输入 [..., 2*I_local] -> gate/up 各 [..., I_local] -> 输出 [..., I_local]。
    最后一维缩为一半，前面的 batch、序列长度或 token 数等维度保持不变。
    例：输入 [[1,2,10,20]]，前两维是 gate，后两维是 up，输出约为 [[7.3106,35.2319]]。

    张量并行时，MergedColumnParallelLinear 已在每个 rank 按 [gate_local | up_local]
    排列本地输出；本层只处理相互对应的本地特征，不执行 all_gather 或 all_reduce。
    本层没有可学习的权重/偏置，训练时梯度可以通过 SiLU 和乘法传回两个投影分支。
    """

    def __init__(self):
        # 初始化 nn.Module 的内部状态；本层没有额外参数，也不需要加载 checkpoint 权重。
        super().__init__()

    # 将 forward 交给 torch.compile 编译；编译器可能融合 SiLU 和逐元素乘法以减少开销。
    # 是否融合以及实际加速取决于后端和输入；该装饰器不保证一定生成一个 GPU kernel。
    # 首次调用通常包含编译/预热开销，输入形状等条件变化还可能触发重新编译。
    # 它不会自动把输入搬到 GPU；下面的操作使用传入张量所在的设备。
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chunk(2, -1)：沿最后一个维度将合并输出拆为两块，保持其他维度不变。
        # 输入 x 此时代表 [gate | up]；解包后变量 x 被重绑定为 gate，变量 y 代表 up。
        # 例：[[1,2,10,20]] -> x=[[1,2]]、y=[[10,20]]。
        # chunk 返回原张量的视图，本身不复制投影数据。
        # 调用方应保证最后一维为偶数且 gate/up 等宽；本方法没有显式检查，
        # 奇数长度可能产生不等宽的分块，后续乘法可能报错或发生非预期广播。
        x, y = x.chunk(2, -1)
        # F.silu(x) 对 gate 的每个元素计算 t*sigmoid(t)，然后 * y 逐元素乘以 up。
        # 这里的 * 不是矩阵乘法；两块正常情况下形状一致，输出仍为 [..., I_local]。
        # 例：SiLU(1)*10≈7.3106，SiLU(2)*20≈35.2319。
        # F.silu 默认不是原地操作，本句也不会原地修改输入的 gate/up 数据。
        # 输出可以直接交给 down_proj；张量并行下由后续行并行层汇总各 rank 的贡献。
        return F.silu(x) * y

if __name__ == "__main__":
    # Example usage
    layer = SiluAndMul().cuda()
    input_tensor = torch.randn(8, 4000, 8000).cuda()  # Example input tensor with shape (8, 4000, 8000)
    
    for _ in range(10):  # Warm-up iterations
        _ = layer(input_tensor)

    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
