import torch
import time 

# 此类虽然命名为 LayerNorm，实际实现的是 RMSNorm，没有减均值，也没有可学习的偏置 beta。
# 标准 LayerNorm 用 (x - mean(x)) / sqrt(mean((x - mean(x))²) + eps) 再做缩放和偏移。
# 本类用 x / sqrt(mean(x²) + eps) 再乘 gamma：按最后一维调整每个向量的尺度。

# 输入可以是 [N, D]、[B, L, D] 等形状，输出保持原形状；不同 token 的统计量互不混合。
# 在 Q/K 归一化中，最后一维也可以是 head_dim；gamma 的长度应与输入最后一维一致。
# 以下例子：x = [[3, 4], [0, 2]]，形状 [2, 2]；gamma = [1, 2]，eps = 1e-5。
class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        # 初始化 nn.Module，使后面赋值的 Parameter 能注册到模块中。
        super().__init__()
        # Use nn.Parameter to make gamma learnable and loadable from checkpoints
        # gamma 是每个特征维度的缩放系数，本例形状为 [2]，不同 token 共用这两个系数。
        # detach 切断与传入 gamma 原有计算图的关联；clone 创建独立存储，不与传入张量共享数据。
        # 包装为 Parameter 后注册名为 weight，默认可训练，也便于加载 checkpoint 中的 *.weight。
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        # 分母里加一个小正数；即使某行全零，分母也为 sqrt(eps)，避免直接除以零。
        self.eps = eps

    @property
    def gamma(self):
        """Backward compatibility: gamma alias for weight"""
        # layer.gamma 和 layer.weight 返回同一个参数，不是复制或额外注册一份 gamma。
        return self.weight

    # 用 torch.compile 编译此计算方法，便于优化张量运算；首次编译可能有开销，数学公式不变。
    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ
        # x.pow(2) = [[9, 16], [0, 4]]。
        # dim=-1 表示沿每行的特征维度求均值，不是将两个 token 的所有元素一起求均值。
        # 两行均方分别为 (9+16)/2 = 12.5、(0+4)/2 = 2。
        # keepdim=True 保留被归约的维度：结果为 [[12.50001], [2.00001]]，形状 [2, 1]。
        # 变量名 variance 在这里表示“均方 + eps”，并不是减去均值后计算的统计方差。
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        # sqrt_variance ≈ [[3.535535], [1.414217]]，每个 token 有自己的 RMS 分母。
        sqrt_variance = variance.sqrt()
        # 广播规则：[2, 2] / [2, 1]，同一行的两个特征除以该行的分母。
        # x / sqrt_variance ≈ [[0.848528, 1.131370], [0, 1.414210]]。
        # 再乘 weight=[1, 2]：第 0 个特征乘 1，第 1 个特征乘 2，且两行都使用相同的权重。
        # 最终 x_norm ≈ [[0.848528, 2.262741], [0, 2.828420]]，形状仍为 [2, 2]。
        # 这是逐特征缩放，不是矩阵乘法；输出不保证均值为零，也不限制在 [-1, 1]。
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 有残差时先相加，再归一化；模型调用中 x 和 residual 具有相同的形状。
        # 例：传入 x = [[1, 2], [0, 1]]，residual = [[2, 2], [0, 1]]。
        # 相加得到 [[3, 4], [0, 2]]，恰好是上面的 RMSNorm 示例输入。
        # 此处 x 重新绑定到相加结果，没有原地修改调用方传入的 x 或 residual。
        x = x + residual
        # 实际返回一个二元组 (归一化结果, 未归一化的相加结果)：
        #   第一个张量 ≈ [[0.848528, 2.262741], [0, 2.828420]]，供后续注意力或 MLP 计算。
        #   第二个张量 = [[3, 4], [0, 2]]，供调用方保存为更新后的 residual，继续残差累加。
        # 所以调用方可写 normalized_x, updated_residual = layer(x, residual)。
        return self.rms_forward(x), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # 调用 layer(x) 或 layer(x, residual) 时，nn.Module 会进入此 forward，按残差是否存在分支。
        # 注意：虽然返回类型标注为 Tensor，带 residual 的分支实际返回两个 Tensor 组成的 tuple。
        if residual is not None:
            # layer([[1, 2], [0, 1]], residual=[[2, 2], [0, 1]])：先相加，返回 (RMSNorm(x+residual), x+residual)。
            return self.residual_rms_forward(x, residual)
        else:
            # layer([[3, 4], [0, 2]])：仅返回归一化后的单个 Tensor，不返回 residual。
            return self.rms_forward(x)

if __name__ == "__main__":
    # Example usage
    x = torch.randn(8,4000,8000).cuda()
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)
    layer = LayerNorm(gamma=gamma).cuda()
    residual = torch.full_like(x,fill_value=1)

    for _ in range(10): # Warm-up iterations
        _ = layer(x)
    
    # Without residuals
    times = [] 
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    # With residuals
    times.clear()
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x,residual)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
    
