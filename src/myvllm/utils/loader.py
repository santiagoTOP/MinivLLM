import torch
from torch import nn
import os
from safetensors import safe_open
from transformers import AutoConfig
import re


def default_weight_loader(param, weight):
    """Default weight loader that copies weight data to parameter."""
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch: param {param.shape} vs weight {weight.shape}")
    param.data.copy_(weight)


def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str):
    """
    Load weights from a Hugging Face model checkpoint into the custom model.
    Handles QKV and gate_up weight merging for optimized layers.

    Args:
        model: The target model to load weights into
        model_name_or_path: Path to local checkpoint or Hugging Face model name
    """
    from huggingface_hub import snapshot_download

    # Try to resolve the path - could be local or from HF cache
    checkpoint_path = None

    # First, try local paths
    if model_name_or_path.startswith('~'):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    # If not a local path, try to download from HuggingFace
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"]  # Skip non-safetensors weights
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Load all safetensors files in the checkpoint directory
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Collect all weights from HF model
    hf_weights = {}
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for weight_name in f.keys():
                hf_weights[weight_name] = f.get_tensor(weight_name)

    # Now map and load weights into custom model
    loaded_params = set()
    skipped_params = []

    # Process each HF weight
    for hf_name, hf_weight in hf_weights.items():
        try:
            # 1. Handle QKV merge (q_proj + k_proj + v_proj → qkv_projection)
            if '.self_attn.q_proj.weight' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    k_name = hf_name.replace('q_proj', 'k_proj')
                    v_name = hf_name.replace('q_proj', 'v_proj')

                    if k_name in hf_weights and v_name in hf_weights:
                        q_weight = hf_weight
                        k_weight = hf_weights[k_name]
                        v_weight = hf_weights[v_name]

                        # Concatenate q, k, v along output dimension
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

                        custom_name = f"model.layers.{layer_idx}.self_attn.qkv_projection.weight"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(qkv_weight)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(k_name)
                            loaded_params.add(v_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 2. 将 checkpoint 中分开的 gate_proj/up_proj 权重加载到模型的 gate_up 参数。
            # 此时文件中的张量已读入 hf_weights；本分支负责名称映射、布局转换和参数写入。
            # 原始模型：两次投影 gate=x @ W_gate.T、up=x @ W_up.T。
            # 本项目的 MergedColumnParallelLinear 将两份权重合并，以一次线性运算得到 gate/up。
            # 以下用 hf_name="model.layers.3.mlp.gate_proj.weight" 跟踪同一层的加载过程。
            elif '.mlp.gate_proj.weight' in hf_name:
                # 从参数名提取层编号：layers\. 匹配字面上的 "layers."，(\d+) 捕获后面的数字。
                # 示例匹配 "layers.3"；若未找到层编号，则不进入下面的加载逻辑。
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    # group(1) 取第一个捕获组，示例得到字符串 "3"，用于定位目标模型的第 3 层。
                    layer_idx = layer_match.group(1)
                    # 仅替换投影名称，保留层编号和其余路径，得到同一层的 up 参数名：
                    # "model.layers.3.mlp.up_proj.weight"。
                    up_name = hf_name.replace('gate_proj', 'up_proj')

                    # 只有对应 up 权重也存在时才能合并；缺少它时，本分支不会执行参数写入。
                    if up_name in hf_weights:
                        # hf_weight 是外层遍历 hf_weights.items() 当前拿到的 gate 权重。
                        # up_weight 则通过同层 up 参数名从字典中取出。
                        # 设 H=hidden_size、I=intermediate_size，两份完整权重通常都是 [I,H]。
                        gate_weight = hf_weight
                        up_weight = hf_weights[up_name]

                        # PyTorch 权重布局为 [输出特征, 输入特征]，dim=0 沿输出维度上下拼接。
                        # 合并后形状 [2I,H]，顺序为 [W_gate; W_up]，不是逐元素相加。
                        # 例：W_gate=[[1,0],[0,1]]，W_up=[[10,0],[0,10]]，则合并后为
                        # [[1,0],[0,1],[10,0],[0,10]]；输入 x=[[1,2]] 得到 [[1,2,10,20]]。
                        # 一次 x @ gate_up_weight.T 等价于沿最后一维拼接两个独立投影的结果。
                        # 输出的前 I 维是 gate、后 I 维是 up；SiluAndMul 按这个顺序拆成两半，
                        # 再计算 SiLU(gate)*up，因此权重拼接顺序必须与输出拆分顺序一致。
                        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

                        # 将 checkpoint 的两个源参数名映射到本项目的一个目标参数名。
                        # 示例目标："model.layers.3.mlp.gate_up.weight"。
                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.weight"
                        try:
                            # 按完整属性路径查找模型中已创建的 nn.Parameter，并不创建新层或新参数。
                            param = model.get_parameter(custom_name)
                            # 真正的加载操作：将合并权重原地复制到目标参数的数据存储中。
                            # 这里不执行前向计算，也不会自动调用参数上附加的 weight_loader。
                            # 对当前 MergedColumnParallelLinear，P=tp_size=1 时目标也是 [2I,H]。
                            # P>1 时本地目标是 [2I/P,H]，完整源仍是 [2I,H]，通常会形状不匹配。
                            # 多进程正确布局应为每个 rank 各自的 [gate_local; up_local]，
                            # 不能将完整 [W_gate; W_up] 直接均分，否则可能拆开两个投影。
                            # 若改为支持分片加载，应以以下两次调用替代直接 copy_：
                            #   param.weight_loader(param, gate_weight, loaded_weight_id=0)
                            #   param.weight_loader(param, up_weight, loaded_weight_id=1)
                            # 该子类方法会按 rank 选源分片，并写到本地 gate/up 各自对应的位置。
                            param.data.copy_(gate_up_weight)
                            # 复制成功后才登记：目标参数已加载，两个 checkpoint 源参数已使用。
                            # loaded_params 只是状态集合，用于后续加载统计和跳过已合并的 up 参数，
                            # 不保存权重张量，也不参与模型前向计算。
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(up_name)
                        except AttributeError:
                            # get_parameter 找不到目标路径时记录跳过原因，供加载总结报告。
                            # copy_ 的形状不匹配通常抛 RuntimeError，不由此处的 AttributeError 捕获；
                            # 它会进入外层 except Exception，记录当前 hf_name 的加载错误。
                            skipped_params.append((custom_name, "Parameter not found"))

            # 3. Handle gate_up merge for bias (if present)
            elif '.mlp.gate_proj.bias' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_bias_name = hf_name.replace('gate_proj', 'up_proj')

                    if up_bias_name in hf_weights:
                        gate_bias = hf_weight
                        up_bias = hf_weights[up_bias_name]
                        gate_up_bias = torch.cat([gate_bias, up_bias], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.bias"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(gate_up_bias)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(up_bias_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 4. Skip k_proj, v_proj, up_proj (already merged)
            elif any(x in hf_name for x in ['.k_proj.', '.v_proj.', '.up_proj.']):
                if hf_name not in loaded_params:
                    skipped_params.append((hf_name, "Merged into qkv_projection or gate_up"))

            # 5. All other parameters: load directly (names match HF)
            else:
                try:
                    param = model.get_parameter(hf_name)
                    if param.shape != hf_weight.shape:
                        # Handle vocab size mismatch for embeddings/lm_head
                        if len(param.shape) > 0 and len(hf_weight.shape) > 0:
                            min_size = min(param.shape[0], hf_weight.shape[0])
                            param.data[:min_size].copy_(hf_weight[:min_size])
                        else:
                            param.data.copy_(hf_weight)
                    else:
                        param.data.copy_(hf_weight)
                    loaded_params.add(hf_name)
                except AttributeError:
                    skipped_params.append((hf_name, "Parameter not found"))

        except Exception as e:
            skipped_params.append((hf_name, f"Error: {str(e)}"))

    # Check for model parameters that weren't loaded
    unloaded_params = []
    for name, param in model.named_parameters():
        if name not in loaded_params:
            unloaded_params.append(name)

    print(f"\n{'='*80}")
    print(f"Weight Loading Summary:")
    print(f"{'='*80}")
    print(f"Successfully loaded: {len([p for p in loaded_params if not any(x in p for x in ['.k_proj.', '.v_proj.', '.up_proj.'])])} parameter groups")

    if unloaded_params:
        print(f"\n⚠️  WARNING: {len(unloaded_params)} model parameters NOT loaded from checkpoint:")
        for name in unloaded_params[:15]:
            param = dict(model.named_parameters())[name]
            print(f"  - {name} (shape: {param.shape}, mean: {param.data.mean():.6f})")
        if len(unloaded_params) > 15:
            print(f"  ... and {len(unloaded_params) - 15} more")

    if skipped_params:
        # Group skipped by reason
        merged_skips = [s for s in skipped_params if "Merged" in s[1]]
        not_found_skips = [s for s in skipped_params if "not found" in s[1]]
        no_mapping_skips = [s for s in skipped_params if "No mapping" in s[1]]

        if merged_skips:
            print(f"Skipped (merged into other weights): {len(merged_skips)}")
        if not_found_skips:
            print(f"Skipped (not found in model): {len(not_found_skips)}")
            for name, reason in not_found_skips[:5]:
                print(f"  - {name}")
            if len(not_found_skips) > 5:
                print(f"  ... and {len(not_found_skips) - 5} more")
        if no_mapping_skips:
            print(f"Skipped (no mapping rule): {len(no_mapping_skips)}")
            for name, reason in no_mapping_skips[:5]:
                print(f"  - {name}")
            if len(no_mapping_skips) > 5:
                print(f"  ... and {len(no_mapping_skips) - 5} more")

    print(f"{'='*80}")
    return loaded_params
