from modelscope.hub.snapshot_download import snapshot_download

# local_dir: 文件直接下载到该目录（平铺），不再生成 models/xxx--yyy/snapshots 嵌套结构
# 下载新模型时改 model_id 和 local_dir 即可
snapshot_download(
    model_id="Qwen/Qwen3-0.6B",
    local_dir="/root/autodl-tmp/Models/Qwen3-0.6B",
)
