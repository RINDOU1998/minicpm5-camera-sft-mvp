# PyTorch 2.7.1 下载超时：镜像与代理

先在服务器仓库根目录激活 `.venv` 环境。以下镜像方案针对 Linux x86_64、Python 3.11、glibc ≥ 2.28；保留 PyTorch 2.7.1 与 CUDA 12.6 的实验版本。

## 首选：从国内 PyPI 镜像安装

PyPI 的 `torch==2.7.1` 在 Linux x86_64 上依赖 CUDA 12.6 运行库，因此本轮可以直接用 PyPI 镜像。不要在下面的 PyPI 镜像命令中改写成 `2.7.1+cu126`；那个版本标记属于另一条发布索引路径。

```bash
source .venv/bin/activate
python -m pip install torch==2.7.1 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120 --retries 10
```

若清华源连接仍不稳定，改用阿里云，二选一：

```bash
python -m pip install torch==2.7.1 \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --timeout 120 --retries 10
```

这里替换的是 `--index-url`，不是只添加 `--extra-index-url`。超时参数表示网络读超时，不是整次安装的总时限。torch wheel 本身约 821 MB，另有 CUDA 依赖需要下载；如果旧 pip 下载中断，可先从同一镜像升级 pip 再重试，保留其正常缓存。

其他训练依赖也可以从同一镜像安装：

```bash
python -m pip install -r requirements-train.txt \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120 --retries 10
```

## 备用：通过 SSH 使用你电脑已有的代理

服务器的 `127.0.0.1` 指向服务器本身，不能直接填你电脑的本地代理端口。若你电脑上的 HTTP 代理可访问 PyTorch 下载站，可用 SSH 反向转发。需要服务器允许 SSH TCP 转发。

在**你的电脑**终端执行，替换三个大写占位值；`LOCAL_PROXY_PORT` 是电脑上实际 HTTP/混合代理端口，`SSH_PORT` 是服务器 SSH 端口：

```bash
ssh -p SSH_PORT -N -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:17890:127.0.0.1:LOCAL_PROXY_PORT USER@SERVER
```

若在 Windows PowerShell 执行，将上述命令写成一行，不能使用 Bash 的反斜杠换行：

```powershell
ssh -p SSH_PORT -N -o ExitOnForwardFailure=yes -R 127.0.0.1:17890:127.0.0.1:LOCAL_PROXY_PORT USER@SERVER
```

保持该终端运行。在**服务器**的已激活环境中执行：

```bash
python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126 \
  --proxy http://127.0.0.1:17890 --timeout 120 --retries 10
```

安装完在电脑端 Ctrl+C 结束隧道。该例只将代理监听在服务器回环地址，不需要开放公网代理端口。代理方案使用你已有的可用代理，不提供未经验证的公共代理地址。

## 安装后确认版本与 GPU

```bash
python -m pip check
python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.__version__.split('+')[0] == '2.7.1', 'Unexpected torch version'
assert torch.version.cuda == '12.6', 'CPU wheel or unexpected CUDA build'
assert torch.cuda.is_available(), 'CUDA GPU unavailable; inspect driver and visible devices'
assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'
print('GPU:', torch.cuda.get_device_name(0), 'BF16: supported')
PY
```

若安装提示 `Requirement already satisfied`，仍需执行上述检查；已有 CPU 或其他 CUDA 构建不应被当作本轮基线。确认属于错误构建时，先在这个专用环境内 `python -m pip uninstall -y torch`，再从上述镜像安装。

## 本次核对证据

2026-10-19 从本机读取 [PyPI 2.7.1 发布元数据](https://pypi.org/pypi/torch/2.7.1/json)，确认 Linux x86_64 依赖 `nvidia-cuda-runtime-cu12==12.6.77`；[清华索引](https://pypi.tuna.tsinghua.edu.cn/simple/torch/) 和 [阿里云索引](https://mirrors.aliyun.com/pypi/simple/torch/) 均列出 `torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl`，索引 SHA256 与 PyPI 一致，两个镜像的 wheel HEAD 请求均返回 200、长度 821174704 字节。

SHA256：`06eea61f859436622e78dd0cdd51dbc8f8c6d76917a9cf0555a333f9eac31ec1`。此次没有下载完整 wheel，也未在你的服务器执行安装或验证吞吐；镜像索引与文件可达不等于已经完成服务器安装。
