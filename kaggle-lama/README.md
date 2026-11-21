# Kaggle 漫画消字批处理

把 Saber 检测 + LaMA 修复推理放到 Kaggle GPU 执行。调用方准备图片、提交私有任务，等待并下载该任务版本的输出。任务处理完就退出；不用保持本机 API、浏览器或本机电脑在线，不需要 Tunnel。

本地仅有图片解码校验、打包和下载，不加载模型。

## 首次部署（每个使用者做一次）

- 用你自己的 Kaggle 账号和 Key（`--key-file` 或环境变量提供）。
- 按 `../README.md` 的“准备”一节跑 `build_runtime.py` +
  `publish-runtime`，生成你自己的 `deployment.json`
  （含你自己的私有运行环境 Dataset 名字与 SHA256）。
- `deployment.json` 固定运行环境的 SHA256；每次调用只上传新的图片任务。
- Saber 固定提交 `13491d83c0f0cb974ed62a6ba662d164f721f24e`，沿用四个模型和本仓库的 `engine/lama_engine.py`。校验清单见 `runtime-manifest.json`。
- 使用 `NvidiaTeslaT4` GPU。代码串行使用 CUDA:0，不叠加两块 GPU 的显存。

## 调用方快速使用

需要 Python 3.11 或更新版本，Windows/Linux 均可。以下命令在本目录执行；Windows 可把 `python` 换成虚拟环境解释器的完整路径。

```text
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-client.txt
```

Linux 使用 `.venv/bin/python`。后续示例以 `python` 代指安装好依赖的解释器。

凭证优先通过调用方已有的 Kaggle 环境变量/配置提供。也可在需要联网的命令中使用 `--key-file`，支持原始 `KGAT_…` 文本或旧版 `kaggle.json`。凭证只在调用方进程内读取，绝不上传到 Dataset/Notebook。参数只传文件路径，不传 Token 内容。

```text
python client.py --key-file /path/to/kaggle.txt account
python client.py prepare --input /path/to/chapter.cbz --output .work/chapter-001
python client.py --key-file /path/to/kaggle.txt submit --job .work/chapter-001/job.json
python client.py --key-file /path/to/kaggle.txt status --job .work/chapter-001/job.json
python client.py --key-file /path/to/kaggle.txt wait --job .work/chapter-001/job.json
python client.py --key-file /path/to/kaggle.txt fetch --job .work/chapter-001/job.json --output .work/chapter-001-result
```

`prepare` 接受图片文件夹（递归）或 ZIP/CBZ，按相对路径排序，保留源文件名和原始图片字节。每个任务使用新目录；`job.json` 必须保留，它记录任务与远端资源的对应关系。`submit` 返回后远端独立执行；轮询或下载可以稍后在其他进程中完成。任务提交前的上传阶段仍需要调用方在线。

每个任务创建唯一私有 input Dataset 和唯一私有 Notebook，固定只提交一个版本，防止并发任务和 SDK 的“最新版本”行为串结果。勿在网页编辑、重跑或发布这些任务 Notebook。若需要重跑，请重新 `prepare` 创建新任务。

`wait` 默认最多等待 12 小时，每 30 秒查询一次，仅在状态变化时输出。超时可稍后继续；不会取消或重新提交。查看远端日志：

```text
python client.py --key-file /path/to/kaggle.txt logs --job .work/chapter-001/job.json
```

## 参数与输出

可选 `prepare --options options.json`：

```json
{
  "defaults": {"mask_dilate_size": 10, "mask_box_expand_ratio": 20},
  "pages": {
    "chapter/005.jpg": {"extra_boxes": [[50, 70, 150, 300]]},
    "chapter/020.jpg": {"mask_dilate_size": 12}
  }
}
```

`pages` 的键必须是输入中的完整相对路径（ZIP 内使用 `/`），参数覆盖 defaults。默认值、范围和处理语义沿用现有 API：膨胀大小整数 0–64，扩框百分比 0–100，补框最多 64 个整数矩形，按图像尺寸裁剪后必须有面积。补框只参与掩膜区域筛选，不保证强制擦除；分割未覆盖的文字依然可能保留，也不会增加返回的 YOLO boxes。

下载目录主要有：

```text
job-result.json       # 汇总、逐页记录、模型/运行环境来源、结果 ZIP 校验和
results.zip
  job-result.json     # 同一汇总，但不含 ZIP 自身的 archive 字段
  p000001/result.json
  p000001/cleaned.png
  p000001/mask.png
  p000001/precise-mask.png
  ...
notebook.log          # UTF-8 远端日志
```

页记录通过 `source_name` 对应原图，包含原图 SHA256、宽高、实际参数、耗时、`boxes` 和输出文件的大小/SHA256。`boxes` 保留 `x1/y1/x2/y2/score/polygon/angle`。`angle` 是长边方向轴角，API 层归一化到 `[-90,90)`，不能直接当成嵌字旋转角。

这是批处理文件协议，**不是 HTTP API 的直接替换**：三个 base64 内容改存 PNG。页记录中 `files.inpainted_image_b64`、`files.mask_b64`、`files.precise_mask_b64` 的值为 `{path, sha256, bytes}`，旧字段名用于对应原 API，值不再是 base64 字符串。

最终 `mask.png` 为 L 模式二值图（255=修复，0=保留）；`precise-mask.png` 为 L 模式灰度分割图，不要擅自二值化。修复图维持原图尺寸，掩膜外像素不修改。

Notebook 的 `complete` 仅表示程序结束。初始化失败可能只输出带 `error` 的 `job-result.json`；单页失败会明确记录且继续后续页。汇总状态有 `completed`、`partial`、`failed`。`fetch` 会核验任务 ID、运行环境、页数、汇总一致性、ZIP 和每个文件的校验和；仅 `completed` 返回退出码 0，`partial/failed` 返回非零，不能当全部成功。下载目标必须是新目录；下载中断后使用另一个新目录重试。

## 限制与清理

- 输入只接受非动画 PNG/JPEG，每页 ≤20 MiB、≤3000 万像素；每任务最多 5000 页。
- 提交前按原图尺寸保守估算输出，超过 18 GB 要拆批；远端还有实际输出大小保护。Notebook 通常有 20 GB 保存输出上限，运行环境放在 `/tmp`，避免把模型重复保存为输出。
- Kaggle 有 GPU 配额、排队及单次时长限制，不能视为有即时响应保证的 API。超大区域仍沿用 `disable_resize=True`，可能因显存不足失败，不暗中缩图降低效果。
- GPU/驱动/底层库差异会带来少量数值、坐标取整或修复像素差异；不承诺跨硬件逐字节相同。结果 provenance 记录实际 Python、GPU、Torch 和主要依赖版本。
- 输入占私有 Dataset 空间；运行成功的输出保存在该 Notebook 任务版本中，关闭会话后可以再次下载。不要依赖运行时临时目录保存数据。
- 不自动删除远端资源。调用方确认下载验证完成后自行清理；以下命令先预览，仅 `--yes` 才删除**这个任务的 Notebook 及 input Dataset**，不会删除共享 runtime。

```text
python client.py --key-file /path/to/kaggle.txt cleanup --job .work/chapter-001/job.json
python client.py --key-file /path/to/kaggle.txt cleanup --job .work/chapter-001/job.json --yes
```

若提交发生网络中断，先根据 `job.json` 中的唯一资源名检查状态；处于 `submitting` 时工具拒绝盲目重复提交，以免重复消耗 GPU。若远端 Notebook 已存在，通过同一 job.json 查询和下载即可；若确认没有创建成功，再建一个新任务。上传/提交失败遗留的资源名也保存在 job.json，调用方可在 Kaggle 网页手动清理。

## 维护：重建运行环境

日常调用不需要此步骤。仅在有意升级引擎/模型时，构建并发布新的私有 Dataset，不覆盖当前共享版本：

```text
python build_runtime.py --saber-root /path/to/Saber-Translator --engine ../engine/lama_engine.py --output .work/new-runtime
python client.py --key-file /path/to/kaggle.txt publish-runtime --bundle .work/new-runtime --output .work/new-deployment.json
python client.py --key-file /path/to/kaggle.txt submit --job .work/new-job/job.json --deployment .work/new-deployment.json
```

构建器只打包固定 Git 提交中的 `src`、LICENSE、指定引擎和四个模型，不打包仓库凭证或用户配置。模型仍归原作者所有。`.bin` 是 ZIP 格式容器，用此扩展名避免 Kaggle 自动解压改变布局。Notebook 通过 SHA256 检查后，在临时目录解包模型和源代码，保留 Saber 对资源目录的校验。

离线测试：`python -m unittest discover -s . -p test_batch.py`。GPU 验证结果见 `GPU_VALIDATION.md` 和 `gpu-validation.json`。

平台参考：[Notebook 文档](https://www.kaggle.com/docs/notebooks)、[官方 CLI](https://github.com/Kaggle/kaggle-cli)。配额与运行镜像由平台控制，以账号页面和结果 provenance 为准。
