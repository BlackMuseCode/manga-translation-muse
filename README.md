# manga-translation-muse

日文漫画 → 简体中文的完整翻译流程：Kaggle 免费 GPU 消字（LaMA）、
Kaggle Model Proxy 翻译（Claude Sonnet 5）、Saber 排版嵌字、打包 CBZ。

这套流程自 2026 年 10 月起在生产中使用，翻过多部连载漫画。
给另一个 Muse 用时，把 `docs/share-prompt.md` 里那段话直接粘贴给它即可。

## 流程总览

```text
原图（自备，JPG/PNG）
  → 1. 消字：Kaggle GPU 跑 Saber 检测 + LaMA 修复，得到干净底板 + 检测框
  → 2. 翻译：裁图拼册 → Sonnet 5 看图翻译对白 → translations.json
  → 3. 排版：去重叠框 → Saber 渲染嵌字（黑气泡自动白字、溢出自动缩字）
  → 4. 打包：rendered/*.png 打成 [zh].cbz
```

每步都是独立的脚本，可以单独重跑；中间产物都落在章节目录里，
断在哪一步就从哪一步续跑。

## 前置条件

- 一个 Kaggle 账号（免费 GPU 每周约 30 小时，两张 T4），以及它的 API token
  （`kaggle.json` 或 `KGAT_...` 文本）。token 只放你本机，绝不提交进仓库。
- Python ≥ 3.11，7z（解压 Saber 整包：Ubuntu/Debian `sudo apt install p7zip-full`，
  macOS `brew install p7zip`，Windows 从 7-zip.org 安装），约 5 GB 空闲磁盘
  （模型约 650 MB；整包下载约 3.3 GB，解压过程峰值约 4 GB，完成后临时文件自动清理）。
- 原图自备：本仓库不含任何漫画原图，也不含下载器（见“不包含什么”）。

## 准备（只需做一次）

```text
git clone https://github.com/BlackMuseCode/manga-translation-muse
cd manga-translation-muse

# Saber 源码，钉死已验证的提交
git clone https://github.com/MashiroSaber03/Saber-Translator saber
cd saber && git checkout 13491d83c0f0cb974ed62a6ba662d164f721f24e && cd ..

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 下载 4 个推理模型并按 SHA-256 校验（约 3.3 GB 整包 + 284 MB 直接下载）
python fetch_models.py

# 自建 Kaggle 推理环境（私有 Dataset + Notebook 模板，只做一次）
python kaggle-lama/build_runtime.py --saber-root saber --output .work/new-runtime
python kaggle-lama/client.py --key-file ~/kaggle.txt publish-runtime \
    --bundle .work/new-runtime --output .work/new-deployment.json
cp .work/new-deployment.json kaggle-lama/deployment.json
```

`kaggle.txt` 里放你的 Kaggle API token（`KGAT_...` 原文或 `kaggle.json`
内容）。`deployment.json` 是你自己的部署描述，含你自己的 dataset 名字，
不要分享给别人——别人按上面这套自己建。

## 跑一话

约定：章节工作目录在 `~/manga-work/` 下，每话一个目录，
里面放原图 `001.jpg`、`002.jpg`……（按页编号）。下面以 `c001` 为例。

### 1. 消字（Kaggle GPU）

```text
cd kaggle-lama
# 提交（两话同日更新时先一起 prepare + submit，各占一张 T4，再逐话等）
.venv/bin/python client.py --key-file ~/kaggle.txt prepare \
    --input ~/manga-work/c001 --output ~/manga-work/c001/kaggle-job
.venv/bin/python client.py --key-file ~/kaggle.txt submit \
    --job ~/manga-work/c001/kaggle-job/job.json
# 等（每话上限 30 分钟；超时可稍后继续等，不会取消任务）
.venv/bin/python client.py --key-file ~/kaggle.txt wait \
    --job ~/manga-work/c001/kaggle-job/job.json --timeout 1800
# 取回 + 适配（验 SHA-256，转成 out/ 布局）
.venv/bin/python client.py --key-file ~/kaggle.txt fetch \
    --job ~/manga-work/c001/kaggle-job/job.json --output ~/manga-work/c001/kaggle-job/fetch
.venv/bin/python adapt_kaggle_out.py \
    --fetch-dir ~/manga-work/c001/kaggle-job/fetch --out-dir ~/manga-work/c001/out
# 取回验完没问题后，删远端任务（Notebook + input Dataset）
.venv/bin/python client.py --key-file ~/kaggle.txt cleanup \
    --job ~/manga-work/c001/kaggle-job/job.json --yes
```

`out/` 里每页有 `{页}.boxes.json`（检测框，含 polygon/angle）、
`{页}.inpainted.png`（干净底板）、`{页}.mask.png`、
`{页}.precise-mask.png`。只看 Notebook 的 complete 不够：
必须 `job-result.json` 里 `status == completed` 且每页 `ok`。

可选质检：`python work/check_plates.py ~/manga-work c001`
（掩膜外像素一致性 + 逐框墨迹覆盖率粗筛）。

失败处理：某话 submit 失败、wait 超时或 fetch 失败，只对失败页建小任务
重跑（临时目录只放失败页原图，同样四步，adapt 时加 `--force`）；
仍失败的页单独处理，不要整话盲目重提交。
`prepare` 时目录里只能有原图 JPG，不要在已有 out/、crops/、rendered/
的目录上 prepare。

### 2. 翻译

```text
# 裁图：按检测框逐框裁原图拼册（小框放大），只作翻译输入
python work/make_crops.py ~/manga-work c001
# 翻译：Sonnet 5 默认档看图翻，术语表按系列给
python kaggle-lama/translate_sheets.py --key-file ~/kaggle.txt \
    --work-dir ~/manga-work/c001 \
    --terms-file terms/choujin-x-TERMS.md \
    --series-intro "作品是《超人X》，主角是黑原时央与东吾妻，超能力（超人）题材。"
```

产物是 `translations.json`（`{"001": [{"box": 0, "zh": "译文"}]}`）。
拟声词、招牌、标题按提示词跳过（`zh` 留空）；看不清、拿不准的框
自动跳过不入表，记在 `translation-report.json` 里。
失败的图册记在报告的 `failed_sheets` 里，补跑时加 `--sheets <图册名>`。

不用 Model Proxy 时，也可以按 `docs/helper-brief-template.md`
让另一个 AI 照着裁图图册翻，再把译文并进 `translations.json`。

### 3. 排版打包

```text
# 去重叠框：默认只演练，加 --yes 才写盘（定稿修正后不要再跑）
python work/dedupe_overlaps.py ~/manga-work c001
python work/dedupe_overlaps.py ~/manga-work c001 --yes
# 渲染：Saber 嵌字，溢出自动缩字号，黑气泡自动白字
python work/batch_render.py ~/manga-work c001
# 打包
cd ~/manga-work/c001/rendered && zip -q -0 -j "../c001 [zh].cbz" $(ls *.png | sort)
```

渲染字体默认 `/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc`
（Noto Sans CJK Bold），换字体设环境变量 `MANGA_FONT`。

可选：`python work/qc_review.py ~/manga-work/c001` 生成编号叠加图和
未译框裁图，逐页核对译文有没有错位、漏译；`work/annotate_sfx.py`
给气泡内未译音效加中文小字旁注（详见脚本内说明）。

## 目录结构

```text
fetch_models.py        4 个模型下载 + SHA-256 校验
requirements.txt       全部依赖
kaggle-lama/
  client.py            Kaggle GPU 批处理客户端（prepare/submit/wait/fetch/cleanup）
  batch_runner.py      远端任务执行器（client.py 内部调用）
  adapt_kaggle_out.py  取回结果转 out/ 布局
  translate_sheets.py  Model Proxy 翻译（裁图图册 → translations.json）
  model_proxy.py       独立 Model Proxy 客户端（自己的 Kaggle token）
  build_runtime.py     自建推理环境（Saber 指定提交 + 引擎 + 模型打包）
  README.md            Kaggle 部分的详细命令、参数与异常处理
  deployment.json.example  部署描述模板（publish-runtime 后按实填）
  runtime-manifest.json    模型清单与 SHA-256（fetch_models.py 的校验基准）
work/
  make_crops.py        逐框裁图拼册
  check_plates.py      消字后质检
  dedupe_overlaps.py   重叠框合并
  batch_render.py      Saber 嵌字渲染
  annotate_sfx.py      气泡内音效中文旁注
  qc_review.py         编号叠加图 + 未译框裁图复核
engine/lama_engine.py  推理引擎适配器（build_runtime.py 打包用）
saber-src/             Saber 渲染代码摘录（GPL-3.0，见 saber-src/LICENSE）
terms/                 各系列术语表示例（人名译法等）
docs/
  helper-brief-template.md  不用 Model Proxy 时的翻译帮手简报模板
  share-prompt.md           分享给其他 Muse 的提示词
```

## 成本（2026-11 实测）

- Kaggle 免费 GPU：每周约 30 小时（两张 T4 各 15 GB 显存），不绑卡；
  一话消字通常几分钟。注意 GPU 小时每周刷新，过期作废。
- 翻译：Sonnet 5 默认档约 $0.59/话；Model Proxy 每天 $10、每月 $100 上限，
  用自己的 Kaggle 账号额度结算。不要用 `response_format json_object`
 （对 Sonnet 5 会 400），`max_tokens` 须 4096。

## 不包含什么

- 不含漫画原图、下载器、图源：原图自备，请用正版渠道。
- 不含 Windows LaMA API 回落：消字只走 Kaggle。
- 不含上传发布（网盘、Komga、阅读器）：打包出 CBZ 后自行上传。

## 授权

- 本仓库的流水线脚本（`work/`、`kaggle-lama/`、`engine/`、
  `fetch_models.py`）为 MIT 协议（见 LICENSE）。
- `saber-src/` 是 Saber-Translator（MashiroSaber03/Saber-Translator）
  的代码摘录，遵循 GPL-3.0（见 `saber-src/LICENSE`）。
- 模型权重各归原作者：YOLO（MIT，YSGforMTL/YSGYoloDetector）、
  LaMA-manga（MIT，mayocream/lama-manga）、ComicTextDetector（GPL-3.0，
  zyddnys/manga-image-translator）、DBNet（随 Saber 发布包分发）。
  `fetch_models.py` 只从上游下载、不重新分发权重。
