# shot-eval

`shot-eval` 是一个面向 AI 生成视频的**逐 shot、多模态、证据驱动评估工具**。

它将每个视频片段与其生成时的分镜设计、视觉节拍、原始口播、风格锁和最终视频 Prompt 对照，输出可定位到时间点的扣分证据、总体诊断分、渲染模式盲判，以及可选的“口播 → plan → Prompt → 视频”根因分析与 Prompt 优化建议。

> 本仓库只包含评估与后评估 Prompt 优化能力；不包含视频生成服务、视频模型权重、真实视频样本、云端凭据或任何运行产物。

## 功能

- 六维逐 shot 评估：画面还原、分镜还原、风格遵循、科学准确、信息清晰、教育价值；
- 独立 Render 盲判：`photoreal` / `cg` / `illustrated` / `vox` / `unclear`；
- 每个扣分项保留：维度、严重度、时间点、画面证据、期望内容；
- 72 权重 Overall 诊断分、覆盖度、Critical / Major / Minor 计数；
- HTML 报告：视频、逐轮判词、扣分跳转、生成模型标签、Prompt 和原始口播；
- 两份完成报告的同页逐 shot 对比；
- 可选链路归因与 grounded Prompt patch；
- Google ADK 后评估优化 Agent：安全应用已锚定 patch，生成重生成计划，但**永不生成付费视频**。

## 安装

需要 Python 3.11 及以上。

```bash
git clone https://github.com/<your-org>/shot-eval.git
cd shot-eval
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[optimizer]"
```

不使用 Prompt 优化 Agent 时，可以只安装核心依赖：

```bash
pip install -e .
```

复制环境示例：

```bash
cp .env.example .env
```

完整云端认证与部署方式见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 输入目录约定

一次评估 run 需要以下最小结构：

```text
runs/<run-name>/
├── video_prompts.json
└── videos/
    ├── A01-01.mp4
    ├── A01-02.mp4
    └── ...
```

`video_prompts.json` 是生成流水线输出的 bundle。至少需要：

```json
{
  "meta": {
    "video_provider": "gemini_omni",
    "video_model": "example-video-model",
    "llm_model": "example-prompt-model",
    "planner_model": "example-planner-model",
    "granularity": "shot"
  },
  "video_prompts": [
    {
      "unit_id": "A01-01",
      "ok": true,
      "request": {"clip_script": "原始口播片段"},
      "scene_design": "...",
      "visual_beat": "...",
      "video_prompt": "...",
      "effective_style_lock": "..."
    }
  ]
}
```

视频文件按 `<unit_id>.mp4` 位于 `videos/`。即使 bundle 中记录的是旧机器的绝对路径，装载器也会回退到当前 `videos/<unit_id>.mp4`。

## 配置 Vertex AI

评估使用 Google Gen AI SDK + Vertex AI。先完成 ADC：

```bash
gcloud auth login
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
export GOOGLE_CLOUD_LOCATION=global
```

也可以使用服务账号：

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/secure/path/service-account.json
export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
```

不要把 `.env`、服务账号 JSON、access token 或 API key 提交到 GitHub。

## 快速开始

### 1. 检查输入与判据，不发送请求

```bash
shot-eval runs/example --dry-run
```

### 2. 小批量冒烟

```bash
shot-eval runs/example \
  --limit 2 \
  --repeats 1 \
  --render off
```

### 3. 正式逐 shot 评估

```bash
shot-eval runs/example \
  --repeats 3 \
  --judge-model gemini-3.8-flash \
  --out runs/example/eval/judge-38flash
```

调用量：

```text
片段数 × repeats × (render=off ? 1 : 2)
```

例如 20 个片段、3 轮、开启 Render 盲判，共约 120 次模型调用。

### 4. 可选：链路归因和 Prompt 优化建议

```bash
shot-eval runs/example \
  --repeats 3 \
  --judge-model gemini-3.8-flash \
  --optimize-prompt \
  --optimization-model gemini-3.8-flash \
  --out runs/example/eval/judge-38flash
```

该模式会在视频评审结束后，对每个 shot 增加一次**纯文本**归因调用：

```text
原始口播 → storyboard plan → generated Prompt → 视频
```

它不会重新生成视频。

## 报告与前端

`--out` 会输出：

```text
runs/example/eval/judge-38flash/
├── shots-<name>-<timestamp>.json
└── shots-<name>-<timestamp>.html
```

HTML 可直接打开；视频按相对路径加载，因此需要保留 `runs/<run-name>/` 的目录结构。

启动本地展示服务：

```bash
shot-eval-serve runs --host 127.0.0.1 --port 8010
```

访问：

```text
http://127.0.0.1:8010/
```

服务首页显示：

- 视频生成模型、视频 Provider、Prompt LLM、Planner LLM；
- 评审模型和评估口径；
- Overall、六维均值/中位数、Render 和归因概览。

### 对比两份报告

访问：

```text
http://127.0.0.1:8010/compare
```

选择任意两份报告后，页面按 `source + scene_id` 一一配对，默认全部展开：

- 左右视频；
- 原始口播；
- 送入视频模型的完整 Prompt；
- 六维分数与差异；
- 每轮观察、判词、合规、Render、扣分证据。

页面会提示评审尺度是否相同。不同视频生成模型是可比较的实验变量，不会被视为评审口径冲突；不同 Judge 模型或判据版本则不应直接比较分数。

## Overall 诊断分

Overall 是排序和诊断分，不是上线门禁。

| 块 | 权重 |
|---|---:|
| 科学准确 | 20 |
| 画面还原 | 12 |
| 信息清晰 | 10 |
| 分镜还原 | 8 |
| 教育价值 | 8 |
| 风格遵循 | 4 |
| Render 匹配 | 6 |
| 画面合规 | 4 |
| 合计 | 72 |

六维按跨 shot **算术平均**计算；Render 使用匹配率；合规使用无字幕/水印/品牌的成功轮次比例。缺失块从覆盖权重中移除后再归一化：

```text
overall = Σ(block_pct × weight) / covered_weight
```

必须同时查看 `critical` 数量和具体证据。大量满分 shot 仍可能掩盖单个致命科学错误。

## Prompt Optimization Agent

可选的 ADK Agent 从完成的评估报告中读取已锚定 patch，安全生成新的 bundle 和重生成计划：

```bash
shot-eval-optimize \
  --run runs/example \
  --eval runs/example/eval/judge-38flash/shots-example-<timestamp>.json \
  --project <your-gcp-project-id> \
  --location global
```

默认安全策略：

- 仅应用多轮评估、置信度足够、anchor 仍匹配的 `valid_patches`；
- 不自动修改 plan、style lock、render mode；
- 不信任 `rejected_patches`；
- 不覆盖原 bundle；
- 输出 bundle 会清空旧 `videos` 映射，避免新 Prompt 错配旧视频；
- 不调用视频生成接口。

详见 [prompt_optimizer_agent/README.md](prompt_optimizer_agent/README.md)。

## 依赖与命令

```bash
shot-eval --help
shot-eval-serve --help
shot-eval-report --help
shot-eval-optimize --help
```

运行测试：

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

## 安全与发布范围

提交 GitHub 前请确认：

```text
不提交 .env
不提交服务账号 JSON
不提交 access token / API key
不提交 runs/、eval/、optimizations/、视频文件
不提交本机绝对路径或内部服务端点
```

`.gitignore` 已忽略常见运行产物和凭据文件。请在 CI 中额外运行秘密扫描，例如：

```bash
grep -RInE '(AIza|sk-|bsk-|PRIVATE KEY)' .
```

## License

本项目的发布许可证由维护者决定。请在公开发布前添加适合组织政策的 `LICENSE` 文件。
