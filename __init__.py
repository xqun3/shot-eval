"""shot_eval —— 可独立发布的逐分镜片段视频评审模块。

抽取的边界
----------
上游那个包做三件事：整片评估（runner / bench / scoring / narration）、
评测集管理（dataset 的 EvalCase / 目录发现 / 上传）、以及逐片段评审。
这里只带走**第三件**，其余一概不要：

* 不要 `EvalCase` —— `judge_scene()` 本来就只需要一个 `style_lock` 字符串，
  上游特意把它解开过（见 judge.py 的注释）。跟着 EvalCase 走等于要求
  「必须先有成片 + state.json」，而我们手上就是一堆独立的单段 mp4。
* 不要 `media.py` 的切片 —— 片段已经是独立文件，不用从成片里按时间窗切。
* 不要 12 板块的百分制 `scoring.py` —— 那是给整片打总分的，逐片段用不上。
  本模块自带 `scoring.py`，用 8 个块 / 72 总权重的加权诊断总分，详见 README。
* 不要 `core.config` —— 只用到三个值，就地重写（见 `config.py`）。

**基础六维判据来自上游**；本模块 v3 另外扩展四模式渲染盲判与链路归因/Prompt
优化模板。判据即口径，报告用 `version_tag()` 的版本和内容哈希阻止跨口径误比较。

模块布局
--------
| 文件 | 来源 | 说明 |
|---|---|---|
| `config.py`   | 重写 | vertex project/location + 默认评审模型 |
| `prompts.py`  | 复制 | 只改了 `prompts.yaml` 的查找方式 |
| `prompts.yaml`| 扩展 | 上游六维 + 四模式盲判 + 归因与优化模板 |
| `models.py`   | 复制 | `SceneSlice` + 所有 pydantic 判决 schema |
| `judge.py`    | 复制 | `judge_scene` / `judge_render_blind` / 计分 |
| `bench.py`    | 复制 | `SceneTask` / `run_batch` / 聚合 / 报告 / `--expect` |
| `adapters.py` | 新写 | 通用 `video_prompts.json` → `SceneTask` 装载器 |
| `scoring.py`  | 新写 | 8 块 72 权重加权诊断总分，stdlib-only |
| `cli.py`      | 新写 | 命令行入口 |
"""


# 这里**不做** re-export。`judge` 一被导入就会拉进 google-genai，
# 而 `python -m shot_eval --help` 不该为了打印帮助去装 SDK；
# 各模块之间也就不必绕着包的部分初始化状态互相引用。
# 用法：`from shot_eval import adapters, bench, judge`。

