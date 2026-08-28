# LLM 实践训练营

这是一个按“调用 API → 设计 Prompt → 本地推理 → SFT/LoRA → 偏好对齐”逐步深入的代码训练路径。每章只有一个核心目标，示例尽量短小，并共享一致的命令行和配置方式。

## 学习地图

| 章节 | 你会学到 | 入口 |
| --- | --- | --- |
| 01 | 使用 OpenAI 兼容的 LLM API，配置环境变量和处理错误 | [`chapters/01_api/chat.py`](chapters/01_api/chat.py) |
| 02 | zero-shot、few-shot、角色/约束 Prompt 与分步推理 | [`chapters/02_prompting/prompts.py`](chapters/02_prompting/prompts.py) |
| 03 | 从 Hugging Face 下载模型并在本地推理 | [`chapters/03_local_inference/download.py`](chapters/03_local_inference/download.py) |
| 04 | 使用 TRL + PEFT 对模型进行 LoRA SFT | [`chapters/04_sft_lora/train.py`](chapters/04_sft_lora/train.py) |
| 05 | 使用 DPO 做偏好对齐，并理解 PPO 的训练环路 | [`chapters/05_alignment/dpo_train.py`](chapters/05_alignment/dpo_train.py) |

> 建议按顺序完成。前三章可在 CPU 上体验；第四、五章建议使用 CUDA GPU。训练前先用小数据集、小 `max_steps` 跑通流程。

## 1. 安装

项目要求 Python 3.10+。先创建虚拟环境，然后按学习进度安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[api]'

# 本地推理
pip install -e '.[inference]'

# SFT / DPO（包含 inference 依赖）
pip install -e '.[train]'
```

复制环境变量模板：

```bash
cp .env.example .env
set -a; source .env; set +a
```

不要提交真实的 API Key 或 Hugging Face Token。

## 2. 按章节练习

### 01 — 调用 LLM API

```bash
export OPENAI_API_KEY='...'
python chapters/01_api/chat.py '用三句话解释什么是 Transformer'
```

`OPENAI_BASE_URL` 可选，因此该示例也可连接实现 OpenAI Responses API 的兼容网关。代码将 system instruction 与用户输入分开，Key 只从环境变量读取。

**练习：** 增加 `--temperature`；将输出保存为 JSONL；给网络错误增加指数退避重试。

### 02 — Prompt 构造

先查看四种 Prompt，无需 API Key：

```bash
python chapters/02_prompting/prompts.py --pattern few_shot
python chapters/02_prompting/prompts.py --pattern structured_reasoning --run
```

- **zero-shot**：直接描述任务和输出格式。
- **few-shot**：提供少量输入/输出示例，帮助模型模仿边界与格式。
- **role + constraints**：明确角色、约束、验收标准。
- **structured reasoning**：要求模型先分析再给结论；生产环境通常更适合请求“简短依据/可验证步骤”，而不是依赖冗长的隐藏思维过程。

**练习：** 为同一批输入比较四种 Prompt，把“格式正确率”作为自动评测指标。

### 03 — 下载并本地推理

```bash
python chapters/03_local_inference/download.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen-0.5b

python chapters/03_local_inference/infer.py \
  --model-path models/qwen-0.5b --prompt '用一句话介绍杭州'
```

下载脚本使用 `snapshot_download`，支持断点续传和固定 `--revision`。推理脚本优先使用 CUDA，并通过 tokenizer 的 chat template 组织对话。首次练习应选择小模型；使用模型前请阅读其模型卡和许可证。

### 04 — SFT + LoRA

示例数据为 JSONL，每行包含一个 `messages` 数组：

```json
{"messages":[{"role":"user","content":"1+1 等于几？"},{"role":"assistant","content":"1+1 等于 2。"}]}
```

运行一个短训练：

```bash
# 先在 CPU 上检查格式，避免启动训练后才发现坏数据
python tools/validate_data.py sft data/sft_demo.jsonl

python chapters/04_sft_lora/train.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/sft_demo.jsonl --output-dir outputs/sft-lora \
  --max-steps 20
```

LoRA 只训练低秩适配器，显著减少可训练参数。真实项目中需要划分训练/验证集，检查数据授权与隐私，并比较基座模型和微调模型的离线评测结果。

### 05 — DPO / PPO 对齐

DPO 数据每行包含 `prompt`、`chosen` 和 `rejected`：

```bash
python tools/validate_data.py preference data/preferences_demo.jsonl

python chapters/05_alignment/dpo_train.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/preferences_demo.jsonl --output-dir outputs/dpo \
  --max-steps 20
```

DPO 直接从偏好对学习，工程链路通常比 PPO 短。PPO 则需要策略模型、参考模型、奖励模型和价值模型，并循环执行“生成 → 奖励打分 → 优势估计 → 策略更新”。[`ppo_workflow.py`](chapters/05_alignment/ppo_workflow.py) 给出了不依赖框架版本的最小训练环路骨架；把它接入具体 TRL 版本前，应以该版本文档为准。

## 3. 推荐项目节奏

1. **先跑通**：使用 demo 数据和 20 steps 确认数据、显存和保存流程。
2. **再建立基线**：定义任务指标，并保存基座模型结果。
3. **再改一个变量**：每次只修改 Prompt、数据或超参数中的一类。
4. **记录实验**：至少记录代码版本、模型 revision、随机种子、数据版本与指标。
5. **最后做安全检查**：人工抽检事实性、偏见、隐私泄漏和拒答边界。

## 4. 代码质量检查

```bash
python -m unittest discover -s tests -v
python -m compileall chapters tools tests
```

## 目录结构

```text
chapters/       每章可独立执行的示例
data/           可直接跑通流程的微型数据
tests/          不需要 GPU/API 的单元测试
tools/          训练前即可运行的数据质量工具
```
