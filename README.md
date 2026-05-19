# AI-Human Classroom Core Validation

基于异构 LLM Ensemble 的课堂转录稿教学质量评估与不确定性路由系统。

## 系统概述

本系统已支持基于 UNESCO《学生AI能力框架》的四维评估（以人为本、AI伦理、AI技术与应用、AI系统设计），核心流程如下：

1. **加载**结构化课堂转录 JSON。
2. **切片**：支持两种策略 —— 按固定时间窗口切分，或由 LLM 自动识别课程教学阶段（导入/讲授/活动/讨论/总结）。
3. **评估**：每个切片按四个维度分别评估；每个维度用可配置模型池（ensemble_models）轮询进行 Monte Carlo 多次采样（默认 10-20 次），生成评分、理由和能力编码。
4. **语义聚类**：通过 Embedding 余弦相似度判断评估理由的语义等价性，使用双向蕴含边 + 连通分量（networkx / 内置图遍历）聚类。
5. **双熵计算**：计算分数熵（评分一致性）和语义熵（理由聚类一致性）。
6. **路由决策**：两项熵值均低于阈值 → `auto_accept`，否则 → `human_review`。
7. **高熵辩论机制**：高熵切片先触发 2 轮内部辩论，收敛后可重新判定自动通过。
8. **专家主动学习**：被拦截切片会生成模拟专家改判记录并写入本地记忆库，用于后续检索增强。
9. **专家控制台数据包**：自动生成《AI 分歧诊断书》JSON。

该设计利用异构模型的多样化推理发现评价中的不确定性区域，确保只有高置信度评估才自动通过。

## 项目架构

```
AI-Human/
├── classroom_ai/                     # 核心代码包
│   ├── pipeline/
│   │   ├── core_validation.py        # 主流程编排 + 兼容包装函数
│   │   └── debate_orchestrator.py    # Phase 2 多智能体辩论协议骨架
│   ├── llm/
│   │   ├── base.py                   # LLM 抽象基类
│   │   ├── factory.py                # LLM 工厂模式 (mock_local|ollama|openai_compatible)
│   │   ├── mock_local.py             # 本地 Mock LLM（离线烟测）
│   │   ├── ollama.py                 # Ollama 本地 LLM provider
│   │   └── openai_compatible.py      # OpenAI 兼容 API 适配器 (LiteLLM 风格)
│   ├── embedding/
│   │   ├── base.py                   # Embedder 抽象基类
│   │   ├── factory.py                # Embedding 工厂模式 (hashing|ollama|openai_compatible)
│   │   ├── hashing.py                # 哈希嵌入（零依赖离线可用）
│   │   ├── ollama.py                 # Ollama 嵌入 provider
│   │   └── openai_compatible.py      # OpenAI 兼容嵌入适配器
│   ├── evaluation/
│   │   ├── prompts.py                # CLASS 量表评估 prompt
│   │   └── parser.py                 # LLM 输出 JSON 解析器（鲁棒修复: markdown fence/控制字符/全角引号/转义/逗号）
│   ├── schemas/
│   │   ├── transcript.py             # 转录稿数据模型
│   │   └── slice.py                  # 切片数据模型（含 phase_label）
│   ├── slicing/
│   │   ├── time_slicer.py            # 固定时间窗口切片
│   │   └── phase_slicer.py           # LLM 驱动的课程阶段检测切片
│   └── uncertainty/
│       ├── entropy.py                # 香农熵计算
│       ├── semantic_entropy.py       # 语义熵: 双向蕴含边 + 连通分量聚类 + EmbeddingEntailmentJudge
│       └── decision.py               # 多数投票 + 双阈值路由决策
├── configs/                          # 配置文件
│   ├── local_mock.yaml               # Mock 模式（离线可运行）
│   ├── local_mock_ensemble.yaml      # Mock + ensemble 模型池
│   ├── local_ollama.yaml             # Ollama 单模型模式
│   ├── local_ollama_ensemble.yaml    # Ollama + 异构模型池轮询
│   ├── local_cluster.yaml            # 本地 GPU 集群配置（含 ensemble_models 映射）
│   └── cloud_api.yaml                # 云端 API 配置（LiteLLM 风格路由）
├── scripts/                          # 命令行工具
│   ├── run_core_validation.py        # 基础管线入口
│   ├── convert_transcript.py         # 原始转录 txt → JSON 转换
│   ├── test_transcript.py            # 完整测试运行器
│   └── build_human_review_diagnostic.py # Phase 5 诊断书 JSON 打包
├── data/sample/                      # 测试数据
│   ├── lesson_001.json               # 示例课堂转录
│   ├── S3-4.json                     # S3-4 AI基础概念课
│   ├── S5-2.json                     # S5-2 AI绘画模型课
│   └── S5-2_full.json                # S5-2 全文转录（19 segments）
├── outputs/                          # 测试报告输出（不纳入版本控制）
├── tests/                            # 单元测试
├── pyproject.toml
└── requirements.txt
```

## 快速开始

### 1. 安装

```bash
# Python >= 3.11，标准库即可运行 mock 模式
pip install -e .
```

### 2. 选择运行模式

#### 模式 A：Mock 模式（离线，无需模型）

```bash
python scripts/run_core_validation.py \
  --transcript data/sample/lesson_001.json \
  --config configs/local_mock.yaml \
  --output outputs/mock_result.json
```

#### 模式 B：Ollama 单模型（本地 LLM）

首先拉取所需模型：

```bash
ollama pull Qwen2.5:7b-instruct-q4_K_M
```

然后运行：

```bash
python scripts/run_core_validation.py \
  --transcript data/sample/S5-2_full.json \
  --config configs/local_ollama.yaml \
  --output outputs/ollama_result.json
```

#### 模式 C：Ollama 异构模型池 Ensemble（推荐）

利用多模型轮询发现评价分歧：

```bash
# 拉取多个异构模型
ollama pull deepseek-r1:7b
ollama pull gemma3:4b-it-q4_K_M
ollama pull llama3.1:8b

# 运行 ensemble 评估
python scripts/run_core_validation.py \
  --transcript data/sample/S5-2_full.json \
  --config configs/local_ollama_ensemble.yaml \
  --output outputs/ensemble_result.json
```

#### 模式 D：云端 API

```bash
export OPENAI_API_KEY=sk-xxx
python scripts/run_core_validation.py \
  --transcript data/sample/lesson_001.json \
  --config configs/cloud_api.yaml \
  --output outputs/cloud_result.json
```


## 三阶段解耦流程（新增）

系统现支持“Stage 1 → Stage 2 → Stage 3”解耦运行，阶段间通过 JSON 文件传递状态，且每次运行会创建独立输出目录。

### Stage 1：AI 独立全量初评

```bash
python scripts/step1_auto_evaluate.py   --transcript data/sample/S5-2_full.json   --config configs/local_ollama_ensemble.yaml   --output-root outputs
```

输出：
- `outputs/run_YYYYmmdd_HHMMSS/stage1_result.json`
- 以及基础中间态 `outputs/stage1_result.json`（便于 Stage 2 默认读取）

说明：
- Parser 会自动剥离 markdown fenced code block（如 ```json）。
- 分数字段支持从字符串提取（如“4分”）；无效/越界统一记为 `0`。
- `score_entropy` 计算会剔除 `0` 分；有效票数 `<2` 时强制设为 `999.0` 触发人工审核。

### Stage 2：人工靶向微干预（Streamlit）

```bash
streamlit run scripts/step2_human_review_app.py
```

操作流程：
1. 在页面中填写/确认 `stage1_result.json` 路径并加载。
2. 仅对 `decision == human_review` 的“切片-维度”填写专家选择。
3. 如选择 `CUSTOM`，填写“自定义意见”。
4. 保存生成 `stage2_human_feedback.json`。

### Stage 3：AI 局部重评与终报生成

```bash
python scripts/step3_resolve_and_report.py   --config configs/local_ollama_ensemble.yaml   --stage1 outputs/run_YYYYmmdd_HHMMSS/stage1_result.json   --stage2 outputs/stage2_human_feedback.json   --output-root outputs
```

输出：
- `outputs/run_YYYYmmdd_HHMMSS/final_report.json`

说明：
- 只会对 Stage 2 中 `needs_human_intervention=true` 的指定切片/维度进行重评。
- 重评结果会覆盖 Stage 1 对应位置，并将该维度 `decision` 更新为 `resolved_by_human_ai_collab`。

### 微干预模块（`classroom_ai/pipeline/micro_intervention.py`）

- `generate_expert_question(...)`：根据高分派/低分派理由生成专家单选题（JSON）。
- 选项中会自动追加固定项：`{"id": "CUSTOM", "text": "以上选项都不准确，我有自己的看法。"}`。
- `resolve_with_expert_feedback(...)`：融合冲突上下文、题目及专家意见，输出最终 1-7 分、理由、能力编码。

## 配置文件说明

```yaml
# configs/local_ollama_ensemble.yaml
runtime:
  mode: local

slicing:
  strategy: time             # time=时间窗口 | phase=阶段检测
  window_seconds: 300        # 时间窗口大小（秒）
  overlap_seconds: 60        # 重叠量（秒）

llm:
  provider: ollama           # mock_local | ollama | openai_compatible
  host: http://localhost:11434
  model: Qwen2.5:7b-instruct-q4_K_M  # 主模型
  temperature: 0.7
  max_tokens: 2048
  monte_carlo_samples: 10    # 每个切片的采样次数
  ensemble_models: deepseek-r1:7b,gemma3:4b-it-q4_K_M,llama3.1:8b  # 模型池轮询

embedding:
  provider: hashing          # hashing | ollama | openai_compatible
  dimensions: 256            # hashing 模式下的向量维度

uncertainty:
  semantic_entropy_threshold: 1.5       # 语义熵上限（超过触发人工审核）
  score_entropy_threshold: 1.0          # 分数熵上限
  embedding_similarity_threshold: 0.75  # Embedding 余弦相似度聚类阈值
```

### 关键配置字段

| 字段 | 说明 |
|------|------|
| `llm.ensemble_models` | 逗号分隔的模型列表，Monte Carlo 采样时轮询切换不同模型。留空则仅使用主模型 |
| `embedding.provider` | 语义聚类的嵌入来源。生产环境推荐 `ollama` 或 `openai_compatible`，避免 hashing 语义失真 |
| `uncertainty.embedding_similarity_threshold` | 两个评估理由判为语义等价的余弦相似度下限 |

### 切片策略对比

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `strategy: time` | 按固定窗口 + 重叠切分 | 快速测试、无 LLM 环境 |
| `strategy: phase` | LLM 识别课程阶段（导入/讲授/活动/讨论/总结） | 正式评估，细粒度阶段分析 |

## 核心设计

### LLM/Embedding 工厂模式

通过 [factory.py](classroom_ai/llm/factory.py) 和 [factory.py](classroom_ai/embedding/factory.py) 实现 provider 解耦，根据配置在本地 Ollama 与云端 OpenAI-compatible / LiteLLM 风格之间自动切换。

兼容包装函数 `build_llm()` / `build_embedder()` 保留在 [core_validation.py](classroom_ai/pipeline/core_validation.py)，避免旧调用直接失效。

### Ensemble 模型池轮询

当配置 `ensemble_models` 列表时，每次 Monte Carlo 采样自动轮询到不同异构模型。不同模型的评分和推理风格产生多样化输出，使语义熵和分数熵能够捕获真正的评价分歧。

### 语义熵：连通分量聚类

替代了早期版本的"固定余弦阈值贪心聚类"：

1. 使用 [EmbeddingEntailmentJudge](classroom_ai/uncertainty/semantic_entropy.py) 基于 embedding 余弦相似度判断两个理由是否语义等价
2. 构建双向蕴含边
3. 通过 networkx 连通分量（或内置图遍历降级）完成聚类
4. 计算聚类标签的香农熵作为语义熵

### 解析器鲁棒性

[parser.py](classroom_ai/evaluation/parser.py) 对 LLM 输出的 JSON 进行了多层鲁棒修复：

- Markdown 代码块剥离（` ```json ` / ` ``` `）
- 控制字符替换（0x00-0x1f, 0x7f）
- 中文全角引号转半角（`"" ''`）
- 非法转义符修复
- JSON 尾部逗号移除
- 数组元素间缺逗号补充

### Phase 2：多智能体辩论

[debate_orchestrator.py](classroom_ai/pipeline/debate_orchestrator.py) 提供了两轮会话收敛框架：
- 两个评估代理交替提出论点和反对论点
- 通过反驳/修订循环判断是否收敛
- 输出收敛状态、轮次和更新后的评估理由

## 原始转录格式转换

原始 txt 转录（时间戳 + 说话人格式）可通过以下命令转为标准 JSON：

```bash
python scripts/convert_transcript.py <原始文件.txt> <课程ID>
# 例如：
python scripts/convert_transcript.py S5-2_原文.txt S5-2_full
```

## 输出报告结构

报告（`outputs/{lesson_id}_xxx.json`）包含：

| 字段 | 说明 |
|------|------|
| `lesson_id` | 课程 ID |
| `config` | 本次测试使用的完整配置 |
| `slice_count` | 切片总数 |
| `results[]` | 每切片的完整评估结果 |
| `results[].slice_id` | 切片标识（含时间范围） |
| `results[].score_distribution` | 评分分布直方图 |
| `results[].majority_score` | 多数投票分数 |
| `results[].score_entropy` | 分数香农熵 |
| `results[].semantic_entropy` | 语义聚类熵 |
| `results[].semantic_clusters[]` | 语义聚类详情（簇大小、代表理由、成员索引） |
| `results[].decision` | `auto_accept` / `human_review` |
| `results[].samples[]` | 每次采样的完整评估（分数、理由、证据、原始输出） |

## Provider 支持矩阵

| Provider | LLM | Embedding | 说明 |
|----------|-----|-----------|------|
| `mock_local` | ✅ | — | 基于关键词计数的模拟 LLM，离线可用 |
| `ollama` | ✅ | ✅ | 本地 Ollama，无需联网 |
| `openai_compatible` | ✅ | ✅ | 任意 OpenAI 兼容 API（含 LiteLLM 代理） |
| `hashing` | — | ✅ | 基于 BLAKE2b 哈希的轻量嵌入，零依赖 |

## 依赖

- Python >= 3.11
- 核心管线零第三方依赖（仅 Python 标准库）
- Ollama 模式需要本地运行的 [Ollama](https://ollama.com/) 服务
- 语义聚类 `networkx` 为可选依赖，缺失时自动降级到内置图遍历

## 测试

```bash
python -m pytest tests/ -v
```

## 典型实验结果（S5-2 AI绘画模型课）

使用 Ollama ensemble（deepseek-r1:7b / gemma3:4b / llama3.1:8b，10 samples × 4 slices）：

| Slice | Majority | Score Entropy | Semantic Entropy | Decision |
|-------|----------|---------------|-------------------|----------|
| 导入+GAN 讲解 | 6 | 0.995 | 1.228 | ✅ auto_accept |
| GAN 深化+互动 | 5 | **1.523** | **1.748** | 🔴 human_review |
| 扩散模型讲解 | 5 | 0.956 | 1.228 | ✅ auto_accept |
| 课程总结 | 5 | 0.940 | 0.940 | ✅ auto_accept |

系统正确识别出 GAN 深化+互动片段是唯一存在跨模型严重分歧的区域，人工审核命中率 25%。

## Phase 4/5 新增能力

- **Phase 4（主动学习）**：在 `configs/local_ollama.yaml` 通过 `phase4_rag` 配置启用模拟专家改判，默认将首个 `human_review` 切片改判为 4 分并写入 `outputs/phase4_expert_memory.json`。
- **Phase 5（控制台数据组装）**：执行以下命令可生成《AI 分歧诊断书》：

```bash
python scripts/build_human_review_diagnostic.py --result outputs/ollama_result.json --output outputs/human_review_diagnostic.json
```

- 诊断书包含：切片时间戳、分布熵、高低分理由对照、标准化 evidence。


### UNESCO 四维增强配置（新增）

```yaml
evaluation:
  dimensions: human_centered,ai_ethics,ai_tech_and_app,ai_system_design
  dimension_concurrency: 4

uncertainty:
  semantic_entropy_threshold: 1.5
  score_entropy_threshold: 1.0
  dimension_thresholds:
    ai_ethics:
      semantic_entropy_threshold: 1.2
      score_entropy_threshold: 0.9

phase4_rag:
  enabled: true
  retrieve_top_k: 2
  memory_path: outputs/phase4_expert_memory.json
```
