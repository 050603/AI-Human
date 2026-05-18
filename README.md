# AI-Human Classroom Core Validation

基于 LLM 的课堂转录稿教学质量评估与不确定性路由系统。

## 系统概述

本系统对课堂转录稿进行 CLASS 量表评估，核心流程如下：

1. **加载**结构化课堂转录 JSON。
2. **切片**：支持两种策略 —— 按固定时间窗口切分，或由 LLM 自动识别课程教学阶段（导入/讲授/活动/讨论/总结）。
3. **评估**：每个切片用 LLM 进行 Monte Carlo 多次采样（默认 20 次），生成评分和理由。
4. **聚类**：用语义嵌入模型将评估理由进行贪心余弦聚类。
5. **双熵计算**：计算分数熵（评分一致性）和语义熵（理由一致性）。
6. **路由决策**：两项熵值均低于阈值 → `auto_accept`（自动通过），否则 → `human_review`（人工审核）。

该设计确保只有评估高度稳定时才自动通过，否则触发人工复核。

## 项目结构

```
AI-Human/
├── classroom_ai/                  # 核心代码包
│   ├── pipeline/
│   │   └── core_validation.py     # 主流程编排
│   ├── schemas/
│   │   ├── transcript.py          # 转录稿数据模型
│   │   └── slice.py               # 切片数据模型（含 phase_label）
│   ├── slicing/
│   │   ├── time_slicer.py         # 固定时间窗口切片
│   │   └── phase_slicer.py        # LLM 驱动的课程阶段检测切片
│   ├── llm/
│   │   ├── base.py                # LLM 抽象基类
│   │   ├── mock_local.py          # 本地 Mock LLM（离线烟测）
│   │   ├── ollama.py              # Ollama 本地 LLM provider
│   │   └── openai_compatible.py   # OpenAI 兼容 API 适配器
│   ├── embedding/
│   │   ├── base.py                # Embedder 抽象基类
│   │   ├── hashing.py             # 哈希嵌入（零依赖离线可用）
│   │   ├── ollama.py              # Ollama 嵌入 provider
│   │   └── openai_compatible.py   # OpenAI 兼容嵌入适配器
│   ├── evaluation/
│   │   ├── prompts.py             # CLASS 量表评估 prompt
│   │   └── parser.py              # LLM 输出 JSON 解析器
│   └── uncertainty/
│       ├── entropy.py             # 香农熵计算
│       ├── semantic_entropy.py    # 语义熵 + 贪心余弦聚类
│       └── decision.py            # 多数投票 + 路由决策
├── configs/                       # 配置文件
│   ├── local_mock.yaml            # Mock 模式（离线可运行，无需模型）
│   ├── local_ollama.yaml          # Ollama 模式（本地 LLM + 嵌入）
│   └── api_template.yaml          # OpenAI 兼容 API 模板
├── scripts/                       # 命令行工具
│   ├── run_core_validation.py     # 基础管线入口
│   ├── test_transcript.py         # 完整测试运行器（生成详细报告）
│   ├── convert_transcript.py      # 原始转录 txt → JSON 转换
│   ├── convert_s3-4.py            # S3-4 专用转换
│   └── test_s3-4.py               # S3-4 专用测试运行器
├── data/sample/                   # 测试数据
│   ├── lesson_001.json            # 示例课堂转录
│   ├── S3-4.json                  # S3-4 AI基础概念课（标准化）
│   └── S5-2.json                  # S5-2 AI绘画模型课（标准化）
├── outputs/                       # 测试报告输出（不纳入版本控制）
├── tests/                         # 单元测试
├── notebooks/                     # Jupyter Notebook 演示
├── pyproject.toml
└── requirements.txt
```

## 快速开始

### 1. 安装

```bash
# Python >= 3.11，零第三方依赖
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

#### 模式 B：Ollama 模式（本地 LLM，推荐）

首先拉取所需模型：

```bash
ollama pull Qwen2.5:7b          # 评估 LLM
ollama pull nomic-embed-text    # 语义嵌入
```

然后运行：

```bash
# 基础管线
python scripts/run_core_validation.py \
  --transcript data/sample/S3-4.json \
  --config configs/local_ollama.yaml \
  --output outputs/ollama_result.json

# 完整测试（含详细过程报告）
python scripts/test_transcript.py S3-4
python scripts/test_transcript.py S5-2
```

#### 模式 C：OpenAI 兼容 API

修改 `configs/api_template.yaml` 填入 `api_base` 和 `api_key_env`，然后运行：

```bash
python scripts/run_core_validation.py \
  --transcript data/sample/lesson_001.json \
  --config configs/api_template.yaml \
  --output outputs/api_result.json
```

## 配置文件说明

```yaml
# configs/local_ollama.yaml
runtime:
  mode: local

slicing:
  strategy: phase          # phase=阶段检测 | time=时间窗口
  window_seconds: 600      # 时间模式下的窗口大小（可调整）
  overlap_seconds: 120     # 时间模式下的重叠量

llm:
  provider: ollama         # mock_local | ollama | openai_compatible
  host: http://localhost:11434
  model: Qwen2.5:7b        # 可换为其他 Ollama 模型
  monte_carlo_samples: 20  # 每个切片的采样次数

embedding:
  provider: ollama         # hashing | ollama | openai_compatible
  model: nomic-embed-text

uncertainty:
  similarity_threshold: 0.82        # 余弦相似度聚类阈值
  semantic_entropy_threshold: 0.75  # 语义熵上限
  score_entropy_threshold: 0.90     # 分数熵上限
```

### 切片策略对比

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `strategy: time` | 按固定窗口 + 重叠切分 | 快速离线测试、无 LLM 环境 |
| `strategy: phase` | LLM 识别课程阶段（导入/讲授/活动/讨论/总结） | 正式评估，细粒度阶段分析 |

`strategy: time` 时可通过 `window_seconds` 调整窗口大小，建议根据课堂时长设置（例如 40 分钟课程可设 300-600 秒）。

## 原始转录格式转换

原始 txt 转录（时间戳 + 说话人格式）可通过以下命令转为标准 JSON：

```bash
python scripts/convert_transcript.py <原始文件.txt> <课程ID>
# 例如：
python scripts/convert_transcript.py S5-2_原文.txt S5-2
```

## 输出报告结构

完整测试报告（`outputs/{lesson_id}_full_test_report.json`）包含：

| 字段 | 说明 |
|------|------|
| `report_meta` | 报告元信息（时间、版本、模型） |
| `transcript_info` | 原始转录详情（全部段落的文本和时间戳） |
| `config` | 本次测试使用的完整配置 |
| `slicing_info` | 切片策略与各切片元信息 |
| `slice_results[]` | 每切的完整评估结果 |
| `slice_results[].phase_label` | 课程阶段名称（仅 phase 模式） |
| `slice_results[].score_distribution` | 评分分布 |
| `slice_results[].semantic_clusters` | 语义聚类详情（簇代表理由、成员索引） |
| `slice_results[].decision` | 路由决策：`auto_accept` / `human_review` |
| `slice_results[].samples` | 每次采样的完整评估（评分、理由、证据） |
| `summary` | 汇总（auto_accept 率、总体评分、耗时） |
| `timing` | 性能统计（LLM 调用次数、平均耗时等） |

## Provider 支持矩阵

| Provider | LLM | Embedding | 说明 |
|----------|-----|-----------|------|
| `mock_local` | ✅ | — | 基于关键词计数的模拟 LLM，离线可用 |
| `ollama` | ✅ | ✅ | 本地 Ollama，无需联网 |
| `openai_compatible` | ✅ | ✅ | 任意 OpenAI 兼容 API |
| `hashing` | — | ✅ | 基于 BLAKE2b 哈希的轻量嵌入 |

## 依赖

- Python >= 3.11
- 核心管线零第三方依赖（仅 Python 标准库）
- Ollama 模式需要本地运行的 [Ollama](https://ollama.com/) 服务

## 测试

```bash
python -m pytest tests/ -v
```