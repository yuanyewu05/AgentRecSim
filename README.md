# AgentRecSim：基于多 Agent 用户仿真的推荐系统评测框架

AgentRecSim 的核心目标是：让多个具有不同用户画像的 LLM Agent 在推荐环境中模拟真人浏览行为，并通过它们产生的点击、翻页、停止及画像变化，评估不同推荐系统的表现。

项目中的 WebSim 不是研究终点，而是承载推荐结果和用户动作的仿真环境。核心实验不需要启动浏览器或 Flask 服务；多 Agent 运行器会直接调用 Python 推荐环境，并发完成大规模用户行为仿真。

本项目并不是直接接入某个在线推荐平台，而是将公开数据集和推荐算法部署到本地，构建一个可控、可复现的推荐环境。推荐结果由本地推荐模型生成，LLM Agent 只负责根据用户画像和当前推荐内容做出 `click`、`next` 或 `stop` 的行为决策。这样的职责划分可以将推荐算法与用户行为仿真分离，便于在统一实验条件下比较不同推荐模型。

## 研究流程

```text
静态用户画像 profiles.jsonl
          │
          ▼
随机或顺序抽取多个用户
          │
          ▼
为每名用户创建相互隔离的 Agent 状态
          │
          ▼
推荐模型生成当前可见条目
          │
          ▼
LLM Agent 根据画像、历史和当前推荐作出决策
      click / next / stop
          │
          ▼
执行动作并更新该 Agent 的动态画像
          │
          ▼
继续下一轮，直到主动停止或达到最大轮数
          │
          ▼
输出逐轮事件、最终记忆和实验汇总
```

这个流程可用于研究：

- 不同推荐模型面对同一组虚拟用户时的行为差异；
- 推荐结果能否促使不同类型用户点击；
- 用户何时继续探索、何时主动结束会话；
- 多轮交互后兴趣、满意度、耐心和疲劳度如何变化；
- 不同画像群体在同一推荐模型下是否表现出不同反馈；
- 大规模 Agent 实验的吞吐量、失败率和可复现性。

## 核心能力

- 多 Agent 仿真：一次实验运行多个相互独立的逻辑 Agent。
- 画像驱动决策：每个 Agent 根据自己的长期画像、动态状态和当前推荐进行判断。
- 真人式动作空间：Agent 每轮自主选择 `click`、`next` 或 `stop`。
- 动态画像：每次操作后更新兴趣权重、满意度、耐心、疲劳、探索倾向和近期行为。
- 无浏览器运行：核心实验无需启动网页或 Flask 服务，适合批量运行。
- 异步并发：批量调度 Agent，并单独限制真实 LLM API 并发数。
- 可复现实验：支持固定画像、随机抽样和随机种子。
- 多推荐模型：支持 SASRec、LightGCN、Mult-VAE、PopRec、BPR-MF、GRU4Rec 和 BERT4Rec。
- 多数据集：支持 MovieLens-1M、Amazon All Beauty 和 Amazon Magazine Subscriptions。
- 完整日志：保存每轮决策、决策理由、画像变化、最终状态和汇总指标。

## 快速运行多 Agent 实验

### 1. 安装环境

```bash
git clone https://github.com/yuanyewu05/D8EAX.git
cd D8EAX
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

建议使用 Python 3.11 或 3.12。

### 2. 配置 LLM 接口

在项目根目录创建 `.env`：

```dotenv
YUNWU_API_KEY=your_api_key
YUNWU_BASE_URL=https://your-openai-compatible-endpoint/v1
YUNWU_MODEL=your_model_name
```

接口需要兼容 OpenAI Chat Completions。`.env` 已被 `.gitignore` 排除，请勿将真实 API Key 提交到 GitHub。

### 3. 准备用户画像

仓库中的 `profiles.jsonl` 可以直接用于实验。也可以重新生成画像：

```bash
python generate_profiles.py --count 100 --seed 42 --output profiles.jsonl
```

文件每行表示一名虚拟用户，包含人口属性、内容偏好、厌恶项、探索倾向、流行度偏好、耐心和独立随机种子。

### 4. 准备数据和推荐模型

默认使用 MovieLens-1M 和 `poprec`。数据集建议放在项目同级目录：

```text
父目录/
├── D8EAX/
└── WebSim_Dataset/
    ├── MM-ML-1M-main/
    │   ├── ratings.dat
    │   ├── movies.dat
    │   ├── movies_details_clean.csv     # 可选
    │   └── posters/                     # 可选
    └── Amazon_MM_2018/
        ├── All_Beauty/
        │   ├── raw/*.json.gz
        │   ├── raw/meta_*.json.gz
        │   └── posters/                 # 可选
        └── Magazine_Subscriptions/
            ├── raw/*.json.gz
            ├── raw/meta_*.json.gz
            └── posters/                 # 可选
```

默认模型权重目录为 `artifacts/`：

```text
artifacts/
├── sasrec_ml1m.pt
├── lightgcn_ml1m.pt
├── multvae_ml1m.pt
├── poprec_ml1m.pt
├── bprmf_ml1m.pt
├── gru4rec_ml1m.pt
└── bert4rec_ml1m.pt
```

数据和权重未提交到 Git 仓库。缺少指定模型权重时，可以先运行对应的 `train_*.py` 训练脚本。数据和模型路径也可以通过环境变量覆盖，例如：

```powershell
$env:ML1M_DATASET_DIR = "D:\datasets\MM-ML-1M-main"
$env:POPREC_ML1M_MODEL_PATH = "D:\models\poprec_ml1m.pt"
```

### 5. 启动实验

下面的命令从画像库中无放回随机抽取 5 名用户，每名 Agent 最多执行 20 轮，并允许最多 2 个 LLM 请求同时进行：

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m large_scale.large_scale_runner `
  --count 5 `
  --profile-selection random `
  --sample-seed 42 `
  --profiles .\profiles.jsonl `
  --dataset ml1m `
  --model poprec `
  --track 20 `
  --batch-size 5 `
  --max-concurrency 2 `
  --max-retries 3 `
  --request-timeout 120
```

macOS / Linux：

```bash
python -m large_scale.large_scale_runner \
  --count 5 \
  --profile-selection random \
  --sample-seed 42 \
  --profiles ./profiles.jsonl \
  --dataset ml1m \
  --model poprec \
  --track 20 \
  --batch-size 5 \
  --max-concurrency 2 \
  --max-retries 3 \
  --request-timeout 120
```

核心实验不需要提前执行 `python app.py`，也不需要启动浏览器。

## Agent 如何模拟用户

每名 Agent 都有两层画像：

1. 静态画像：年龄、用户群体、喜欢与不喜欢的内容、流行度偏好、探索倾向等长期特征。
2. 动态画像：当前兴趣权重、满意度、耐心、疲劳、连续翻页次数、累计点击和近期动作等会话状态。

每轮决策时，LLM 会接收：

- 当前 Agent 的静态画像；
- 经过前几轮操作更新后的动态画像；
- 当前页面可见的推荐条目；
- 最近点击和动作历史；
- 是否还存在下一页。

LLM 随后自主返回一种动作：

| 动作 | 含义 | 对后续状态的影响 |
| --- | --- | --- |
| `click` | 对当前某个推荐条目感兴趣 | 记录点击、更新偏好，并根据新历史重新排序推荐 |
| `next` | 当前条目吸引力不足，继续浏览 | 翻到下一页，同时增加浏览消耗和连续翻页状态 |
| `stop` | 用户不愿继续当前会话 | 结束该 Agent，不再产生后续动作 |

不同 Agent 的状态、历史和动态画像相互隔离。`--max-concurrency` 只控制同时发出的 LLM 请求数量，不会让不同 Agent 共享记忆。

## 实验参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--count` | 本次运行的 Agent 总数 | `2` |
| `--profile-selection` | `sequential` 顺序选择或 `random` 随机选择画像 | `sequential` |
| `--start-index` | 顺序选择时的起始画像索引 | `0` |
| `--sample-seed` | 随机抽样种子；固定后可复现相同画像集合 | 无 |
| `--profiles` | 用户画像 JSONL 文件 | `profiles.jsonl` |
| `--dataset` | 推荐数据集 | `ml1m` |
| `--model` | 被评测的推荐模型 | `poprec` |
| `--track` | 每名 Agent 的最大交互轮数 | `1` |
| `--batch-size` | 每批提交到异步队列的 Agent 数量 | `20` |
| `--max-concurrency` | 同时进行的真实 LLM API 请求上限 | `2` |
| `--max-retries` | 单次决策最大尝试次数 | `3` |
| `--request-timeout` | 单次 LLM 请求超时秒数 | `120` |
| `--result-dir` | 实验结果根目录 | `large_scale_runs` |

`--batch-size` 不等于真实 API 并发数；实际并发由 `--max-concurrency` 控制。

## 输出与推荐系统评估

每次运行会在 `large_scale_runs/<运行时间>_llm_large/` 下生成：

| 文件 | 内容 |
| --- | --- |
| `events.jsonl` | 每轮 Agent 的动作、目标条目、决策理由、耗时以及画像更新前后状态 |
| `summary.json` | Agent 数量、模型、数据集、事件数、点击数、翻页数、停止数、失败数和吞吐量 |
| `memory.json` | 每名 Agent 的最终状态、点击历史、动作历史和动态画像 |
| `scheduler.log` | 批次调度、API 重试、异常和运行进度 |

可以先固定画像抽样种子和其他实验参数，再分别修改 `--model` 运行多组实验。比较各组 `summary.json` 和 `events.jsonl`，即可分析不同推荐模型产生的点击、浏览深度、主动停止及用户群体差异。

需要注意：点击数、翻页数和停止数是仿真行为统计，不应单独视为推荐质量结论。更可靠的比较应同时控制画像集合、数据集、交互轮数、LLM 模型和提示词版本，并进行多次重复实验。

## 推荐模型训练

MovieLens-1M 的七种模型可以分别训练：

```bash
python train_poprec.py
python train_bprmf.py
python train_lightgcn.py
python train_multvae.py
python train_gru4rec.py
python train_sasrec.py
python train_bert4rec.py
```

Amazon 数据集提供批量训练脚本：

```bash
./scripts/train_amazon_all_beauty_all.sh
./scripts/train_amazon_magazine_subscriptions_all.sh
```

生成的 `.pt` 权重默认存放在 `artifacts/`，并被 `.gitignore` 排除。

## 可选：WebSim 网页演示

网页界面用于人工查看推荐和调试，不是运行多 Agent 实验的前置服务。

```powershell
python app.py
```

然后访问：

```text
http://127.0.0.1:19001
```

网页服务提供数据集与模型选择、卡片点击、翻页、评论和会话统计等功能。可以通过以下接口检查配置：

```bash
curl http://127.0.0.1:19001/health
```

## 项目结构

```text
D8EAX/
├── large_scale/
│   ├── large_scale_runner.py   # 多 Agent 异步实验入口
│   ├── llm_policy.py           # LLM 用户决策与输出校验
│   ├── profile_updater.py      # 每轮操作后的动态画像更新
│   ├── websim_env.py           # 无浏览器推荐交互环境
│   ├── rule_policy.py          # 轻量规则策略
│   └── genre_utils.py          # 标签与类型处理
├── profiles.jsonl              # 用户画像库
├── generate_profiles.py        # 用户画像生成器
├── recommender.py              # 多推荐模型统一加载与推理
├── app.py                      # 可选的 WebSim 网页服务
├── train_*.py                  # 推荐模型训练脚本
├── artifacts/                  # 本地模型权重
├── scripts/                    # 服务和批量训练脚本
├── templates/                  # 网页模板
├── static/                     # 网页静态资源
└── requirements.txt
```

## 实验复现建议

- 固定 `--sample-seed`，保证不同推荐模型使用同一组用户画像。
- 固定 `YUNWU_MODEL` 和提示词代码版本，避免决策策略发生变化。
- 对每个推荐模型运行多组不同种子的实验，不以单次结果下结论。
- 同时保存 `summary.json`、`events.jsonl` 和代码提交号。
- 比较模型时保持 `--count`、`--track`、并发和超时参数一致。
- 公开实验结果时说明所用数据、推荐权重、LLM 服务和采样配置。

## 安全说明

- 不要提交 `.env`、API Key 或其他凭据。
- 数据集和 `.pt` 模型权重默认不纳入 Git。
- 实验输出可能包含完整用户画像和决策理由，公开前请检查内容。
- 仿真 Agent 的行为不等同于真实用户研究结论，需要结合真实数据验证。
