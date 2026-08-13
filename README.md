# WebSim：多数据集、多模型推荐系统交互仿真平台

WebSim 是一个面向推荐系统实验的本地仿真平台。项目将推荐模型、可交互网页、虚拟用户 Agent 和批量实验工具整合在一起，可用于观察用户点击行为、对比不同推荐算法，以及开展基于大语言模型的用户行为仿真。

![WebSim 方法概览](Method.png)

## 主要功能

- 支持 MovieLens-1M、Amazon All Beauty 和 Amazon Magazine Subscriptions 三类数据集。
- 支持 SASRec、LightGCN、Mult-VAE、PopRec、BPR-MF、GRU4Rec 和 BERT4Rec 七种推荐模型。
- 提供两种交互界面：四卡片网格页和单卡片滑动页。
- 根据用户点击历史实时刷新推荐结果，每页返回 4 个条目。
- 记录当前会话中的曝光、点击和热度变化。
- 支持发表评论、查看历史评论和为评论点赞。
- 提供完整的模型训练脚本和 Amazon 数据集批量训练脚本。
- 支持无需浏览器和 Flask 服务的大规模异步 LLM Agent 仿真。

## 系统结构

```text
浏览器
      │
      ▼
Flask Web 服务（app.py）
      │
      ├── 数据集与条目元数据（MovieCatalog）
      ├── 推荐模型统一入口（MovieRecommender）
      ├── 评论、点赞与会话统计
      └── 海报文件或缺省 SVG 海报
```

推荐模型权重仅在对应文件存在时加载。若当前数据集没有可用模型，系统仍可展示随机条目；若模型无法生成有效结果，也会回退到随机推荐。

## 技术栈

- Python 3.11+
- Flask
- PyTorch
- NumPy / pandas
- 原生 HTML、CSS 和 JavaScript
- CAMEL-AI / OpenAI 兼容接口

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone <your-repository-url>
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

### 2. 准备数据集

数据集和模型权重体积较大，不随 Git 仓库上传。推荐将数据放在项目同级的 `WebSim_Dataset` 目录：

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

也可以通过环境变量指定任意数据目录：

```powershell
$env:ML1M_DATASET_DIR = "D:\datasets\MM-ML-1M-main"
$env:AMAZON_ALL_BEAUTY_DATASET_DIR = "D:\datasets\All_Beauty"
$env:AMAZON_MAGAZINE_SUBSCRIPTIONS_DATASET_DIR = "D:\datasets\Magazine_Subscriptions"
```

### 3. 准备模型权重

默认权重目录为 `artifacts/`，命名规则如下：

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

Amazon 权重使用对应数据集后缀，例如：

```text
sasrec_amazon_all_beauty.pt
bert4rec_amazon_magazine_subscriptions.pt
```

模型权重也可通过环境变量单独覆盖，详见下方“环境变量”。

### 4. 启动服务

```bash
python app.py
```

默认监听 `127.0.0.1:19001`：

- 网格推荐页：<http://127.0.0.1:19001/>
- 滑动推荐页：<http://127.0.0.1:19001/swipe>
- 健康检查：<http://127.0.0.1:19001/health>

macOS / Linux 也可以使用脚本启动：

```bash
./scripts/run_swipe_page.sh
```

该脚本默认使用端口 `19002`，并在服务就绪后打开滑动页面。停止服务：

```bash
./scripts/stop_swipe_page.sh
```

## 交互页面

### 网格页 `/`

- 每次展示 4 个随机或推荐条目。
- 点击卡片后将条目加入最近 20 条交互历史，并生成下一批推荐。
- 支持翻页、重置、评论、评论点赞以及历史评论显隐。
- 卡片展示评分、基础热度和当前会话产生的曝光/点击增量。

### 滑动页 `/swipe`

- 一次展示一个条目。
- 向上滑动表示选择当前条目，并据此刷新推荐。
- 向下滑动浏览推荐列表中的下一项。
- 支持键盘方向键操作。

## 训练推荐模型

七个模型均有独立训练入口。以下以 MovieLens-1M 为例：

```bash
python train_poprec.py  --dataset-dir ../WebSim_Dataset/MM-ML-1M-main --output-model artifacts/poprec_ml1m.pt
python train_sasrec.py  --dataset-dir ../WebSim_Dataset/MM-ML-1M-main --output-model artifacts/sasrec_ml1m.pt --epochs 10
python train_lightgcn.py --dataset-dir ../WebSim_Dataset/MM-ML-1M-main --output-model artifacts/lightgcn_ml1m.pt --epochs 30
python train_multvae.py --dataset-dir ../WebSim_Dataset/MM-ML-1M-main --output-model artifacts/multvae_ml1m.pt --epochs 30
python train_bprmf.py   --dataset-dir ../WebSim_Dataset/MM-ML-1M-main --output-model artifacts/bprmf_ml1m.pt --epochs 50
python train_gru4rec.py --dataset-dir ../WebSim_Dataset/MM-ML-1M-main --output-model artifacts/gru4rec_ml1m.pt --epochs 30
python train_bert4rec.py --dataset-dir ../WebSim_Dataset/MM-ML-1M-main --output-model artifacts/bert4rec_ml1m.pt --epochs 30
```

训练脚本会按时间顺序构造用户序列，并输出 Hit Rate、NDCG 等评估指标。可运行 `python <训练脚本> --help` 查看模型特有参数。

macOS / Linux 可一键训练某个 Amazon 数据集的全部模型：

```bash
./scripts/train_amazon_all_beauty_all.sh
./scripts/train_amazon_magazine_subscriptions_all.sh
```

通用入口：

```bash
./scripts/train_amazon_mm2018_all.sh <dataset_dir> <model_suffix>
```

## 虚拟用户 Agent

Agent 功能需要 OpenAI 兼容的大模型接口。请在项目根目录创建 `.env`；该文件已被 `.gitignore` 排除，不要提交密钥。

```dotenv
YUNWU_API_KEY=your_api_key
YUNWU_BASE_URL=https://your-openai-compatible-endpoint/v1
YUNWU_MODEL=your_model_name
```

### 生成用户画像

```bash
python generate_profiles.py --count 100 --seed 42 --output profiles.jsonl
```

每行是一名虚拟用户的 JSON 画像，包含人口属性、内容偏好、交互倾向和独立随机种子。

### 无浏览器并行仿真

`large_scale/large_scale_runner.py` 直接调用 WebSim 的 Python 环境与 LLM 策略，不启动浏览器，适合批量实验：

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m large_scale.large_scale_runner `
  --start-index 0 `
  --count 4 `
  --profiles .\profiles.jsonl `
  --track 2 `
  --batch-size 4 `
  --max-concurrency 2 `
  --max-retries 3 `
  --request-timeout 120 `
  --result-dir .\large_scale_runs
```

macOS / Linux：

```bash
python -m large_scale.large_scale_runner \
  --profiles profiles.jsonl \
  --dataset ml1m \
  --model poprec \
  --count 100 \
  --track 5 \
  --batch-size 20 \
  --max-concurrency 4
```

并行参数含义：

- `--count`：本次运行的 Agent 总数。
- `--batch-size`：每批提交到异步队列的 Agent 数量。
- `--max-concurrency`：同时进行的真实 LLM API 请求上限，也是实际 API 并发控制参数。

可选的画像选择方式：

- `--profile-selection sequential`：从 `--start-index` 开始顺序选择。
- `--profile-selection random`：随机抽样，可用 `--sample-seed` 保证复现。

每次运行会在 `large_scale_runs/` 下生成：

- `events.jsonl`：逐轮行为事件。
- `summary.json`：实验汇总。
- `memory.json`：按 Agent 保存的最终状态与历史。
- `scheduler.log`：调度与错误日志。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 检查数据集路径、可用模型和默认模型 |
| `GET` | `/api/init` | 初始化会话并返回 4 个随机条目 |
| `POST` | `/api/select` | 记录点击并根据历史生成推荐 |
| `POST` | `/api/next` | 获取推荐结果的下一页 |
| `GET` | `/api/comments` | 按数据集和条目查询评论 |
| `POST` | `/api/comment` | 发布评论 |
| `POST` | `/api/comment/like` | 为评论点赞 |
| `POST` | `/api/session/end` | 清除当前仿真会话状态 |
| `GET` | `/poster/<dataset_key>/<item_id>` | 返回条目海报或缺省 SVG |
| `POST` | `/api/yunwu-test` | 测试大模型接口连接 |

初始化示例：

```bash
curl "http://127.0.0.1:19001/api/init?dataset_key=ml1m&model_name=sasrec"
```

点击并请求推荐：

```bash
curl -X POST http://127.0.0.1:19001/api/select \
  -H "Content-Type: application/json" \
  -d '{"movie_id":"296","dataset_key":"ml1m","model_name":"sasrec"}'
```

接口使用 Flask Cookie Session 保存浏览历史，因此连续请求 API 时需要复用 Cookie。

## 环境变量

### 服务配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `19001` | Flask 服务端口 |
| `FLASK_SECRET_KEY` | 启动时随机生成 | Session 签名密钥；稳定部署时应显式设置 |
| `DEFAULT_DATASET_KEY` | `ml1m` | 默认数据集 |
| `WEBSIM_INIT_RANDOM_SEED` | `42` | 初始随机采样种子 |

### 数据集路径

- `ML1M_DATASET_DIR`
- `AMAZON_ALL_BEAUTY_DATASET_DIR`
- `AMAZON_MAGAZINE_SUBSCRIPTIONS_DATASET_DIR`

### 模型路径

MovieLens-1M 同时支持通用别名：

- `SASREC_ML1M_MODEL_PATH` 或 `SASREC_MODEL_PATH`
- `LIGHTGCN_ML1M_MODEL_PATH` 或 `LIGHTGCN_MODEL_PATH`
- `MULTVAE_ML1M_MODEL_PATH` 或 `MULTVAE_MODEL_PATH`
- `POPREC_ML1M_MODEL_PATH` 或 `POPREC_MODEL_PATH`
- `BPRMF_ML1M_MODEL_PATH` 或 `BPRMF_MODEL_PATH`
- `GRU4REC_ML1M_MODEL_PATH` 或 `GRU4REC_MODEL_PATH`
- `BERT4REC_ML1M_MODEL_PATH` 或 `BERT4REC_MODEL_PATH`

Amazon 模型变量遵循以下格式：

```text
<MODEL>_AMAZON_ALL_BEAUTY_MODEL_PATH
<MODEL>_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH
```

其中 `<MODEL>` 可取 `SASREC`、`LIGHTGCN`、`MULTVAE`、`POPREC`、`BPRMF`、`GRU4REC` 或 `BERT4REC`。

## 项目目录

```text
D8EAX/
├── app.py                         # Flask 服务与 API
├── recommender.py                 # 数据目录与七种模型的统一推理入口
├── dataset_utils.py               # MovieLens / Amazon 用户序列加载
├── *_rec.py / *gcn.py / *vae.py  # 模型定义
├── train_*.py                     # 各模型训练入口
├── templates/                     # HTML 页面
├── static/                        # 前端脚本与样式
├── artifacts/                     # 权重、评论和训练结果
├── generate_profiles.py           # 虚拟用户画像生成
├── large_scale/                   # 无浏览器的大规模仿真
├── scripts/                       # 服务管理与批量训练脚本
└── docs/                          # UML 和软件工程图
```

## 常见问题

### `/health` 中数据集出现错误

检查对应数据集目录是否存在，并确认目录结构符合上面的约定。自定义路径时使用数据集环境变量覆盖默认路径。

### `available_models` 为空

表示当前数据集没有找到任何 `.pt` 权重。页面仍会展示随机条目，但不会执行模型推荐。请训练模型，或通过模型路径环境变量指向已有权重。

### 页面没有海报

海报是可选资源。系统找不到本地图片时会自动返回带条目标题的 SVG 占位图，不影响推荐和仿真流程。

### Agent 提示缺少环境变量

确认项目根目录存在 `.env`，且包含 `YUNWU_API_KEY`、`YUNWU_BASE_URL` 和 `YUNWU_MODEL`。接口必须兼容 OpenAI 风格的聊天补全调用。

## 开发检查

提交前可运行基础语法检查：

```bash
python -m compileall -q .
```

模型权重、数据集、`.env`、虚拟环境、日志和仿真结果均已在 `.gitignore` 中排除。

## 文档

更详细的系统结构与 UML 图位于 `docs/`：

- `docs/software_engineering_diagrams.md`
- `docs/software_engineering_diagrams.pdf`
- `docs/websim_system_uml.pdf`

## License

当前仓库尚未包含开源许可证。在公开复用、分发或二次开发前，请先补充合适的 `LICENSE` 文件，并分别确认所用数据集的授权条款。
