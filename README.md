# PP-OCR 图文识别 API

基于 PP-DocLayout-M 与 PP-OCRv6 Medium 的本地 HTTP 服务，提供：

- 文档方向矫正：识别前自动判断旋转角度并矫正，结果坐标仍映射回原图
- 文字识别：输出文字、置信度、文本框和多边形坐标
- 图片/图形区域检测：输出图片、图表等区域坐标
- 区域坐标嵌入：将图片区域信息关联到识别文字
- 阅读顺序整理：按行聚类后从左到右排序，适配旋转矫正后的图片
- Web 页面：上传图片、查看叠加框、文字表格、区域表格、排版文本和原始 JSON

运行时推理统一使用 ONNX Runtime，不依赖 PaddlePaddle。

## 目录结构

```text
app/                 FastAPI 应用与 Web 页面
scripts/             模型下载脚本
models/              本地模型目录（首次准备后生成）
requirements.txt     运行时依赖
requirements-build.txt 模型准备依赖（与运行时一致）
```

## 环境要求

- 支持 Python 3.10 和 3.14；当前项目已在 Python 3.14 下验证
- 不需要 PaddlePaddle / paddle2onnx，也不需要做模型转换
- 推荐 8 GB 以上内存
- CPU 可运行；如有 ONNX Runtime GPU 环境，可配置 `ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`

所有 Python 依赖下载默认使用清华镜像：

```bash
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 准备模型

四个模型都直接从 ModelScope 下载现成 ONNX，不需要 PaddlePaddle，也不需要本地转换。

其中 PP-DocLayout-M ONNX 来自 ModelScope 的 `hzkitty/KittyDoc` 模型仓。已验证其输入输出与本项目一致：

- 输入：`image`、`scale_factor`
- 输出：检测框和数量
- 标签：与官方 PP-DocLayout-M 的 23 类一致

推荐一键准备：

```powershell
cd d:\doit\pp-ocr
powershell -ExecutionPolicy Bypass -File scripts/prepare_models.ps1
```

手动准备也可以：

```powershell
cd d:\doit\pp-ocr
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-build.txt
python scripts/download_models.py --models-dir ./models
python scripts/verify_setup.py
```

`verify_setup.py` 会加载并预热四个模型，输出 `ready=True` 即表示环境正常。

准备完成后模型目录应为：

```text
models/
  orientation/
    inference.onnx
    inference.yml
  layout/
    inference.onnx
    inference.yml
    config.json
  ocr/
    det/
      inference.onnx
      inference.yml
    rec/
      inference.onnx
      inference.yml
      keys.txt
```

模型来源：

- `PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx`：文档方向分类
- `hzkitty/KittyDoc`：PP-DocLayout-M ONNX
- `PaddlePaddle/PP-DocLayout-M`：版面类别与预处理配置
- `PaddlePaddle/PP-OCRv6_medium_det_onnx`
- `PaddlePaddle/PP-OCRv6_medium_rec_onnx`

## 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开：

- Web 页面：<http://127.0.0.1:8000/>
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

## API

### 综合识别

```bash
curl -X POST http://127.0.0.1:8000/v1/analyze \
  -F "file=@test.png" \
  -F "region_types=image,chart,header_image,footer_image" \
  -F "layout_threshold=0.5" \
  -F "ocr_threshold=0.5" \
  -F "include_text=true" \
  -F "text_in_region=annotate" \
  -F "region_association=hybrid" \
  -F "return_markdown=true"
```

### 只检测区域

```bash
curl -X POST http://127.0.0.1:8000/v1/layout -F "file=@test.png"
```

### 只识别文字

```bash
curl -X POST http://127.0.0.1:8000/v1/ocr -F "file=@test.png"
```

## 主要配置

复制 `.env.example` 为 `.env` 后按需修改：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `MODELS_DIR` | `./models` | 模型目录 |
| `STRICT_MODEL_LOAD` | `true` | 模型加载失败时是否终止启动 |
| `ONNX_PROVIDERS` | `CPUExecutionProvider` | ONNX Runtime providers |
| `MAX_UPLOAD_SIZE_MB` | `20` | 上传图片大小限制 |
| `DEFAULT_REGION_TYPES` | `image,chart,header_image,footer_image` | 默认区域类型 |
| `LAYOUT_BBOX_SCALE_MODE` | `auto` | 版面框坐标模式；如坐标偏小可尝试 `resized` |
| `OCR_LIMIT_SIDE_LEN` | `960` | OCR 检测长边；图片小、标点细或括号容易漏识别时建议调到 `1280` 或 `1600`，会更慢但更准 |
| `ENABLE_ORIENTATION_CORRECTION` | `true` | 识别前是否自动矫正图片方向 |
| `ORIENTATION_THRESHOLD` | `0.3` | 方向分类置信度阈值；低于阈值不矫正 |

无论是否执行旋转矫正，API 返回的 `bbox` 和 `polygon` 都会映射回用户上传的原图坐标系，Web 页面叠加框也会直接对应原图。

## Docker

```bash
docker build -t pp-ocr-api .
docker run --rm -p 8000:8000 pp-ocr-api
```

## 测试

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-dev.txt
pytest -q
```
