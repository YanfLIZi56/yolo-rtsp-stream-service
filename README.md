# YOLO RTSP 实时检测与推流服务

基于 FastAPI、PyAV、Ultralytics YOLO 和 MediaMTX 的实时视频流目标检测服务。  
支持图片检测接口和动态创建/停止 RTSP 流的 YOLO 处理任务，并将结果推送为 WebRTC 可播放的流。

## 功能特性

- **图片检测**：上传图片进行 YOLO 推理，返回检测结果和标注图。
- **实时流处理**：传入 RTSP 地址，自动拉流、YOLO 检测、画框推流。
- **多模型支持**：每个流可指定不同的 YOLO 模型文件。
- **自动停流**：支持设置运行时长，到时自动停止推流。
- **动态管理**：提供流列表查询、手动停止等 REST 接口。
- **配置集中化**：所有配置项在 `config.yml` 中修改。

## 项目结构

```
├── main.py           # FastAPI 主程序
├── Config.py         # 配置管理（读取 config.yml）
├── application.yml   # 应用配置文件
├── NacosConfig.py    # Nacos 服务注册（可选）
├── RabbitmqConfig.py    # RabbitMQ 配置
├── requirements.txt  # Python 依赖
└── README.md         # 本文件
```
## 环境要求

- Python 3.10+
- FFmpeg（系统需安装，用于视频编解码）
- MediaMTX 服务（用于 RTSP 流转 WebRTC）
- Nacos
- RabbitMQ

( 如果不想要Nacos和RabbitMQ，可以看看首次提交的代码 )

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 MediaMTX

#### 1.docker部署
确保 MediaMTX 正在运行，并可通过 RTSP 和 WebRTC 访问。
建议使用 `--network host` 模式启动容器：

```bash
docker run -d --name mediamtx --network host bluenviron/mediamtx
```

#### 2. 运行二进制文件

**下载地址：** (https://github.com/bluenviron/mediamtx/releases)

**安装步骤：**
1. 从上方链接下载对应系统的压缩包
2. 解压到本地目录，得到 `mediamtx` 文件夹
3. 进入该目录并启动服务

### 3. 修改配置文件

编辑 `config.yml`，将 `mediamtx.host` 和 `mediamtx.port` 改为实际rtsp推流的 MediaMTX 地址：

```yaml
mediamtx:
  host: "你的IP"
  port: 8554
```

其他配置如端口、帧率、默认模型路径可按需调整。

### 4. 启动服务

```bash
python main.py
```

服务将运行在 `http://0.0.0.0:8000`（可在 `config.yml` 中修改）。

## 接口说明

### 1. 图片检测

**POST** `/detect`

- **Content-Type**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 上传的图片文件 |
| conf | float | 否 | 置信度阈值 (默认 0.25) |
| model_path | str | 否 | 模型路径 (默认用配置的模型) |

**示例**：

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test.jpg" \
  -F "conf=0.5"
```

### 2. 创建实时流

**POST** `/streams`

- **Content-Type**: `application/json`

```json
{
  "rtsp_url": "rtsp://摄像头地址/路径",
  "duration": 1800,
  "model_path": null,
  "device_id": 1
}
```
| 字段 | 类型          | 必填 | 说明                   |
|------|-------------|----|----------------------|
| rtsp_url | string      | 是  | 源 RTSP 流地址           |
| duration | int         | 否  | 运行秒数，0 表示无限 (直到手动停止) |
| model_path | string/null | 否  | 模型文件路径，null 则使用默认模型  |
|  device_id          | int/null    | 是  | 设备id, 可选             |

**成功响应**：

```json
{
  "status": "success",
  "stream_id": "a1b2c3d4",
  "rtsp_url": "rtsp://.../原路径/yolo",
  "webrtc_url": "原路径/yolo"
}
```

返回的 `webrtc_url`只有路径, 需要反向代理

推流成功后，可在浏览器中打开 `mediamtx的IP:8889/原路径/yolo` 观看实时检测画面。

### 3. 查看所有流

**GET** `/streams`

### 4. 停止指定流

**DELETE** `/streams/{stream_id}`

### 5. 健康检查

**GET** `/api/health`

## 配置说明

所有业务参数均在 `config.yml` 中，核心项：

| 配置项                         | 说明                  | 默认值                   |
|-----------------------------|---------------------|-----------------------|
| service.host                | 服务监听地址              | 0.0.0.0               |
| service.port                | 服务监听端口              | 8000                  |
| nacos.server_addr           | nacos服务地址           | http://127.0.0.1:8848 |
| nacos.group                 | nacos注册中心的组         | DEFAULT_GROUP         |
| nacos.service_name          | 注册的服务名              | py-service            |
| mediamtx.host               | rtsp推流的 MediaMTX 地址 | 127.0.0.1             |
| mediamtx.port               | rtsp推流的 MediaMTX 端口 | 8554                  |
| yolo.default_model          | 默认 YOLO 模型路径        | ../yolo11n.pt         |
| yolo.target_fps             | 推流帧率                | 15                    |
| yolo.stream_connect_timeout | 拉流超时(秒)             | 5                     |
| rabbitmq.host               | RabbitMQ 服务器地址      | 127.0.0.1             |
| rabbitmq.port               | RabbitMQ 端口         | 5672                  |
| rabbitmq.virtual_host       | RabbitMQ 虚拟主机       | /                     |
| rabbitmq.username           | RabbitMQ 用户名        | guest                 |
| rabbitmq.password           | RabbitMQ 密码         | guest                 |
| rabbitmq.exchange           | RabbitMQ 交换机名称      | yolo.direct.exchange  |
| rabbitmq.routing_key        | RabbitMQ 路由键        | yolo.statistics       |
| rabbitmq.stats_interval     | 统计发送间隔（秒）           | 30                    |
| temp_img                    | 图片检测结果保存目录          | ./model_results       |

## 注意事项

- 确保 MediaMTX 与当前服务网络互通，且已开放 8554 (RTSP) 和 8889 (WebRTC) 端口。
- 若使用 GPU 加速，请在运行环境设置 `CUDA_VISIBLE_DEVICES`。
- 停止服务时，所有正在进行的流处理任务会自动终止。
