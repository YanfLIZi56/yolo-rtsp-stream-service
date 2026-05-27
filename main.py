import time
import threading
import queue
import uuid
import re
import logging
from typing import Dict, Optional
from contextlib import asynccontextmanager
from datetime import datetime

import av
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from ultralytics import YOLO
import os

from Config import get_config

# ========== 加载配置 ==========
config = get_config()

# ========== 日志配置 ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-service")

# ========== 全局配置 ==========
TARGET_FPS = config.target_fps
DEFAULT_MODEL_PATH = config.default_model_path  # 默认模型路径
STREAM_CONNECT_TIMEOUT = config.stream_connect_timeout  # 拉流超时（秒）
MEDIAMTX_WEBRTC_BASE = config.mediamtx_webrtc_base  # 请改为实际 MediaMTX 地址
TEMP_IMG = config.temp_img_dir

# ========== 模型池管理 ==========
default_model: Optional[YOLO] = None
model_pool: Dict[str, YOLO] = {}  # model_path -> YOLO 实例
model_pool_lock = threading.Lock()


def get_model(model_path: Optional[str] = None) -> YOLO:
    """获取模型实例，支持按路径加载和缓存"""
    if not model_path:
        return default_model
    if model_path not in model_pool:
        with model_pool_lock:
            if model_path not in model_pool:
                logger.info(f"加载新模型: {model_path}")
                model_pool[model_path] = YOLO(model_path)
    return model_pool[model_path]


# ========== 流处理组件 ==========
# 推理队列：(stream_id, model_path, img)
inference_queue = queue.Queue(maxsize=20)
results_dict: Dict[str, np.ndarray] = {}  # stream_id -> 最新推理结果
results_lock = threading.Lock()

# 活跃的流处理器映射
stream_processors: Dict[str, 'StreamProcessor'] = {}
stream_lock = threading.Lock()

# 全局运行标志
global_running = True


# ========== 推理工作线程 ==========
def inference_worker():
    """全局推理线程，从队列取帧，用指定模型推理，结果放入字典"""
    while global_running:
        try:
            stream_id, model_path, img = inference_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            selected_model = get_model(model_path)
            results = selected_model(img, verbose=False)
            annotated = results[0].plot()  # BGR 图像
            with results_lock:
                results_dict[stream_id] = annotated
        except Exception as e:
            logger.error(f"推理错误 stream={stream_id}: {e}")
        finally:
            inference_queue.task_done()


# ========== 单路流处理器 ==========
class StreamProcessor:
    def __init__(self, stream_id: str, source_url: str, duration: int, model_path: Optional[str] = None):
        self.stream_id = stream_id
        self.source_url = source_url
        self.duration = duration
        self.model_path = model_path  # 该流使用的模型路径
        self.target_url = self._build_target_url(source_url)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.started_successfully = False

        self.capture_thread = None
        self.push_thread = None
        self.timer = None

    @staticmethod
    def _build_target_url(source_url: str) -> str:
        source = source_url.rstrip('/')
        return f"{source}/yolo"

    @staticmethod
    def _parse_webrtc_path(rtsp_url: str) -> Optional[str]:
        m = re.match(r'rtsp://[^/]+(/.*)', rtsp_url)
        return m.group(1) if m else None

    def get_webrtc_url(self) -> str:
        path = self._parse_webrtc_path(self.target_url)
        if not path:
            return ""
        return f"{MEDIAMTX_WEBRTC_BASE}{path}"

    def capture_loop(self):
        """拉流线程"""
        try:
            logger.info(f"[{self.stream_id}] 开始拉流: {self.source_url}")
            container = av.open(self.source_url, options={
                'rtsp_transport': 'tcp',
                'stimeout': str(STREAM_CONNECT_TIMEOUT * 1_000_000)
            })
            video_stream = container.streams.video[0]
            self.started_successfully = True

            while self.running and global_running:
                for packet in container.demux(video_stream):
                    for frame in packet.decode():
                        img = frame.to_ndarray(format='bgr24')
                        with self.frame_lock:
                            self.latest_frame = img
                        # 放入推理队列（携带模型路径）
                        try:
                            inference_queue.put_nowait((self.stream_id, self.model_path, img))
                        except queue.Full:
                            pass
                        if not self.running or not global_running:
                            break
                if not self.running or not global_running:
                    break
            container.close()
        except Exception as e:
            logger.error(f"[{self.stream_id}] 拉流异常: {e}")
            self.started_successfully = False
        finally:
            logger.info(f"[{self.stream_id}] 拉流线程退出")
            self.running = False

    def push_loop(self):
        """推流线程"""
        try:
            output_container = av.open(self.target_url, 'w', format='rtsp',
                                       options={'rtsp_transport': 'tcp'})
            output_stream = output_container.add_stream('h264', rate=TARGET_FPS)
            output_stream.width = 640
            output_stream.height = 480
            output_stream.pix_fmt = 'yuv420p'
            output_stream.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'crf': '23',
                'profile': 'baseline'
            }
            frame_interval = 1.0 / TARGET_FPS
            last_push_time = 0

            while self.running and global_running:
                with results_lock:
                    annotated = results_dict.pop(self.stream_id, None)
                if annotated is None:
                    time.sleep(0.005)
                    continue

                now = time.time()
                if now - last_push_time >= frame_interval:
                    new_frame = av.VideoFrame.from_ndarray(annotated, format='bgr24')
                    new_frame.pts = None
                    for packet in output_stream.encode(new_frame):
                        output_container.mux(packet)
                    last_push_time = now

            # 清理
            if output_stream:
                for packet in output_stream.encode():
                    output_container.mux(packet)
            output_container.close()
        except Exception as e:
            logger.error(f"[{self.stream_id}] 推流异常: {e}")
        finally:
            logger.info(f"[{self.stream_id}] 推流线程退出")

    def start(self) -> bool:
        """启动拉流/推流线程，返回是否成功"""
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

        # 等待第一帧成功（或超时）
        start_time = time.time()
        while time.time() - start_time < STREAM_CONNECT_TIMEOUT:
            if self.started_successfully:
                break
            if self.capture_thread.is_alive() is False:
                return False
            time.sleep(0.1)
        else:
            self.stop()
            return False

        self.push_thread = threading.Thread(target=self.push_loop, daemon=True)
        self.push_thread.start()

        if self.duration > 0:
            self.timer = threading.Timer(self.duration, self.stop)
            self.timer.daemon = True
            self.timer.start()

        logger.info(f"[{self.stream_id}] 流已启动，推流: {self.target_url}")
        return True

    def stop(self):
        """停止拉流和推流"""
        if not self.running:
            return
        logger.info(f"[{self.stream_id}] 正在停止...")
        self.running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        if self.push_thread and self.push_thread.is_alive():
            self.push_thread.join(timeout=2)
        if self.timer:
            self.timer.cancel()

        with stream_lock:
            if self.stream_id in stream_processors:
                del stream_processors[self.stream_id]

        with results_lock:
            results_dict.pop(self.stream_id, None)

        logger.info(f"[{self.stream_id}] 已停止")


# ========== FastAPI 应用生命周期 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global default_model, global_running

    # 加载默认模型
    try:
        default_model = YOLO(DEFAULT_MODEL_PATH)
        logger.info(f"默认模型已加载: {DEFAULT_MODEL_PATH}")
    except Exception as e:
        logger.error(f"默认模型加载失败: {e}")
        raise e

    # 启动推理线程
    infer_thread = threading.Thread(target=inference_worker, daemon=True)
    infer_thread.start()
    logger.info("全局推理线程已启动")
    yield

    # 关闭时清理
    global_running = False
    with stream_lock:
        for sp in list(stream_processors.values()):
            sp.stop()


app = FastAPI(title="YOLO 检测与流服务", version="2.0", lifespan=lifespan)


# ========== 图片检测接口 ==========
@app.post("/detect")
async def detect(
        file: UploadFile = File(...),
        conf: float = Form(default=0.25, description="置信度阈值，范围(0, 1]"),
        model_path: str = Form(default="../yolo11n.pt", description="模型路径")
):
    if conf > 1 or conf <= 0:
        raise HTTPException(status_code=400, detail="置信度阈值范围是(0, 1]")

    # 读取图片
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 推理
    current_model = get_model(model_path)  # 使用模型池
    start_time = time.time()
    results = current_model(img, conf=conf, verbose=False)
    detect_time = round(time.time() - start_time, 4)

    # 封装结果
    detections = []
    for result in results:
        for box in result.boxes:
            class_name = current_model.names[int(box.cls)]
            detections.append({
                "class": class_name,
                "conf": round(float(box.conf), 4),
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 2),
                    "y1": round(float(box.xyxy[0][1]), 2),
                    "x2": round(float(box.xyxy[0][2]), 2),
                    "y2": round(float(box.xyxy[0][3]), 2)
                }
            })

    # 保存结果图
    try:
        now = datetime.now()
        date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
        full_dir = os.path.join(TEMP_IMG, date_path)
        os.makedirs(full_dir, exist_ok=True)
        file_name = f'{uuid.uuid4()}.jpg'
        out_path = f'{date_path}/{file_name}'
        results[0].save(filename=os.path.join(full_dir, file_name))
    except Exception as e:
        logger.warning(f"保存结果图失败: {e}")
        out_path = 'no-data.jpg'

    return {
        "code": 0,
        "msg": "success",
        "filename": file.filename,
        "result": out_path,
        "count": len(detections),
        "detections": detections,
        "time": detect_time
    }


# ========== 流处理接口 ==========
class StreamRequest(BaseModel):
    rtsp_url: str = Field(..., description="源 RTSP 流地址")
    duration: int = Field(0, ge=0, description="持续时间（秒），0 表示无限")
    model_path: Optional[str] = Field(None, description="模型文件路径，不传则使用默认模型")


class StreamResponse(BaseModel):
    status: str
    stream_id: Optional[str] = None
    rtsp_url: Optional[str] = None
    webrtc_url: Optional[str] = None
    message: Optional[str] = None


@app.post("/streams", response_model=StreamResponse)
def create_stream(req: StreamRequest):
    source = req.rtsp_url.strip()
    duration = req.duration
    model_path = req.model_path

    # 检查是否已存在相同的源+模型
    with stream_lock:
        for sp in stream_processors.values():
            if sp.source_url == source and sp.model_path == model_path:
                return StreamResponse(
                    status="success",
                    stream_id=sp.stream_id,
                    rtsp_url=sp.target_url,
                    webrtc_url=sp.get_webrtc_url(),
                    message="Stream already exists"
                )

    stream_id = str(uuid.uuid4())[:8]
    processor = StreamProcessor(stream_id, source, duration, model_path)

    if not processor.start():
        return StreamResponse(status="error", message="Failed to connect to source stream or timeout")

    with stream_lock:
        stream_processors[stream_id] = processor

    return StreamResponse(
        status="success",
        stream_id=stream_id,
        rtsp_url=processor.target_url,
        webrtc_url=processor.get_webrtc_url()
    )


@app.delete("/streams/{stream_id}", response_model=StreamResponse)
def stop_stream(stream_id: str):
    with stream_lock:
        processor = stream_processors.get(stream_id)
    if not processor:
        raise HTTPException(status_code=404, detail="Stream not found")
    processor.stop()
    return StreamResponse(status="success", message=f"Stream {stream_id} stopped")


@app.get("/streams")
def list_streams():
    with stream_lock:
        streams = []
        for sp in stream_processors.values():
            streams.append({
                "stream_id": sp.stream_id,
                "source_url": sp.source_url,
                "target_url": sp.target_url,
                "webrtc_url": sp.get_webrtc_url(),
                "model_path": sp.model_path or DEFAULT_MODEL_PATH,
                "running": sp.running
            })
    return streams


# ========== 健康检查 ==========
@app.get("/api/health")
async def health():
    return {"status": "running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.service_host, port=config.service_port)
