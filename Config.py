import os
import yaml

class Config:
    _instance = None
    _config = None

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "application.yml")
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    # ==================== 只读属性 ====================
    @property
    def service_host(self) -> str:
        return self._config.get("service", {}).get("host", "0.0.0.0")

    @property
    def service_port(self) -> int:
        return int(self._config.get("service", {}).get("port", 8000))

    @property
    def mediamtx_webrtc_base(self) -> str:
        return self._config.get("mediamtx", {}).get("webrtc_base", "http://127.0.0.1:8889")

    @property
    def default_model_path(self) -> str:
        return self._config.get("yolo", {}).get("default_model", "yolo11n.pt")

    @property
    def target_fps(self) -> int:
        return int(self._config.get("yolo", {}).get("target_fps", 15))

    @property
    def stream_connect_timeout(self) -> int:
        return int(self._config.get("yolo", {}).get("stream_connect_timeout", 5))

    @property
    def temp_img_dir(self) -> str:
        return self._config.get("temp_img", "./model_results")


# 全局单例获取函数
def get_config() -> Config:
    return Config()