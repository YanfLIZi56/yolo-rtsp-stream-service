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
    def nacos_enabled(self) -> bool:
        return self._config.get("nacos", {}).get("enabled", False)

    @property
    def nacos_server_addr(self) -> str:
        return self._config.get("nacos", {}).get("server_addr", "127.0.0.1:8848")

    @property
    def nacos_group(self) -> str:
        return self._config.get("nacos", {}).get("group", "DEFAULT_GROUP")

    @property
    def nacos_service_name(self) -> str:
        return self._config.get("nacos", {}).get("service_name", "py-service")

    @property
    def mediamtx_host(self) -> str:
        return self._config.get("mediamtx", {}).get("host", "127.0.0.1")

    @property
    def mediamtx_rtsp_port(self) -> int:
        return int(self._config.get("mediamtx", {}).get("rtsp_port", 8554))

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
    def frame_skip(self) -> int:
        return int(self._config.get("yolo", {}).get("frame_skip", 1))

    @property
    def rabbitmq_host(self) -> str:
        return self._config.get("rabbitmq", {}).get("host", "127.0.0.1")

    @property
    def rabbitmq_port(self) -> int:
        return int(self._config.get("rabbitmq", {}).get("port", 5672))

    @property
    def rabbitmq_virtual_host(self) -> str:
        return self._config.get("rabbitmq", {}).get("virtual_host", "/")

    @property
    def rabbitmq_user(self) -> str:
        return self._config.get("rabbitmq", {}).get("username", "guest")

    @property
    def rabbitmq_pass(self) -> str:
        return self._config.get("rabbitmq", {}).get("password", "guest")

    @property
    def rabbitmq_exchange(self) -> str:
        return self._config.get("rabbitmq", {}).get("exchange", "yolo.direct.exchange")

    @property
    def rabbitmq_routing_key(self) -> str:
        return self._config.get("rabbitmq", {}).get("routing_key", "yolo.statistics")

    @property
    def rabbitmq_error_routing_key(self) -> str:
        return self._config.get("rabbitmq", {}).get("error_routing_key", "yolo.error")

    @property
    def rabbitmq_end_routing_key(self) -> str:
        return self._config.get("rabbitmq", {}).get("end_routing_key", "yolo.end")

    @property
    def rabbitmq_stats_interval(self) -> int:
        return int(self._config.get("rabbitmq", {}).get("stats_interval", 30))

    @property
    def temp_img_dir(self) -> str:
        return self._config.get("temp_img", "./model_results")

    @property
    def origin_img_dir(self) -> str:
        return self._config.get("origin_img", "./origin_img")


# 全局单例获取函数
def get_config() -> Config:
    return Config()