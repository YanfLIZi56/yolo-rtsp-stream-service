import logging
import threading
from v2.nacos import NacosNamingService, ClientConfigBuilder, RegisterInstanceParam, DeregisterInstanceParam
from Config import get_config

config = get_config()
logger = logging.getLogger("nacos-config")

class NacosManager:
    """Nacos 服务注册管理器（单例模式）"""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.naming_client = None
        self.enabled = config.nacos_enabled

        if not self.enabled:
            return

        self._init_nacos()

    def _init_nacos(self):
        """初始化 Nacos 客户端配置"""
        self.server_addresses = config.nacos_server_addr
        self.group = config.nacos_group
        self.service_name = config.nacos_service_name
        self.service_ip = config.service_host
        self.service_port = config.service_port

        self.client_config = (ClientConfigBuilder()
                              .server_address(self.server_addresses)
                              .build())

    async def _ensure_client(self):
        """确保 Nacos 客户端已创建"""
        if not self.enabled:
            return None

        if self.naming_client is None:
            self.naming_client = await NacosNamingService.create_naming_service(self.client_config)

        return self.naming_client

    async def register_service(self):
        """注册服务到 Nacos"""
        if not self.enabled:
            return None

        try:
            naming_client = await self._ensure_client()
            res = await naming_client.register_instance(
                request=RegisterInstanceParam(
                    service_name=self.service_name,
                    ip=self.service_ip,
                    port=self.service_port,
                    enable=True,
                    healthy=True,
                )
            )
            logger.info(f"========= Nacos 服务注册成功 =========")
            return res
        except Exception as e:
            logger.error(f"========= Nacos 注册失败: {e} =========")

    async def deregister_service(self):
        """从 Nacos 注销服务"""
        if not self.enabled:
            return None

        try:
            naming_client = await self._ensure_client()
            res = await naming_client.deregister_instance(
                request=DeregisterInstanceParam(
                    service_name=self.service_name,
                    ip=self.service_ip,
                    port=self.service_port
                )
            )
            logger.info("========= Nacos 注销成功 =========")
            return res
        except Exception as e:
            logger.error(f"========= Nacos 注销失败: {e} =========")

    def is_enabled(self) -> bool:
        """检查 Nacos 是否启用"""
        return self.enabled


# 全局单例获取函数
def get_nacos_manager() -> NacosManager:
    """获取 Nacos 管理器单例"""
    return NacosManager()
