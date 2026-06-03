import pika
import threading
from Config import get_config

config = get_config()


class RabbitMQConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_pool()
        return cls._instance

    def _init_pool(self):
        self.connection_params = pika.ConnectionParameters(
            host=config.rabbitmq_host,
            port=config.rabbitmq_port,
            virtual_host=config.rabbitmq_virtual_host,
            credentials=pika.PlainCredentials(
                config.rabbitmq_user, config.rabbitmq_pass
            )
        )

    def get_channel(self):
        """每个线程调用此方法获取独立 channel"""
        connection = pika.BlockingConnection(self.connection_params)
        channel = connection.channel()
        # 声明交换机（确保存在）
        channel.exchange_declare(
            exchange=config.rabbitmq_exchange,
            exchange_type='direct',
            durable=True
        )
        return connection, channel


# 全局单例
rabbitmq_pool = RabbitMQConnectionPool()
