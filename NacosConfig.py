from v2.nacos import NacosNamingService, ClientConfigBuilder, RegisterInstanceParam, DeregisterInstanceParam
from Config import get_config

config = get_config()

NACOS_SERVER_ADDRESSES = config.nacos_server_addr
NACOS_GROUP = config.nacos_group
SERVICE_NAME = config.nacos_service_name
SERVICE_IP = config.service_host
SERVICE_PORT = config.service_port

client_config = (ClientConfigBuilder()
                 .server_address(NACOS_SERVER_ADDRESSES)
                 .build())
async def register_service():
    """ 注册服务 """
    naming_client = await NacosNamingService.create_naming_service(client_config)
    res = await naming_client.register_instance(
        request=RegisterInstanceParam(
            service_name=SERVICE_NAME,
            ip=SERVICE_IP,
            port=SERVICE_PORT,
            enable=True,
            healthy=True,
        )
    )
    return res


async def deregister_service():
    """ 注销服务 """
    naming_client = await NacosNamingService.create_naming_service(client_config)
    res = await naming_client.deregister_instance(
        request=DeregisterInstanceParam(
            service_name=SERVICE_NAME,
            ip=SERVICE_IP,
            port=SERVICE_PORT
        )
    )
    return res
