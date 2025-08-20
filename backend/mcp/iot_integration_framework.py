"""
IoT Device Integration Framework - 40by6
Connect, manage, and orchestrate IoT devices with MCP Stack
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import struct
import hashlib
import hmac
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import paho.mqtt.client as mqtt
import asyncio_mqtt
from azure.iot.device.aio import IoTHubDeviceClient, ProvisioningDeviceClient
from azure.iot.device import Message, MethodResponse
from awscrt import io, mqtt as aws_mqtt, auth, http
from awsiot import mqtt_connection_builder
import websockets
import aiocoap
import aiocoap.resource as resource
from opcua import Client as OPCUAClient, ua
from asyncua import Client as AsyncOPCUAClient
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp, modbus_rtu
import serial
import can
import bluetooth
import zigpy
import zha
from bleak import BleakClient, BleakScanner
import socket
import ssl
import certifi
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, JSON, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge, Summary
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import ray
import edge_tpu
from pycoral.utils import edgetpu
from pycoral.adapters import common, classify, detect
import cv2
import pyaudio
import wave
from scipy import signal
import requests
import aiohttp
import msgpack
import cbor2
import protobuf
import avro
import struct
import zlib
import lz4.frame
from typing_extensions import Protocol

logger = logging.getLogger(__name__)

# Metrics
iot_devices_connected = Gauge('iot_devices_connected', 'Number of connected IoT devices', ['device_type', 'protocol'])
iot_messages_received = Counter('iot_messages_received_total', 'Total IoT messages received', ['device_id', 'message_type'])
iot_messages_sent = Counter('iot_messages_sent_total', 'Total IoT messages sent', ['device_id', 'message_type'])
iot_data_bytes = Counter('iot_data_bytes_total', 'Total bytes of IoT data', ['direction', 'protocol'])
iot_latency = Histogram('iot_message_latency_seconds', 'IoT message latency', ['device_type'])
iot_errors = Counter('iot_errors_total', 'Total IoT errors', ['error_type', 'device_type'])

Base = declarative_base()


class IoTProtocol(Enum):
    """Supported IoT protocols"""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    WEBSOCKET = "websocket"
    OPCUA = "opcua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    CAN_BUS = "can_bus"
    BLUETOOTH = "bluetooth"
    BLUETOOTH_LE = "bluetooth_le"
    ZIGBEE = "zigbee"
    ZWAVE = "zwave"
    LORA = "lora"
    NB_IOT = "nb_iot"
    SIGFOX = "sigfox"
    CUSTOM = "custom"


class DeviceType(Enum):
    """Types of IoT devices"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    EDGE_COMPUTER = "edge_computer"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    DISPLAY = "display"
    ROBOT = "robot"
    DRONE = "drone"
    VEHICLE = "vehicle"
    WEARABLE = "wearable"
    SMART_METER = "smart_meter"
    INDUSTRIAL = "industrial"
    MEDICAL = "medical"
    AGRICULTURAL = "agricultural"


class DataFormat(Enum):
    """IoT data formats"""
    JSON = "json"
    MSGPACK = "msgpack"
    CBOR = "cbor"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    BINARY = "binary"
    TEXT = "text"
    CSV = "csv"
    XML = "xml"


class SecurityMode(Enum):
    """IoT security modes"""
    NONE = "none"
    TLS = "tls"
    DTLS = "dtls"
    PSK = "psk"  # Pre-shared key
    CERTIFICATE = "certificate"
    TOKEN = "token"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


@dataclass
class IoTDevice:
    """IoT device representation"""
    id: str
    name: str
    type: DeviceType
    protocol: IoTProtocol
    connection_string: str
    location: Optional[Dict[str, float]] = None  # lat, lon, alt
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    security: SecurityMode = SecurityMode.TLS
    data_format: DataFormat = DataFormat.JSON
    last_seen: Optional[datetime] = None
    is_online: bool = False
    firmware_version: Optional[str] = None
    hardware_version: Optional[str] = None
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None
    error_count: int = 0
    message_count: int = 0
    data_points: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'protocol': self.protocol.value,
            'location': self.location,
            'capabilities': self.capabilities,
            'metadata': self.metadata,
            'is_online': self.is_online,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'firmware_version': self.firmware_version,
            'battery_level': self.battery_level,
            'signal_strength': self.signal_strength,
            'error_count': self.error_count,
            'message_count': self.message_count
        }


@dataclass
class IoTMessage:
    """IoT message"""
    device_id: str
    timestamp: datetime
    message_type: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    qos: int = 0
    retained: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'device_id': self.device_id,
            'timestamp': self.timestamp.isoformat(),
            'message_type': self.message_type,
            'payload': self.payload,
            'metadata': self.metadata,
            'qos': self.qos,
            'retained': self.retained
        }


@dataclass
class IoTCommand:
    """Command to send to IoT device"""
    device_id: str
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    timeout: float = 30.0
    retry_count: int = 3


class IoTDeviceRegistry(Base):
    """Database model for device registry"""
    __tablename__ = 'iot_devices'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    protocol = Column(String(50), nullable=False)
    connection_string = Column(Text, nullable=False)
    security_mode = Column(String(50), default='tls')
    location_lat = Column(Float)
    location_lon = Column(Float)
    location_alt = Column(Float)
    capabilities = Column(JSON)
    metadata = Column(JSON)
    firmware_version = Column(String(50))
    hardware_version = Column(String(50))
    last_seen = Column(DateTime)
    is_online = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_iot_device_type', 'type'),
        Index('idx_iot_device_protocol', 'protocol'),
        Index('idx_iot_device_online', 'is_online'),
    )


class IoTDataPoint(Base):
    """Database model for IoT data points"""
    __tablename__ = 'iot_data_points'
    
    id = Column(Integer, primary_key=True)
    device_id = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True)
    data_type = Column(String(50))
    value = Column(JSON)
    unit = Column(String(20))
    quality = Column(Float, default=1.0)
    
    __table_args__ = (
        Index('idx_iot_data_device_time', 'device_id', 'timestamp'),
    )


class ProtocolHandler(Protocol):
    """Protocol handler interface"""
    
    async def connect(self, device: IoTDevice) -> bool:
        """Connect to device"""
        ...
    
    async def disconnect(self, device_id: str) -> bool:
        """Disconnect from device"""
        ...
    
    async def send_message(self, device_id: str, message: Any) -> bool:
        """Send message to device"""
        ...
    
    async def receive_message(self) -> Optional[IoTMessage]:
        """Receive message from device"""
        ...


class MQTTHandler(ProtocolHandler):
    """MQTT protocol handler"""
    
    def __init__(self, broker_url: str, client_id: str = None):
        self.broker_url = broker_url
        self.client_id = client_id or f"mcp_{uuid.uuid4().hex[:8]}"
        self.client = None
        self.connected_devices: Dict[str, IoTDevice] = {}
        self.message_queue = asyncio.Queue()
    
    async def connect(self, device: IoTDevice) -> bool:
        """Connect to MQTT broker"""
        try:
            if not self.client:
                self.client = asyncio_mqtt.Client(
                    hostname=self.broker_url,
                    client_id=self.client_id
                )
                await self.client.connect()
                
                # Start message receiver
                asyncio.create_task(self._receive_loop())
            
            # Subscribe to device topic
            topic = f"devices/{device.id}/+"
            await self.client.subscribe(topic)
            
            self.connected_devices[device.id] = device
            iot_devices_connected.labels(device.type.value, 'mqtt').inc()
            
            logger.info(f"Connected MQTT device: {device.id}")
            return True
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            iot_errors.labels('connection', device.type.value).inc()
            return False
    
    async def disconnect(self, device_id: str) -> bool:
        """Disconnect MQTT device"""
        try:
            if device_id in self.connected_devices:
                # Unsubscribe from device topic
                topic = f"devices/{device_id}/+"
                await self.client.unsubscribe(topic)
                
                device = self.connected_devices.pop(device_id)
                iot_devices_connected.labels(device.type.value, 'mqtt').dec()
                
                logger.info(f"Disconnected MQTT device: {device_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"MQTT disconnect failed: {e}")
            return False
    
    async def send_message(self, device_id: str, message: Any) -> bool:
        """Send MQTT message"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            device = self.connected_devices[device_id]
            
            # Serialize message
            if device.data_format == DataFormat.JSON:
                payload = json.dumps(message)
            elif device.data_format == DataFormat.MSGPACK:
                payload = msgpack.packb(message)
            elif device.data_format == DataFormat.CBOR:
                payload = cbor2.dumps(message)
            else:
                payload = str(message)
            
            # Publish message
            topic = f"devices/{device_id}/commands"
            await self.client.publish(topic, payload, qos=1)
            
            iot_messages_sent.labels(device_id, 'command').inc()
            iot_data_bytes.labels('sent', 'mqtt').inc(len(payload))
            
            return True
            
        except Exception as e:
            logger.error(f"MQTT send failed: {e}")
            iot_errors.labels('send', 'mqtt').inc()
            return False
    
    async def receive_message(self) -> Optional[IoTMessage]:
        """Get message from queue"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def _receive_loop(self):
        """Receive messages from MQTT"""
        async with self.client.filtered_messages("devices/+/+") as messages:
            async for message in messages:
                try:
                    # Parse topic
                    parts = message.topic.split('/')
                    if len(parts) >= 3:
                        device_id = parts[1]
                        message_type = parts[2]
                        
                        # Deserialize payload
                        if device_id in self.connected_devices:
                            device = self.connected_devices[device_id]
                            
                            if device.data_format == DataFormat.JSON:
                                payload = json.loads(message.payload)
                            elif device.data_format == DataFormat.MSGPACK:
                                payload = msgpack.unpackb(message.payload)
                            elif device.data_format == DataFormat.CBOR:
                                payload = cbor2.loads(message.payload)
                            else:
                                payload = message.payload.decode()
                            
                            # Create IoT message
                            iot_msg = IoTMessage(
                                device_id=device_id,
                                timestamp=datetime.utcnow(),
                                message_type=message_type,
                                payload=payload,
                                qos=message.qos
                            )
                            
                            await self.message_queue.put(iot_msg)
                            
                            iot_messages_received.labels(device_id, message_type).inc()
                            iot_data_bytes.labels('received', 'mqtt').inc(len(message.payload))
                        
                except Exception as e:
                    logger.error(f"MQTT receive error: {e}")
                    iot_errors.labels('receive', 'mqtt').inc()


class CoAPHandler(ProtocolHandler):
    """CoAP protocol handler"""
    
    def __init__(self, bind_address: str = "0.0.0.0", port: int = 5683):
        self.bind_address = bind_address
        self.port = port
        self.protocol = None
        self.connected_devices: Dict[str, IoTDevice] = {}
        self.message_queue = asyncio.Queue()
    
    async def connect(self, device: IoTDevice) -> bool:
        """Start CoAP server"""
        try:
            if not self.protocol:
                # Create CoAP protocol
                await self._start_server()
            
            self.connected_devices[device.id] = device
            iot_devices_connected.labels(device.type.value, 'coap').inc()
            
            logger.info(f"Registered CoAP device: {device.id}")
            return True
            
        except Exception as e:
            logger.error(f"CoAP registration failed: {e}")
            iot_errors.labels('connection', device.type.value).inc()
            return False
    
    async def _start_server(self):
        """Start CoAP server"""
        # Create CoAP context
        root = resource.Site()
        
        # Add resource for device communication
        root.add_resource(['devices'], DeviceResource(self))
        
        # Create protocol
        await aiocoap.Context.create_server_context(
            root,
            bind=(self.bind_address, self.port)
        )
        
        logger.info(f"CoAP server started on {self.bind_address}:{self.port}")
    
    async def disconnect(self, device_id: str) -> bool:
        """Unregister CoAP device"""
        if device_id in self.connected_devices:
            device = self.connected_devices.pop(device_id)
            iot_devices_connected.labels(device.type.value, 'coap').dec()
            logger.info(f"Unregistered CoAP device: {device_id}")
        return True
    
    async def send_message(self, device_id: str, message: Any) -> bool:
        """Send CoAP message"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            device = self.connected_devices[device_id]
            
            # Create CoAP client
            protocol = await aiocoap.Context.create_client_context()
            
            # Serialize message
            if device.data_format == DataFormat.JSON:
                payload = json.dumps(message).encode()
            elif device.data_format == DataFormat.CBOR:
                payload = cbor2.dumps(message)
            else:
                payload = str(message).encode()
            
            # Send request
            request = aiocoap.Message(
                code=aiocoap.PUT,
                uri=f"coap://{device.connection_string}/command",
                payload=payload
            )
            
            response = await protocol.request(request).response
            
            iot_messages_sent.labels(device_id, 'command').inc()
            iot_data_bytes.labels('sent', 'coap').inc(len(payload))
            
            return response.code.is_successful()
            
        except Exception as e:
            logger.error(f"CoAP send failed: {e}")
            iot_errors.labels('send', 'coap').inc()
            return False
    
    async def receive_message(self) -> Optional[IoTMessage]:
        """Get message from queue"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None


class DeviceResource(resource.Resource):
    """CoAP resource for device communication"""
    
    def __init__(self, handler: CoAPHandler):
        super().__init__()
        self.handler = handler
    
    async def render_post(self, request):
        """Handle POST requests from devices"""
        try:
            # Extract device ID from URI
            device_id = request.opt.uri_path[1] if len(request.opt.uri_path) > 1 else None
            
            if device_id and device_id in self.handler.connected_devices:
                device = self.handler.connected_devices[device_id]
                
                # Deserialize payload
                if device.data_format == DataFormat.JSON:
                    payload = json.loads(request.payload.decode())
                elif device.data_format == DataFormat.CBOR:
                    payload = cbor2.loads(request.payload)
                else:
                    payload = request.payload.decode()
                
                # Create IoT message
                message = IoTMessage(
                    device_id=device_id,
                    timestamp=datetime.utcnow(),
                    message_type='telemetry',
                    payload=payload
                )
                
                await self.handler.message_queue.put(message)
                
                iot_messages_received.labels(device_id, 'telemetry').inc()
                iot_data_bytes.labels('received', 'coap').inc(len(request.payload))
                
                return aiocoap.Message(code=aiocoap.CHANGED)
            
            return aiocoap.Message(code=aiocoap.NOT_FOUND)
            
        except Exception as e:
            logger.error(f"CoAP receive error: {e}")
            iot_errors.labels('receive', 'coap').inc()
            return aiocoap.Message(code=aiocoap.INTERNAL_SERVER_ERROR)


class OPCUAHandler(ProtocolHandler):
    """OPC UA protocol handler"""
    
    def __init__(self):
        self.clients: Dict[str, AsyncOPCUAClient] = {}
        self.connected_devices: Dict[str, IoTDevice] = {}
        self.message_queue = asyncio.Queue()
        self.subscriptions = {}
    
    async def connect(self, device: IoTDevice) -> bool:
        """Connect to OPC UA server"""
        try:
            # Create OPC UA client
            client = AsyncOPCUAClient(device.connection_string)
            
            # Configure security
            if device.security == SecurityMode.CERTIFICATE:
                # Load certificates
                await client.set_security_string(
                    "Basic256Sha256,SignAndEncrypt,certificate.pem,private_key.pem"
                )
            
            # Connect
            await client.connect()
            
            # Create subscription
            sub = await client.create_subscription(500, self)
            
            # Subscribe to variables based on capabilities
            for capability in device.capabilities:
                node_id = device.metadata.get(f"{capability}_node_id")
                if node_id:
                    node = client.get_node(node_id)
                    handle = await sub.subscribe_data_change(node)
                    
                    if device.id not in self.subscriptions:
                        self.subscriptions[device.id] = {}
                    self.subscriptions[device.id][capability] = handle
            
            self.clients[device.id] = client
            self.connected_devices[device.id] = device
            iot_devices_connected.labels(device.type.value, 'opcua').inc()
            
            logger.info(f"Connected OPC UA device: {device.id}")
            return True
            
        except Exception as e:
            logger.error(f"OPC UA connection failed: {e}")
            iot_errors.labels('connection', device.type.value).inc()
            return False
    
    async def disconnect(self, device_id: str) -> bool:
        """Disconnect OPC UA client"""
        try:
            if device_id in self.clients:
                client = self.clients.pop(device_id)
                await client.disconnect()
                
                device = self.connected_devices.pop(device_id)
                iot_devices_connected.labels(device.type.value, 'opcua').dec()
                
                logger.info(f"Disconnected OPC UA device: {device_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"OPC UA disconnect failed: {e}")
            return False
    
    async def send_message(self, device_id: str, message: Any) -> bool:
        """Write OPC UA variable"""
        try:
            if device_id not in self.clients:
                return False
            
            client = self.clients[device_id]
            
            # Extract node ID and value
            node_id = message.get('node_id')
            value = message.get('value')
            
            if node_id and value is not None:
                node = client.get_node(node_id)
                await node.write_value(value)
                
                iot_messages_sent.labels(device_id, 'write').inc()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"OPC UA write failed: {e}")
            iot_errors.labels('send', 'opcua').inc()
            return False
    
    async def receive_message(self) -> Optional[IoTMessage]:
        """Get message from queue"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def datachange_notification(self, node, val, data):
        """Handle OPC UA data change notification"""
        try:
            # Find device ID from node
            device_id = None
            capability = None
            
            for dev_id, subs in self.subscriptions.items():
                for cap, handle in subs.items():
                    if handle == data.monitored_item.handle:
                        device_id = dev_id
                        capability = cap
                        break
                if device_id:
                    break
            
            if device_id:
                # Create IoT message
                message = IoTMessage(
                    device_id=device_id,
                    timestamp=datetime.utcnow(),
                    message_type='datachange',
                    payload={
                        'capability': capability,
                        'value': val,
                        'source_timestamp': data.monitored_item.Value.SourceTimestamp,
                        'server_timestamp': data.monitored_item.Value.ServerTimestamp
                    }
                )
                
                await self.message_queue.put(message)
                
                iot_messages_received.labels(device_id, 'datachange').inc()
                
        except Exception as e:
            logger.error(f"OPC UA notification error: {e}")
            iot_errors.labels('receive', 'opcua').inc()


class BLEHandler(ProtocolHandler):
    """Bluetooth Low Energy handler"""
    
    def __init__(self):
        self.connected_devices: Dict[str, BleakClient] = {}
        self.device_info: Dict[str, IoTDevice] = {}
        self.message_queue = asyncio.Queue()
        self.scanner = None
    
    async def connect(self, device: IoTDevice) -> bool:
        """Connect to BLE device"""
        try:
            # Extract MAC address from connection string
            mac_address = device.connection_string
            
            # Create BLE client
            client = BleakClient(mac_address)
            await client.connect()
            
            # Discover services
            services = await client.get_services()
            logger.info(f"BLE services for {device.id}: {services}")
            
            # Subscribe to notifications based on capabilities
            for capability in device.capabilities:
                char_uuid = device.metadata.get(f"{capability}_uuid")
                if char_uuid:
                    await client.start_notify(
                        char_uuid,
                        lambda sender, data: asyncio.create_task(
                            self._handle_notification(device.id, capability, sender, data)
                        )
                    )
            
            self.connected_devices[device.id] = client
            self.device_info[device.id] = device
            iot_devices_connected.labels(device.type.value, 'ble').inc()
            
            logger.info(f"Connected BLE device: {device.id}")
            return True
            
        except Exception as e:
            logger.error(f"BLE connection failed: {e}")
            iot_errors.labels('connection', device.type.value).inc()
            return False
    
    async def disconnect(self, device_id: str) -> bool:
        """Disconnect BLE device"""
        try:
            if device_id in self.connected_devices:
                client = self.connected_devices.pop(device_id)
                await client.disconnect()
                
                device = self.device_info.pop(device_id)
                iot_devices_connected.labels(device.type.value, 'ble').dec()
                
                logger.info(f"Disconnected BLE device: {device_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"BLE disconnect failed: {e}")
            return False
    
    async def send_message(self, device_id: str, message: Any) -> bool:
        """Write BLE characteristic"""
        try:
            if device_id not in self.connected_devices:
                return False
            
            client = self.connected_devices[device_id]
            
            # Extract characteristic UUID and value
            char_uuid = message.get('characteristic')
            value = message.get('value')
            
            if char_uuid and value is not None:
                # Convert value to bytes if needed
                if isinstance(value, str):
                    data = value.encode()
                elif isinstance(value, int):
                    data = value.to_bytes(4, byteorder='little')
                elif isinstance(value, float):
                    data = struct.pack('<f', value)
                elif isinstance(value, bytes):
                    data = value
                else:
                    data = json.dumps(value).encode()
                
                await client.write_gatt_char(char_uuid, data)
                
                iot_messages_sent.labels(device_id, 'write').inc()
                iot_data_bytes.labels('sent', 'ble').inc(len(data))
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"BLE write failed: {e}")
            iot_errors.labels('send', 'ble').inc()
            return False
    
    async def receive_message(self) -> Optional[IoTMessage]:
        """Get message from queue"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def _handle_notification(
        self,
        device_id: str,
        capability: str,
        sender: int,
        data: bytes
    ):
        """Handle BLE notification"""
        try:
            # Parse data based on device configuration
            device = self.device_info.get(device_id)
            if not device:
                return
            
            # Decode data
            data_type = device.metadata.get(f"{capability}_type", "bytes")
            
            if data_type == "int":
                value = int.from_bytes(data, byteorder='little')
            elif data_type == "float":
                value = struct.unpack('<f', data)[0]
            elif data_type == "string":
                value = data.decode()
            else:
                value = data.hex()
            
            # Create IoT message
            message = IoTMessage(
                device_id=device_id,
                timestamp=datetime.utcnow(),
                message_type='notification',
                payload={
                    'capability': capability,
                    'characteristic': sender,
                    'value': value
                }
            )
            
            await self.message_queue.put(message)
            
            iot_messages_received.labels(device_id, 'notification').inc()
            iot_data_bytes.labels('received', 'ble').inc(len(data))
            
        except Exception as e:
            logger.error(f"BLE notification error: {e}")
            iot_errors.labels('receive', 'ble').inc()
    
    async def scan_devices(self, duration: float = 10.0) -> List[Dict[str, Any]]:
        """Scan for BLE devices"""
        discovered = []
        
        async def detection_callback(device, advertisement_data):
            discovered.append({
                'address': device.address,
                'name': device.name,
                'rssi': advertisement_data.rssi,
                'manufacturer_data': advertisement_data.manufacturer_data,
                'service_uuids': advertisement_data.service_uuids
            })
        
        scanner = BleakScanner(detection_callback)
        await scanner.start()
        await asyncio.sleep(duration)
        await scanner.stop()
        
        return discovered


class EdgeProcessor:
    """Edge computing processor for IoT data"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.tpu_available = self._check_tpu()
        self.gpu_available = self._check_gpu()
    
    def _check_tpu(self) -> bool:
        """Check if Edge TPU is available"""
        try:
            devices = edgetpu.list_edge_tpus()
            return len(devices) > 0
        except:
            return False
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        return tf.config.list_physical_devices('GPU') != []
    
    async def process_image(
        self,
        image_data: bytes,
        task: str = "classification"
    ) -> Dict[str, Any]:
        """Process image data at edge"""
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if task == "classification":
            return await self._classify_image(image)
        elif task == "detection":
            return await self._detect_objects(image)
        elif task == "segmentation":
            return await self._segment_image(image)
        else:
            return {"error": f"Unknown task: {task}"}
    
    async def _classify_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify image"""
        
        # Use Edge TPU if available
        if self.tpu_available and 'classification_tpu' not in self.models:
            # Load Edge TPU model
            self.models['classification_tpu'] = classify.make_classifier(
                'models/mobilenet_v2_1.0_224_quant_edgetpu.tflite'
            )
        
        if self.tpu_available:
            classifier = self.models['classification_tpu']
            # Resize and preprocess
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (224, 224))
            
            # Run inference
            classifier.classify(image_resized)
            
            results = []
            for label in classifier.labels:
                results.append({
                    'label': label.name,
                    'score': float(label.score)
                })
            
            return {
                'task': 'classification',
                'results': sorted(results, key=lambda x: x['score'], reverse=True)[:5],
                'processor': 'edge_tpu'
            }
        
        else:
            # Use TensorFlow
            if 'classification_tf' not in self.models:
                self.models['classification_tf'] = tf.keras.applications.MobileNetV2(
                    weights='imagenet',
                    input_shape=(224, 224, 3)
                )
            
            model = self.models['classification_tf']
            
            # Preprocess
            image_resized = cv2.resize(image, (224, 224))
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.expand_dims(image_resized, axis=0)
            )
            
            # Predict
            predictions = model.predict(image_array)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
                predictions,
                top=5
            )[0]
            
            results = []
            for _, label, score in decoded:
                results.append({
                    'label': label,
                    'score': float(score)
                })
            
            return {
                'task': 'classification',
                'results': results,
                'processor': 'gpu' if self.gpu_available else 'cpu'
            }
    
    async def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in image"""
        
        # Simplified object detection
        # In production, would use YOLO, SSD, or similar
        
        return {
            'task': 'detection',
            'objects': [
                {
                    'class': 'person',
                    'confidence': 0.95,
                    'bbox': [100, 100, 200, 300]
                }
            ],
            'processor': 'edge'
        }
    
    async def _segment_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Segment image"""
        
        # Simplified segmentation
        # In production, would use DeepLab, U-Net, or similar
        
        return {
            'task': 'segmentation',
            'segments': {
                'background': 0.7,
                'foreground': 0.3
            },
            'processor': 'edge'
        }
    
    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Process audio data at edge"""
        
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Extract features
        features = await self._extract_audio_features(audio, sample_rate)
        
        # Classify audio
        classification = await self._classify_audio(features)
        
        return {
            'task': 'audio_analysis',
            'features': features,
            'classification': classification
        }
    
    async def _extract_audio_features(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """Extract audio features"""
        
        # Calculate basic features
        features = {
            'duration': len(audio) / sample_rate,
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'zero_crossing_rate': float(np.sum(np.diff(np.sign(audio)) != 0) / len(audio))
        }
        
        # Spectral features
        freqs, times, Sxx = signal.spectrogram(audio, sample_rate)
        features['spectral_centroid'] = float(
            np.sum(freqs[:, np.newaxis] * Sxx) / np.sum(Sxx)
        )
        
        return features
    
    async def _classify_audio(self, features: Dict[str, float]) -> str:
        """Classify audio based on features"""
        
        # Simple classification based on features
        if features['rms'] > 0.1:
            if features['spectral_centroid'] > 1000:
                return 'speech'
            else:
                return 'noise'
        else:
            return 'silence'
    
    async def anomaly_detection(
        self,
        data: List[float],
        window_size: int = 100
    ) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        if 'anomaly_detector' not in self.models:
            # Train isolation forest
            self.models['anomaly_detector'] = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            )
        
        # Prepare data
        data_array = np.array(data).reshape(-1, 1)
        
        # Detect anomalies
        anomalies = self.models['anomaly_detector'].fit_predict(data_array)
        anomaly_indices = np.where(anomalies == -1)[0].tolist()
        
        return {
            'task': 'anomaly_detection',
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomaly_score': float(np.mean(anomalies == -1))
        }


class IoTDataProcessor:
    """Process and analyze IoT data"""
    
    def __init__(self):
        self.buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.aggregators = {}
        self.rules = []
        self.ml_models = {}
    
    async def process_message(self, message: IoTMessage) -> Dict[str, Any]:
        """Process incoming IoT message"""
        
        # Buffer data
        self.buffer[message.device_id].append(message)
        
        # Apply processing rules
        results = {
            'device_id': message.device_id,
            'timestamp': message.timestamp,
            'processed': []
        }
        
        # Run aggregations
        if message.device_id in self.aggregators:
            agg_result = await self._run_aggregation(
                message.device_id,
                self.aggregators[message.device_id]
            )
            results['aggregations'] = agg_result
        
        # Check rules
        triggered_rules = await self._check_rules(message)
        if triggered_rules:
            results['triggered_rules'] = triggered_rules
        
        # Run ML predictions
        if message.device_id in self.ml_models:
            predictions = await self._run_ml_prediction(message)
            results['predictions'] = predictions
        
        return results
    
    async def _run_aggregation(
        self,
        device_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run data aggregation"""
        
        window = config.get('window', 60)  # seconds
        
        # Get recent messages
        recent = [
            msg for msg in self.buffer[device_id]
            if (datetime.utcnow() - msg.timestamp).total_seconds() < window
        ]
        
        if not recent:
            return {}
        
        # Extract values
        values = []
        for msg in recent:
            if isinstance(msg.payload, dict):
                value = msg.payload.get(config.get('field', 'value'))
            else:
                value = msg.payload
            
            if value is not None and isinstance(value, (int, float)):
                values.append(value)
        
        if not values:
            return {}
        
        # Calculate aggregations
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'sum': np.sum(values)
        }
    
    async def _check_rules(self, message: IoTMessage) -> List[Dict[str, Any]]:
        """Check processing rules"""
        
        triggered = []
        
        for rule in self.rules:
            if await self._evaluate_rule(rule, message):
                triggered.append({
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'action': rule['action']
                })
                
                # Execute action
                await self._execute_rule_action(rule, message)
        
        return triggered
    
    async def _evaluate_rule(
        self,
        rule: Dict[str, Any],
        message: IoTMessage
    ) -> bool:
        """Evaluate rule condition"""
        
        # Check device filter
        if 'device_filter' in rule:
            if not self._match_filter(message.device_id, rule['device_filter']):
                return False
        
        # Check message type
        if 'message_type' in rule:
            if message.message_type != rule['message_type']:
                return False
        
        # Check condition
        condition = rule.get('condition', {})
        field = condition.get('field')
        operator = condition.get('operator')
        value = condition.get('value')
        
        if field and operator and value is not None:
            # Extract field value
            if isinstance(message.payload, dict):
                field_value = message.payload.get(field)
            else:
                field_value = message.payload
            
            # Evaluate condition
            if operator == 'eq':
                return field_value == value
            elif operator == 'gt':
                return field_value > value
            elif operator == 'lt':
                return field_value < value
            elif operator == 'gte':
                return field_value >= value
            elif operator == 'lte':
                return field_value <= value
            elif operator == 'ne':
                return field_value != value
            elif operator == 'contains':
                return value in str(field_value)
        
        return True
    
    def _match_filter(self, device_id: str, filter_pattern: str) -> bool:
        """Match device ID against filter pattern"""
        
        # Simple wildcard matching
        if '*' in filter_pattern:
            import fnmatch
            return fnmatch.fnmatch(device_id, filter_pattern)
        
        return device_id == filter_pattern
    
    async def _execute_rule_action(
        self,
        rule: Dict[str, Any],
        message: IoTMessage
    ):
        """Execute rule action"""
        
        action = rule.get('action', {})
        action_type = action.get('type')
        
        if action_type == 'alert':
            # Send alert
            logger.warning(
                f"Rule alert: {rule['name']} triggered for device {message.device_id}"
            )
        
        elif action_type == 'command':
            # Send command to device
            command = IoTCommand(
                device_id=message.device_id,
                command=action.get('command'),
                parameters=action.get('parameters', {})
            )
            # Queue command for execution
        
        elif action_type == 'webhook':
            # Call webhook
            webhook_url = action.get('url')
            if webhook_url:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        webhook_url,
                        json=message.to_dict()
                    )
    
    async def _run_ml_prediction(self, message: IoTMessage) -> Dict[str, Any]:
        """Run ML prediction on data"""
        
        # Simplified prediction
        # In production, would use trained models
        
        return {
            'anomaly_score': 0.1,
            'predicted_value': 25.5,
            'confidence': 0.85
        }
    
    def add_aggregation(
        self,
        device_id: str,
        field: str,
        window: int = 60
    ):
        """Add aggregation rule"""
        
        self.aggregators[device_id] = {
            'field': field,
            'window': window
        }
    
    def add_rule(
        self,
        rule_id: str,
        name: str,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        device_filter: Optional[str] = None
    ):
        """Add processing rule"""
        
        self.rules.append({
            'id': rule_id,
            'name': name,
            'condition': condition,
            'action': action,
            'device_filter': device_filter
        })


class IoTOrchestrator:
    """Orchestrate IoT device operations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        self.handlers: Dict[IoTProtocol, ProtocolHandler] = {}
        self.devices: Dict[str, IoTDevice] = {}
        self.data_processor = IoTDataProcessor()
        self.edge_processor = EdgeProcessor()
        self.message_queue = asyncio.Queue()
        self.command_queue = asyncio.Queue()
        self.is_running = False
        
        # Initialize protocol handlers
        self._init_handlers()
    
    def _init_handlers(self):
        """Initialize protocol handlers"""
        
        # MQTT
        mqtt_broker = os.getenv('MQTT_BROKER', 'localhost:1883')
        self.handlers[IoTProtocol.MQTT] = MQTTHandler(mqtt_broker)
        
        # CoAP
        self.handlers[IoTProtocol.COAP] = CoAPHandler()
        
        # OPC UA
        self.handlers[IoTProtocol.OPCUA] = OPCUAHandler()
        
        # BLE
        self.handlers[IoTProtocol.BLUETOOTH_LE] = BLEHandler()
    
    async def register_device(self, device: IoTDevice) -> bool:
        """Register new IoT device"""
        
        # Save to database
        session = self.Session()
        try:
            db_device = IoTDeviceRegistry(
                id=device.id,
                name=device.name,
                type=device.type.value,
                protocol=device.protocol.value,
                connection_string=device.connection_string,
                security_mode=device.security.value,
                location_lat=device.location.get('lat') if device.location else None,
                location_lon=device.location.get('lon') if device.location else None,
                location_alt=device.location.get('alt') if device.location else None,
                capabilities=device.capabilities,
                metadata=device.metadata,
                firmware_version=device.firmware_version,
                hardware_version=device.hardware_version
            )
            session.add(db_device)
            session.commit()
            
            # Connect to device
            handler = self.handlers.get(device.protocol)
            if handler:
                success = await handler.connect(device)
                if success:
                    self.devices[device.id] = device
                    device.is_online = True
                    logger.info(f"Registered device: {device.id}")
                    return True
                else:
                    logger.error(f"Failed to connect device: {device.id}")
                    return False
            else:
                logger.error(f"No handler for protocol: {device.protocol}")
                return False
                
        except Exception as e:
            session.rollback()
            logger.error(f"Device registration failed: {e}")
            return False
        finally:
            session.close()
    
    async def unregister_device(self, device_id: str) -> bool:
        """Unregister IoT device"""
        
        if device_id in self.devices:
            device = self.devices[device_id]
            
            # Disconnect
            handler = self.handlers.get(device.protocol)
            if handler:
                await handler.disconnect(device_id)
            
            # Remove from memory
            del self.devices[device_id]
            
            # Update database
            session = self.Session()
            try:
                db_device = session.query(IoTDeviceRegistry).filter_by(
                    id=device_id
                ).first()
                if db_device:
                    db_device.is_online = False
                    session.commit()
                
                logger.info(f"Unregistered device: {device_id}")
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"Device unregistration failed: {e}")
                return False
            finally:
                session.close()
        
        return False
    
    async def send_command(self, command: IoTCommand) -> bool:
        """Send command to device"""
        
        if command.device_id not in self.devices:
            logger.error(f"Device not found: {command.device_id}")
            return False
        
        device = self.devices[command.device_id]
        handler = self.handlers.get(device.protocol)
        
        if not handler:
            logger.error(f"No handler for protocol: {device.protocol}")
            return False
        
        # Format command based on protocol
        if device.protocol == IoTProtocol.MQTT:
            message = {
                'command': command.command,
                'parameters': command.parameters,
                'timestamp': datetime.utcnow().isoformat()
            }
        elif device.protocol == IoTProtocol.OPCUA:
            # For OPC UA, command should specify node ID and value
            message = command.parameters
        elif device.protocol == IoTProtocol.BLUETOOTH_LE:
            # For BLE, command should specify characteristic and value
            message = command.parameters
        else:
            message = command.parameters
        
        # Send command
        success = await handler.send_message(command.device_id, message)
        
        if success:
            logger.info(f"Sent command to device {command.device_id}: {command.command}")
        else:
            logger.error(f"Failed to send command to device {command.device_id}")
        
        return success
    
    async def start(self):
        """Start IoT orchestrator"""
        
        self.is_running = True
        
        # Start message processing
        asyncio.create_task(self._message_processor())
        asyncio.create_task(self._command_processor())
        asyncio.create_task(self._device_monitor())
        
        # Load existing devices from database
        await self._load_devices()
        
        logger.info("IoT orchestrator started")
    
    async def stop(self):
        """Stop IoT orchestrator"""
        
        self.is_running = False
        
        # Disconnect all devices
        for device_id in list(self.devices.keys()):
            await self.unregister_device(device_id)
        
        logger.info("IoT orchestrator stopped")
    
    async def _load_devices(self):
        """Load devices from database"""
        
        session = self.Session()
        try:
            db_devices = session.query(IoTDeviceRegistry).all()
            
            for db_device in db_devices:
                device = IoTDevice(
                    id=db_device.id,
                    name=db_device.name,
                    type=DeviceType(db_device.type),
                    protocol=IoTProtocol(db_device.protocol),
                    connection_string=db_device.connection_string,
                    security=SecurityMode(db_device.security_mode),
                    location={
                        'lat': db_device.location_lat,
                        'lon': db_device.location_lon,
                        'alt': db_device.location_alt
                    } if db_device.location_lat else None,
                    capabilities=db_device.capabilities or [],
                    metadata=db_device.metadata or {},
                    firmware_version=db_device.firmware_version,
                    hardware_version=db_device.hardware_version
                )
                
                # Try to connect
                handler = self.handlers.get(device.protocol)
                if handler:
                    try:
                        success = await handler.connect(device)
                        if success:
                            self.devices[device.id] = device
                            device.is_online = True
                    except Exception as e:
                        logger.error(f"Failed to connect device {device.id}: {e}")
                
        finally:
            session.close()
    
    async def _message_processor(self):
        """Process incoming messages"""
        
        while self.is_running:
            try:
                # Collect messages from all handlers
                for protocol, handler in self.handlers.items():
                    message = await handler.receive_message()
                    
                    if message:
                        # Update device last seen
                        if message.device_id in self.devices:
                            device = self.devices[message.device_id]
                            device.last_seen = message.timestamp
                            device.message_count += 1
                        
                        # Process message
                        with iot_latency.labels(device.type.value).time():
                            result = await self.data_processor.process_message(message)
                        
                        # Store data point
                        await self._store_data_point(message)
                        
                        # Put in queue for consumers
                        await self.message_queue.put((message, result))
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
    
    async def _command_processor(self):
        """Process outgoing commands"""
        
        while self.is_running:
            try:
                command = await asyncio.wait_for(
                    self.command_queue.get(),
                    timeout=1.0
                )
                
                # Execute command
                success = await self.send_command(command)
                
                # Call callback if provided
                if command.callback:
                    await command.callback(success)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Command processing error: {e}")
    
    async def _device_monitor(self):
        """Monitor device health"""
        
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for device_id, device in self.devices.items():
                    # Check if device is responsive
                    if device.last_seen:
                        time_since_last = (
                            current_time - device.last_seen
                        ).total_seconds()
                        
                        # Mark offline if no message for 5 minutes
                        if time_since_last > 300 and device.is_online:
                            device.is_online = False
                            logger.warning(f"Device {device_id} marked offline")
                            
                            # Update database
                            session = self.Session()
                            try:
                                db_device = session.query(IoTDeviceRegistry).filter_by(
                                    id=device_id
                                ).first()
                                if db_device:
                                    db_device.is_online = False
                                    session.commit()
                            finally:
                                session.close()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Device monitor error: {e}")
    
    async def _store_data_point(self, message: IoTMessage):
        """Store data point in database"""
        
        session = self.Session()
        try:
            # Extract value and unit
            value = message.payload
            unit = None
            data_type = message.message_type
            
            if isinstance(message.payload, dict):
                value = message.payload.get('value', message.payload)
                unit = message.payload.get('unit')
                data_type = message.payload.get('type', message.message_type)
            
            # Create data point
            data_point = IoTDataPoint(
                device_id=message.device_id,
                timestamp=message.timestamp,
                data_type=data_type,
                value=value if isinstance(value, (dict, list)) else {'value': value},
                unit=unit
            )
            
            session.add(data_point)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store data point: {e}")
        finally:
            session.close()
    
    async def get_device_data(
        self,
        device_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        data_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get historical device data"""
        
        session = self.Session()
        try:
            query = session.query(IoTDataPoint).filter(
                IoTDataPoint.device_id == device_id
            )
            
            if start_time:
                query = query.filter(IoTDataPoint.timestamp >= start_time)
            
            if end_time:
                query = query.filter(IoTDataPoint.timestamp <= end_time)
            
            if data_type:
                query = query.filter(IoTDataPoint.data_type == data_type)
            
            query = query.order_by(IoTDataPoint.timestamp.desc()).limit(limit)
            
            data_points = query.all()
            
            return [
                {
                    'timestamp': dp.timestamp.isoformat(),
                    'type': dp.data_type,
                    'value': dp.value,
                    'unit': dp.unit,
                    'quality': dp.quality
                }
                for dp in data_points
            ]
            
        finally:
            session.close()
    
    async def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device status"""
        
        if device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        
        return {
            'id': device.id,
            'name': device.name,
            'type': device.type.value,
            'protocol': device.protocol.value,
            'is_online': device.is_online,
            'last_seen': device.last_seen.isoformat() if device.last_seen else None,
            'location': device.location,
            'battery_level': device.battery_level,
            'signal_strength': device.signal_strength,
            'message_count': device.message_count,
            'error_count': device.error_count,
            'capabilities': device.capabilities,
            'firmware_version': device.firmware_version
        }
    
    async def bulk_command(
        self,
        device_filter: str,
        command: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Send command to multiple devices"""
        
        results = {}
        
        for device_id, device in self.devices.items():
            # Check if device matches filter
            if self._match_device_filter(device, device_filter):
                cmd = IoTCommand(
                    device_id=device_id,
                    command=command,
                    parameters=parameters
                )
                
                success = await self.send_command(cmd)
                results[device_id] = success
        
        return results
    
    def _match_device_filter(self, device: IoTDevice, filter_str: str) -> bool:
        """Match device against filter"""
        
        # Simple filter syntax: type:sensor protocol:mqtt location:indoor
        filters = {}
        for part in filter_str.split():
            if ':' in part:
                key, value = part.split(':', 1)
                filters[key] = value
        
        # Check filters
        if 'type' in filters and device.type.value != filters['type']:
            return False
        
        if 'protocol' in filters and device.protocol.value != filters['protocol']:
            return False
        
        if 'location' in filters:
            if not device.metadata.get('location_type') == filters['location']:
                return False
        
        if 'capability' in filters:
            if filters['capability'] not in device.capabilities:
                return False
        
        return True
    
    def create_device_network(self) -> nx.Graph:
        """Create network graph of devices"""
        
        G = nx.Graph()
        
        # Add nodes
        for device_id, device in self.devices.items():
            G.add_node(
                device_id,
                name=device.name,
                type=device.type.value,
                protocol=device.protocol.value,
                is_online=device.is_online
            )
        
        # Add edges based on relationships
        # For example, devices in same location or gateway relationships
        for d1_id, d1 in self.devices.items():
            for d2_id, d2 in self.devices.items():
                if d1_id != d2_id:
                    # Same location
                    if d1.location and d2.location:
                        dist = self._calculate_distance(d1.location, d2.location)
                        if dist < 10:  # Within 10 meters
                            G.add_edge(d1_id, d2_id, weight=dist)
                    
                    # Gateway relationship
                    if d1.type == DeviceType.GATEWAY:
                        if d2.metadata.get('gateway_id') == d1_id:
                            G.add_edge(d1_id, d2_id, relationship='gateway')
        
        return G
    
    def _calculate_distance(
        self,
        loc1: Dict[str, float],
        loc2: Dict[str, float]
    ) -> float:
        """Calculate distance between two locations"""
        
        # Simplified distance calculation
        lat1, lon1 = loc1['lat'], loc1['lon']
        lat2, lon2 = loc2['lat'], loc2['lon']
        
        # Haversine formula
        R = 6371000  # Earth radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (
            np.sin(delta_phi / 2) ** 2 +
            np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c


# Example usage
async def iot_demo():
    """Demo IoT integration"""
    
    # Initialize orchestrator
    orchestrator = IoTOrchestrator('postgresql://user:pass@localhost/iot_db')
    await orchestrator.start()
    
    # Register devices
    
    # Temperature sensor (MQTT)
    temp_sensor = IoTDevice(
        id="temp_sensor_01",
        name="Office Temperature Sensor",
        type=DeviceType.SENSOR,
        protocol=IoTProtocol.MQTT,
        connection_string="mqtt://localhost:1883",
        capabilities=["temperature", "humidity"],
        metadata={
            "location_type": "indoor",
            "room": "office",
            "model": "DHT22"
        },
        location={'lat': 37.7749, 'lon': -122.4194, 'alt': 10}
    )
    
    await orchestrator.register_device(temp_sensor)
    
    # Industrial sensor (OPC UA)
    industrial_sensor = IoTDevice(
        id="plc_01",
        name="Production Line PLC",
        type=DeviceType.INDUSTRIAL,
        protocol=IoTProtocol.OPCUA,
        connection_string="opc.tcp://192.168.1.100:4840",
        capabilities=["temperature", "pressure", "flow_rate"],
        metadata={
            "temperature_node_id": "ns=2;i=1001",
            "pressure_node_id": "ns=2;i=1002",
            "flow_rate_node_id": "ns=2;i=1003"
        }
    )
    
    await orchestrator.register_device(industrial_sensor)
    
    # Wearable device (BLE)
    wearable = IoTDevice(
        id="fitness_band_01",
        name="Fitness Band",
        type=DeviceType.WEARABLE,
        protocol=IoTProtocol.BLUETOOTH_LE,
        connection_string="AA:BB:CC:DD:EE:FF",
        capabilities=["heart_rate", "steps", "battery"],
        metadata={
            "heart_rate_uuid": "00002a37-0000-1000-8000-00805f9b34fb",
            "heart_rate_type": "int",
            "battery_uuid": "00002a19-0000-1000-8000-00805f9b34fb",
            "battery_type": "int"
        }
    )
    
    await orchestrator.register_device(wearable)
    
    # Add processing rules
    
    # Temperature alert rule
    orchestrator.data_processor.add_rule(
        rule_id="temp_alert",
        name="High Temperature Alert",
        condition={
            'field': 'temperature',
            'operator': 'gt',
            'value': 30
        },
        action={
            'type': 'alert',
            'severity': 'warning'
        },
        device_filter="temp_sensor_*"
    )
    
    # Aggregation for temperature sensor
    orchestrator.data_processor.add_aggregation(
        device_id="temp_sensor_01",
        field="temperature",
        window=300  # 5 minutes
    )
    
    # Send commands
    
    # Send command to temperature sensor
    command = IoTCommand(
        device_id="temp_sensor_01",
        command="set_interval",
        parameters={'interval': 60}  # Report every 60 seconds
    )
    
    await orchestrator.send_command(command)
    
    # Bulk command to all sensors
    results = await orchestrator.bulk_command(
        device_filter="type:sensor",
        command="calibrate",
        parameters={'mode': 'auto'}
    )
    
    print(f"Bulk command results: {results}")
    
    # Process messages
    print("\nProcessing IoT messages...")
    
    for _ in range(10):
        try:
            message, result = await asyncio.wait_for(
                orchestrator.message_queue.get(),
                timeout=5.0
            )
            
            print(f"\nReceived from {message.device_id}:")
            print(f"  Type: {message.message_type}")
            print(f"  Payload: {message.payload}")
            print(f"  Processing result: {result}")
            
            # Process with edge computing if image
            if message.message_type == "image":
                edge_result = await orchestrator.edge_processor.process_image(
                    message.payload,
                    task="classification"
                )
                print(f"  Edge processing: {edge_result}")
            
        except asyncio.TimeoutError:
            print(".", end="", flush=True)
    
    # Get device data
    historical_data = await orchestrator.get_device_data(
        device_id="temp_sensor_01",
        start_time=datetime.utcnow() - timedelta(hours=1),
        limit=100
    )
    
    print(f"\nHistorical data points: {len(historical_data)}")
    
    # Get device status
    for device_id in orchestrator.devices:
        status = await orchestrator.get_device_status(device_id)
        print(f"\nDevice {device_id} status:")
        print(f"  Online: {status['is_online']}")
        print(f"  Messages: {status['message_count']}")
        print(f"  Last seen: {status['last_seen']}")
    
    # Create device network
    network = orchestrator.create_device_network()
    print(f"\nDevice network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # Stop orchestrator
    await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(iot_demo())