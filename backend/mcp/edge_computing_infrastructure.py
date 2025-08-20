"""
Edge Computing Infrastructure - 40by6
Deploy MCP Stack capabilities to edge nodes for distributed processing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
import struct
import socket
import ssl
import os
import sys
import subprocess
import shutil
import tarfile
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import docker
from docker.types import Mount
import kubernetes as k8s
from kubernetes.client import V1Pod, V1Container, V1PodSpec
import ray
from ray import serve
import dask
from dask.distributed import Client as DaskClient, as_completed
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pyspark
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
import tensorflow as tf
import torch
import onnx
import onnxruntime as ort
from transformers import pipeline as hf_pipeline
import cv2
import grpc
import msgpack
import cloudpickle
import pyarrow as pa
import pyarrow.flight as flight
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, JSON, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import gpustat
import nvml
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import websockets
import zmq
import zmq.asyncio
import nanomsg
import nats
from nats.aio.client import Client as NATS
import pulsar
import kafka
from confluent_kafka import Producer, Consumer
import etcd3
import consul
import zookeeper
from kazoo.client import KazooClient
import hazelcast
import ignite
from pyignite import Client as IgniteClient
import aerospike
from cassandra.cluster import Cluster as CassandraCluster
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import prometheus_client
from prometheus_client.parser import text_string_to_metric_families
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from typing_extensions import Protocol
import requests
import yaml
import toml
import configparser
from cryptography.fernet import Fernet
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import jwt
import hashlib
import hmac
import secrets
import asyncio_mqtt
import paho.mqtt.client as mqtt
import aiocoap
import aiocoap.resource as resource
from bleak import BleakClient, BleakScanner
import serial
import can
import modbus_tk
import struct
import binascii

logger = logging.getLogger(__name__)

# Metrics
edge_nodes_active = Gauge('edge_nodes_active', 'Number of active edge nodes', ['node_type', 'location'])
edge_tasks_running = Gauge('edge_tasks_running', 'Number of running edge tasks', ['node_id', 'task_type'])
edge_tasks_completed = Counter('edge_tasks_completed_total', 'Total completed edge tasks', ['node_id', 'task_type', 'status'])
edge_data_processed = Counter('edge_data_processed_bytes', 'Bytes processed at edge', ['node_id', 'data_type'])
edge_latency = Histogram('edge_processing_latency_seconds', 'Edge processing latency', ['node_id', 'task_type'])
edge_resource_usage = Gauge('edge_resource_usage_percent', 'Edge resource usage', ['node_id', 'resource_type'])
edge_errors = Counter('edge_errors_total', 'Total edge errors', ['node_id', 'error_type'])

Base = declarative_base()


class EdgeNodeType(Enum):
    """Types of edge nodes"""
    GATEWAY = "gateway"
    COMPUTE = "compute"
    STORAGE = "storage"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    HYBRID = "hybrid"
    FOG = "fog"
    CLOUDLET = "cloudlet"
    MOBILE = "mobile"
    VEHICLE = "vehicle"
    DRONE = "drone"
    SATELLITE = "satellite"


class EdgeCapability(Enum):
    """Edge node capabilities"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    FPGA = "fpga"
    NEURAL = "neural"
    STORAGE = "storage"
    NETWORK = "network"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    DISPLAY = "display"
    AUDIO = "audio"
    VIDEO = "video"
    RADIO = "radio"
    QUANTUM = "quantum"


class DeploymentMode(Enum):
    """Edge deployment modes"""
    CONTAINER = "container"
    FUNCTION = "function"
    WASM = "wasm"
    NATIVE = "native"
    UNIKERNEL = "unikernel"
    VM = "vm"
    BARE_METAL = "bare_metal"


class SchedulingStrategy(Enum):
    """Task scheduling strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY = "priority"
    AFFINITY = "affinity"
    RANDOM = "random"
    WEIGHTED = "weighted"
    ML_OPTIMIZED = "ml_optimized"
    ENERGY_AWARE = "energy_aware"
    LATENCY_AWARE = "latency_aware"
    COST_AWARE = "cost_aware"


@dataclass
class EdgeNode:
    """Edge node representation"""
    id: str
    name: str
    type: EdgeNodeType
    location: Dict[str, Any]  # lat, lon, alt, region, zone
    capabilities: List[EdgeCapability]
    resources: Dict[str, Any]  # cpu, memory, storage, gpu, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "inactive"
    last_heartbeat: Optional[datetime] = None
    tasks_running: int = 0
    tasks_completed: int = 0
    errors: int = 0
    network_info: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'location': self.location,
            'capabilities': [c.value for c in self.capabilities],
            'resources': self.resources,
            'status': self.status,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'tasks_running': self.tasks_running,
            'tasks_completed': self.tasks_completed,
            'errors': self.errors,
            'metadata': self.metadata
        }


@dataclass
class EdgeTask:
    """Edge computing task"""
    id: str
    name: str
    task_type: str
    input_data: Any
    output_format: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    deadline: Optional[datetime] = None
    retry_count: int = 3
    timeout: float = 300.0
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'task_type': self.task_type,
            'requirements': self.requirements,
            'constraints': self.constraints,
            'priority': self.priority,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'retry_count': self.retry_count,
            'timeout': self.timeout,
            'metadata': self.metadata
        }


@dataclass
class EdgeResult:
    """Edge task result"""
    task_id: str
    node_id: str
    status: str  # success, failed, timeout
    output_data: Any
    processing_time: float
    resource_usage: Dict[str, float]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'node_id': self.node_id,
            'status': self.status,
            'processing_time': self.processing_time,
            'resource_usage': self.resource_usage,
            'error': self.error,
            'metadata': self.metadata
        }


class EdgeNodeRegistry(Base):
    """Database model for edge nodes"""
    __tablename__ = 'edge_nodes'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    location_lat = Column(Float)
    location_lon = Column(Float)
    location_region = Column(String(50))
    location_zone = Column(String(50))
    capabilities = Column(JSON)
    resources = Column(JSON)
    metadata = Column(JSON)
    status = Column(String(20), default='inactive')
    last_heartbeat = Column(DateTime)
    tasks_running = Column(Integer, default=0)
    tasks_completed = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_edge_node_type', 'type'),
        Index('idx_edge_node_status', 'status'),
        Index('idx_edge_node_location', 'location_region', 'location_zone'),
    )


class EdgeTaskHistory(Base):
    """Database model for task history"""
    __tablename__ = 'edge_task_history'
    
    id = Column(String(50), primary_key=True)
    task_name = Column(String(100))
    task_type = Column(String(50))
    node_id = Column(String(50), index=True)
    status = Column(String(20))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    input_size = Column(Integer)
    output_size = Column(Integer)
    resource_usage = Column(JSON)
    error = Column(Text)
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_edge_task_node_time', 'node_id', 'started_at'),
        Index('idx_edge_task_type_status', 'task_type', 'status'),
    )


class ResourceMonitor:
    """Monitor edge node resources"""
    
    def __init__(self):
        self.gpu_available = self._check_gpu()
        if self.gpu_available:
            nvmlInit()
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount() > 0
        except:
            return False
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resources"""
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory
        memory = psutil.virtual_memory()
        
        # Disk
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        resources = {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': cpu_freq.current if cpu_freq else 0,
                'load_avg': os.getloadavg()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        }
        
        # GPU if available
        if self.gpu_available:
            gpu_info = []
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    memory_info = nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpu_info.append({
                        'index': i,
                        'memory_total': memory_info.total,
                        'memory_used': memory_info.used,
                        'memory_free': memory_info.free,
                        'gpu_util': utilization.gpu,
                        'memory_util': utilization.memory
                    })
                
                resources['gpu'] = gpu_info
            except Exception as e:
                logger.error(f"Failed to get GPU info: {e}")
        
        return resources
    
    def check_resource_availability(
        self,
        requirements: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if resources meet requirements"""
        
        current = self.get_system_resources()
        available = True
        details = {}
        
        # Check CPU
        if 'cpu' in requirements:
            cpu_req = requirements['cpu']
            if isinstance(cpu_req, dict):
                if 'count' in cpu_req and current['cpu']['count'] < cpu_req['count']:
                    available = False
                    details['cpu'] = f"Requires {cpu_req['count']} cores, have {current['cpu']['count']}"
                
                if 'percent' in cpu_req and (100 - current['cpu']['percent']) < cpu_req['percent']:
                    available = False
                    details['cpu'] = f"Requires {cpu_req['percent']}% CPU, have {100 - current['cpu']['percent']}% available"
        
        # Check memory
        if 'memory' in requirements:
            mem_req = requirements['memory']
            if isinstance(mem_req, (int, float)):
                if current['memory']['available'] < mem_req:
                    available = False
                    details['memory'] = f"Requires {mem_req} bytes, have {current['memory']['available']} available"
        
        # Check GPU
        if 'gpu' in requirements and requirements['gpu']:
            if not self.gpu_available or not current.get('gpu'):
                available = False
                details['gpu'] = "GPU required but not available"
        
        return available, details


class ModelRegistry:
    """Manage ML models for edge deployment"""
    
    def __init__(self, storage_path: str = "/models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.model_metadata = {}
    
    async def register_model(
        self,
        model_id: str,
        model_data: bytes,
        model_type: str,  # tensorflow, pytorch, onnx, etc.
        metadata: Dict[str, Any]
    ) -> bool:
        """Register a model"""
        
        try:
            # Save model file
            model_path = self.storage_path / f"{model_id}.{model_type}"
            
            with open(model_path, 'wb') as f:
                f.write(model_data)
            
            # Save metadata
            meta_path = self.storage_path / f"{model_id}.meta.json"
            metadata['model_type'] = model_type
            metadata['size'] = len(model_data)
            metadata['registered_at'] = datetime.utcnow().isoformat()
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            
            self.model_metadata[model_id] = metadata
            
            logger.info(f"Registered model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    async def load_model(self, model_id: str) -> Optional[Any]:
        """Load a model"""
        
        if model_id in self.models:
            return self.models[model_id]
        
        try:
            # Load metadata
            meta_path = self.storage_path / f"{model_id}.meta.json"
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            model_type = metadata['model_type']
            model_path = self.storage_path / f"{model_id}.{model_type}"
            
            # Load based on type
            if model_type == 'tensorflow':
                model = tf.keras.models.load_model(str(model_path))
            
            elif model_type == 'pytorch':
                model = torch.load(str(model_path))
                model.eval()
            
            elif model_type == 'onnx':
                model = ort.InferenceSession(str(model_path))
            
            elif model_type == 'pickle':
                with open(model_path, 'rb') as f:
                    model = cloudpickle.load(f)
            
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            self.models[model_id] = model
            self.model_metadata[model_id] = metadata
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    async def optimize_model_for_edge(
        self,
        model_id: str,
        target_device: str = "cpu"  # cpu, gpu, tpu, edge_tpu
    ) -> Optional[str]:
        """Optimize model for edge deployment"""
        
        model = await self.load_model(model_id)
        if not model:
            return None
        
        metadata = self.model_metadata[model_id]
        model_type = metadata['model_type']
        
        optimized_id = f"{model_id}_opt_{target_device}"
        
        try:
            if model_type == 'tensorflow':
                # TensorFlow Lite conversion
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                
                if target_device == 'edge_tpu':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = lambda: self._representative_dataset()
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                    ]
                    converter.inference_input_type = tf.uint8
                    converter.inference_output_type = tf.uint8
                else:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                tflite_model = converter.convert()
                
                # Save optimized model
                opt_path = self.storage_path / f"{optimized_id}.tflite"
                with open(opt_path, 'wb') as f:
                    f.write(tflite_model)
                
                # Register optimized model
                await self.register_model(
                    optimized_id,
                    tflite_model,
                    'tflite',
                    {
                        'original_model': model_id,
                        'target_device': target_device,
                        'optimization': 'quantization'
                    }
                )
                
                return optimized_id
            
            elif model_type == 'pytorch':
                # PyTorch optimization
                if target_device == 'cpu':
                    # Quantization
                    model_int8 = torch.quantization.quantize_dynamic(
                        model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    
                    # Save
                    opt_path = self.storage_path / f"{optimized_id}.pt"
                    torch.save(model_int8, str(opt_path))
                    
                    with open(opt_path, 'rb') as f:
                        model_data = f.read()
                    
                    await self.register_model(
                        optimized_id,
                        model_data,
                        'pytorch',
                        {
                            'original_model': model_id,
                            'target_device': target_device,
                            'optimization': 'quantization'
                        }
                    )
                    
                    return optimized_id
                
                elif target_device == 'gpu':
                    # TorchScript
                    scripted = torch.jit.script(model)
                    
                    opt_path = self.storage_path / f"{optimized_id}.pt"
                    scripted.save(str(opt_path))
                    
                    with open(opt_path, 'rb') as f:
                        model_data = f.read()
                    
                    await self.register_model(
                        optimized_id,
                        model_data,
                        'torchscript',
                        {
                            'original_model': model_id,
                            'target_device': target_device,
                            'optimization': 'jit'
                        }
                    )
                    
                    return optimized_id
            
            elif model_type == 'onnx':
                # ONNX optimization
                import onnx
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                opt_path = self.storage_path / f"{optimized_id}.onnx"
                
                quantize_dynamic(
                    str(model_path),
                    str(opt_path),
                    weight_type=QuantType.QInt8
                )
                
                with open(opt_path, 'rb') as f:
                    model_data = f.read()
                
                await self.register_model(
                    optimized_id,
                    model_data,
                    'onnx',
                    {
                        'original_model': model_id,
                        'target_device': target_device,
                        'optimization': 'quantization'
                    }
                )
                
                return optimized_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return None
    
    def _representative_dataset(self):
        """Representative dataset for quantization"""
        # Generate dummy data - in production use real data
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            yield [data]


class TaskExecutor:
    """Execute tasks on edge nodes"""
    
    def __init__(
        self,
        node_id: str,
        model_registry: ModelRegistry,
        resource_monitor: ResourceMonitor
    ):
        self.node_id = node_id
        self.model_registry = model_registry
        self.resource_monitor = resource_monitor
        self.running_tasks = {}
        self.executor = ProcessPoolExecutor(max_workers=4)
    
    async def execute_task(self, task: EdgeTask) -> EdgeResult:
        """Execute a task"""
        
        start_time = datetime.utcnow()
        
        try:
            # Check resources
            available, details = self.resource_monitor.check_resource_availability(
                task.requirements
            )
            
            if not available:
                return EdgeResult(
                    task_id=task.id,
                    node_id=self.node_id,
                    status='failed',
                    output_data=None,
                    processing_time=0,
                    resource_usage={},
                    error=f"Insufficient resources: {details}"
                )
            
            # Update metrics
            edge_tasks_running.labels(self.node_id, task.task_type).inc()
            self.running_tasks[task.id] = task
            
            # Execute based on task type
            if task.task_type == 'inference':
                result = await self._execute_inference(task)
            
            elif task.task_type == 'training':
                result = await self._execute_training(task)
            
            elif task.task_type == 'data_processing':
                result = await self._execute_data_processing(task)
            
            elif task.task_type == 'stream_processing':
                result = await self._execute_stream_processing(task)
            
            elif task.task_type == 'function':
                result = await self._execute_function(task)
            
            else:
                result = EdgeResult(
                    task_id=task.id,
                    node_id=self.node_id,
                    status='failed',
                    output_data=None,
                    processing_time=0,
                    resource_usage={},
                    error=f"Unknown task type: {task.task_type}"
                )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.processing_time = processing_time
            
            # Get resource usage
            result.resource_usage = self.resource_monitor.get_system_resources()
            
            # Update metrics
            edge_tasks_running.labels(self.node_id, task.task_type).dec()
            edge_tasks_completed.labels(
                self.node_id,
                task.task_type,
                result.status
            ).inc()
            edge_latency.labels(self.node_id, task.task_type).observe(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            edge_errors.labels(self.node_id, 'execution').inc()
            
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='failed',
                output_data=None,
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                resource_usage={},
                error=str(e)
            )
        
        finally:
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _execute_inference(self, task: EdgeTask) -> EdgeResult:
        """Execute ML inference"""
        
        model_id = task.requirements.get('model_id')
        if not model_id:
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='failed',
                output_data=None,
                processing_time=0,
                resource_usage={},
                error="No model_id specified"
            )
        
        # Load model
        model = await self.model_registry.load_model(model_id)
        if not model:
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='failed',
                output_data=None,
                processing_time=0,
                resource_usage={},
                error=f"Failed to load model: {model_id}"
            )
        
        # Get model metadata
        metadata = self.model_registry.model_metadata[model_id]
        model_type = metadata['model_type']
        
        try:
            # Prepare input
            input_data = task.input_data
            
            # Run inference based on model type
            if model_type in ['tensorflow', 'tflite']:
                if model_type == 'tflite':
                    # TFLite inference
                    interpreter = tf.lite.Interpreter(model_content=model)
                    interpreter.allocate_tensors()
                    
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    
                    output = interpreter.get_tensor(output_details[0]['index'])
                else:
                    # TensorFlow inference
                    output = model.predict(input_data)
            
            elif model_type in ['pytorch', 'torchscript']:
                # PyTorch inference
                with torch.no_grad():
                    input_tensor = torch.tensor(input_data)
                    output = model(input_tensor).numpy()
            
            elif model_type == 'onnx':
                # ONNX inference
                input_name = model.get_inputs()[0].name
                output = model.run(None, {input_name: input_data})[0]
            
            else:
                # Generic Python model
                output = model.predict(input_data)
            
            # Process output
            if task.output_format == 'json':
                output_data = {
                    'predictions': output.tolist() if hasattr(output, 'tolist') else output,
                    'model_id': model_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                output_data = output
            
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='success',
                output_data=output_data,
                processing_time=0,
                resource_usage={}
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='failed',
                output_data=None,
                processing_time=0,
                resource_usage={},
                error=str(e)
            )
    
    async def _execute_training(self, task: EdgeTask) -> EdgeResult:
        """Execute federated/edge training"""
        
        # Simplified edge training
        # In production, implement federated learning protocols
        
        try:
            training_data = task.input_data
            model_config = task.requirements.get('model_config', {})
            
            # Create simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train
            history = model.fit(
                training_data['X'],
                training_data['y'],
                epochs=model_config.get('epochs', 5),
                batch_size=model_config.get('batch_size', 32),
                validation_split=0.2,
                verbose=0
            )
            
            # Get model weights for aggregation
            weights = model.get_weights()
            
            output_data = {
                'weights': [w.tolist() for w in weights],
                'metrics': {
                    'loss': history.history['loss'][-1],
                    'accuracy': history.history['accuracy'][-1]
                },
                'samples_trained': len(training_data['X'])
            }
            
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='success',
                output_data=output_data,
                processing_time=0,
                resource_usage={}
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='failed',
                output_data=None,
                processing_time=0,
                resource_usage={},
                error=str(e)
            )
    
    async def _execute_data_processing(self, task: EdgeTask) -> EdgeResult:
        """Execute data processing"""
        
        try:
            # Get processing function
            operation = task.requirements.get('operation', 'transform')
            
            if operation == 'aggregate':
                # Data aggregation
                data = pd.DataFrame(task.input_data)
                result = data.groupby(
                    task.requirements.get('group_by', [])
                ).agg(
                    task.requirements.get('aggregations', {})
                ).to_dict()
            
            elif operation == 'filter':
                # Data filtering
                data = pd.DataFrame(task.input_data)
                conditions = task.requirements.get('conditions', {})
                
                for col, condition in conditions.items():
                    if 'gt' in condition:
                        data = data[data[col] > condition['gt']]
                    if 'lt' in condition:
                        data = data[data[col] < condition['lt']]
                    if 'eq' in condition:
                        data = data[data[col] == condition['eq']]
                
                result = data.to_dict('records')
            
            elif operation == 'transform':
                # Data transformation
                transform_func = task.requirements.get('transform_func')
                if transform_func:
                    # Load and execute transform function
                    func = cloudpickle.loads(transform_func)
                    result = func(task.input_data)
                else:
                    result = task.input_data
            
            elif operation == 'compress':
                # Data compression
                compression = task.requirements.get('compression', 'gzip')
                
                if compression == 'gzip':
                    import gzip
                    compressed = gzip.compress(
                        json.dumps(task.input_data).encode()
                    )
                    result = {
                        'compressed': compressed.hex(),
                        'original_size': len(json.dumps(task.input_data)),
                        'compressed_size': len(compressed),
                        'compression_ratio': len(compressed) / len(json.dumps(task.input_data))
                    }
                
                elif compression == 'lz4':
                    compressed = lz4.frame.compress(
                        json.dumps(task.input_data).encode()
                    )
                    result = {
                        'compressed': compressed.hex(),
                        'original_size': len(json.dumps(task.input_data)),
                        'compressed_size': len(compressed),
                        'compression_ratio': len(compressed) / len(json.dumps(task.input_data))
                    }
            
            else:
                result = task.input_data
            
            edge_data_processed.labels(
                self.node_id,
                operation
            ).inc(len(str(result)))
            
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='success',
                output_data=result,
                processing_time=0,
                resource_usage={}
            )
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='failed',
                output_data=None,
                processing_time=0,
                resource_usage={},
                error=str(e)
            )
    
    async def _execute_stream_processing(self, task: EdgeTask) -> EdgeResult:
        """Execute stream processing"""
        
        try:
            # Simplified stream processing
            window_size = task.requirements.get('window_size', 10)
            operation = task.requirements.get('operation', 'mean')
            
            # Process stream data
            stream_data = task.input_data
            results = []
            
            for i in range(0, len(stream_data), window_size):
                window = stream_data[i:i+window_size]
                
                if operation == 'mean':
                    result = np.mean(window)
                elif operation == 'sum':
                    result = np.sum(window)
                elif operation == 'min':
                    result = np.min(window)
                elif operation == 'max':
                    result = np.max(window)
                elif operation == 'count':
                    result = len(window)
                else:
                    result = window
                
                results.append({
                    'window_start': i,
                    'window_end': min(i + window_size, len(stream_data)),
                    'result': result
                })
            
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='success',
                output_data=results,
                processing_time=0,
                resource_usage={}
            )
            
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='failed',
                output_data=None,
                processing_time=0,
                resource_usage={},
                error=str(e)
            )
    
    async def _execute_function(self, task: EdgeTask) -> EdgeResult:
        """Execute serverless function"""
        
        try:
            # Get function code
            function_code = task.requirements.get('function_code')
            if not function_code:
                raise ValueError("No function code provided")
            
            # Deserialize and execute function
            if isinstance(function_code, str):
                # Base64 encoded function
                import base64
                function_bytes = base64.b64decode(function_code)
            else:
                function_bytes = function_code
            
            func = cloudpickle.loads(function_bytes)
            
            # Execute function
            result = await func(task.input_data) if asyncio.iscoroutinefunction(func) else func(task.input_data)
            
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='success',
                output_data=result,
                processing_time=0,
                resource_usage={}
            )
            
        except Exception as e:
            logger.error(f"Function execution failed: {e}")
            return EdgeResult(
                task_id=task.id,
                node_id=self.node_id,
                status='failed',
                output_data=None,
                processing_time=0,
                resource_usage={},
                error=str(e)
            )


class EdgeScheduler:
    """Schedule tasks across edge nodes"""
    
    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.ML_OPTIMIZED
    ):
        self.strategy = strategy
        self.nodes: Dict[str, EdgeNode] = {}
        self.task_queue = asyncio.Queue()
        self.scheduler_model = None
        
        if strategy == SchedulingStrategy.ML_OPTIMIZED:
            self._init_ml_scheduler()
    
    def _init_ml_scheduler(self):
        """Initialize ML-based scheduler"""
        # Simple neural network for task scheduling
        self.scheduler_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.scheduler_model.compile(
            optimizer='adam',
            loss='binary_crossentropy'
        )
    
    async def schedule_task(self, task: EdgeTask) -> Optional[str]:
        """Schedule task to appropriate node"""
        
        available_nodes = [
            node for node in self.nodes.values()
            if node.status == 'active'
        ]
        
        if not available_nodes:
            logger.warning("No available edge nodes")
            return None
        
        # Select node based on strategy
        if self.strategy == SchedulingStrategy.ROUND_ROBIN:
            selected_node = self._round_robin_select(available_nodes)
        
        elif self.strategy == SchedulingStrategy.LEAST_LOADED:
            selected_node = self._least_loaded_select(available_nodes)
        
        elif self.strategy == SchedulingStrategy.PRIORITY:
            selected_node = self._priority_select(available_nodes, task)
        
        elif self.strategy == SchedulingStrategy.AFFINITY:
            selected_node = self._affinity_select(available_nodes, task)
        
        elif self.strategy == SchedulingStrategy.ML_OPTIMIZED:
            selected_node = await self._ml_optimized_select(available_nodes, task)
        
        elif self.strategy == SchedulingStrategy.ENERGY_AWARE:
            selected_node = self._energy_aware_select(available_nodes)
        
        elif self.strategy == SchedulingStrategy.LATENCY_AWARE:
            selected_node = self._latency_aware_select(available_nodes, task)
        
        elif self.strategy == SchedulingStrategy.COST_AWARE:
            selected_node = self._cost_aware_select(available_nodes, task)
        
        else:
            # Random selection
            import random
            selected_node = random.choice(available_nodes)
        
        if selected_node:
            logger.info(f"Scheduled task {task.id} to node {selected_node.id}")
            return selected_node.id
        
        return None
    
    def _round_robin_select(self, nodes: List[EdgeNode]) -> Optional[EdgeNode]:
        """Round-robin node selection"""
        if not hasattr(self, '_rr_index'):
            self._rr_index = 0
        
        if not nodes:
            return None
        
        selected = nodes[self._rr_index % len(nodes)]
        self._rr_index += 1
        
        return selected
    
    def _least_loaded_select(self, nodes: List[EdgeNode]) -> Optional[EdgeNode]:
        """Select least loaded node"""
        return min(nodes, key=lambda n: n.tasks_running)
    
    def _priority_select(
        self,
        nodes: List[EdgeNode],
        task: EdgeTask
    ) -> Optional[EdgeNode]:
        """Select node based on task priority"""
        
        # High priority tasks go to powerful nodes
        if task.priority >= 8:
            powerful_nodes = [
                n for n in nodes
                if EdgeCapability.GPU in n.capabilities or
                   EdgeCapability.TPU in n.capabilities
            ]
            if powerful_nodes:
                return self._least_loaded_select(powerful_nodes)
        
        return self._least_loaded_select(nodes)
    
    def _affinity_select(
        self,
        nodes: List[EdgeNode],
        task: EdgeTask
    ) -> Optional[EdgeNode]:
        """Select node based on affinity rules"""
        
        # Check for node affinity in task constraints
        preferred_location = task.constraints.get('location')
        preferred_capabilities = task.constraints.get('capabilities', [])
        
        # Filter by location
        if preferred_location:
            location_nodes = [
                n for n in nodes
                if n.location.get('region') == preferred_location or
                   n.location.get('zone') == preferred_location
            ]
            if location_nodes:
                nodes = location_nodes
        
        # Filter by capabilities
        if preferred_capabilities:
            capability_nodes = [
                n for n in nodes
                if all(cap in [c.value for c in n.capabilities]
                      for cap in preferred_capabilities)
            ]
            if capability_nodes:
                nodes = capability_nodes
        
        return self._least_loaded_select(nodes)
    
    async def _ml_optimized_select(
        self,
        nodes: List[EdgeNode],
        task: EdgeTask
    ) -> Optional[EdgeNode]:
        """ML-based node selection"""
        
        if not self.scheduler_model:
            return self._least_loaded_select(nodes)
        
        # Extract features for each node
        scores = []
        
        for node in nodes:
            features = [
                node.tasks_running / 10.0,  # Normalized load
                node.errors / 100.0,  # Normalized error rate
                1.0 if EdgeCapability.GPU in node.capabilities else 0.0,
                1.0 if EdgeCapability.TPU in node.capabilities else 0.0,
                node.resources.get('cpu', {}).get('percent', 50) / 100.0,
                node.resources.get('memory', {}).get('percent', 50) / 100.0,
                task.priority / 10.0,
                1.0 if task.requirements.get('gpu') else 0.0,
                len(task.requirements) / 10.0,
                task.timeout / 1000.0
            ]
            
            # Predict success probability
            score = self.scheduler_model.predict(
                np.array([features]),
                verbose=0
            )[0][0]
            
            scores.append((score, node))
        
        # Select node with highest score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return scores[0][1] if scores else None
    
    def _energy_aware_select(
        self,
        nodes: List[EdgeNode]
    ) -> Optional[EdgeNode]:
        """Select node based on energy efficiency"""
        
        # Estimate energy consumption
        energy_scores = []
        
        for node in nodes:
            # Simple energy model
            cpu_power = node.resources.get('cpu', {}).get('percent', 50) * 2.0  # Watts
            memory_power = node.resources.get('memory', {}).get('percent', 50) * 0.5
            gpu_power = 150 if EdgeCapability.GPU in node.capabilities else 0
            
            total_power = cpu_power + memory_power + gpu_power
            efficiency = (node.tasks_completed + 1) / (total_power + 1)
            
            energy_scores.append((efficiency, node))
        
        # Select most energy efficient
        energy_scores.sort(key=lambda x: x[0], reverse=True)
        
        return energy_scores[0][1] if energy_scores else None
    
    def _latency_aware_select(
        self,
        nodes: List[EdgeNode],
        task: EdgeTask
    ) -> Optional[EdgeNode]:
        """Select node based on expected latency"""
        
        # Estimate latency for each node
        latency_scores = []
        
        for node in nodes:
            # Network latency (simplified)
            network_latency = 10  # ms, would ping in reality
            
            # Processing latency estimate
            queue_latency = node.tasks_running * 100  # ms per queued task
            
            # Capability bonus
            capability_bonus = 0
            if task.requirements.get('gpu') and EdgeCapability.GPU in node.capabilities:
                capability_bonus = -500  # GPU acceleration
            
            total_latency = network_latency + queue_latency + capability_bonus
            
            latency_scores.append((total_latency, node))
        
        # Select lowest latency
        latency_scores.sort(key=lambda x: x[0])
        
        return latency_scores[0][1] if latency_scores else None
    
    def _cost_aware_select(
        self,
        nodes: List[EdgeNode],
        task: EdgeTask
    ) -> Optional[EdgeNode]:
        """Select node based on cost optimization"""
        
        # Cost model
        cost_scores = []
        
        for node in nodes:
            # Base cost
            base_cost = node.metadata.get('hourly_cost', 1.0)
            
            # Resource costs
            cpu_cost = node.resources.get('cpu', {}).get('percent', 50) / 100.0
            memory_cost = node.resources.get('memory', {}).get('percent', 50) / 100.0
            
            # Special resource costs
            gpu_cost = 5.0 if EdgeCapability.GPU in node.capabilities else 0
            
            # Total cost estimate
            total_cost = base_cost * (1 + cpu_cost + memory_cost) + gpu_cost
            
            # Performance factor
            performance = node.tasks_completed / (node.errors + 1)
            
            # Cost efficiency
            efficiency = performance / total_cost
            
            cost_scores.append((efficiency, node))
        
        # Select most cost efficient
        cost_scores.sort(key=lambda x: x[0], reverse=True)
        
        return cost_scores[0][1] if cost_scores else None
    
    def update_scheduler_model(
        self,
        task_id: str,
        node_id: str,
        success: bool
    ):
        """Update ML scheduler with task result"""
        
        if not self.scheduler_model:
            return
        
        # In production, would store and batch training data
        # This is simplified for demonstration


class EdgeOrchestrator:
    """Orchestrate edge computing infrastructure"""
    
    def __init__(
        self,
        database_url: str,
        coordinator_url: Optional[str] = None
    ):
        self.database_url = database_url
        self.coordinator_url = coordinator_url
        
        # Initialize components
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        self.nodes: Dict[str, EdgeNode] = {}
        self.executors: Dict[str, TaskExecutor] = {}
        self.scheduler = EdgeScheduler()
        self.model_registry = ModelRegistry()
        self.resource_monitor = ResourceMonitor()
        
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.is_running = False
        
        # Communication
        self.zmq_context = zmq.asyncio.Context()
        self.command_socket = None
        self.result_socket = None
    
    async def start(self):
        """Start edge orchestrator"""
        
        self.is_running = True
        
        # Start communication
        await self._start_communication()
        
        # Start background tasks
        asyncio.create_task(self._task_dispatcher())
        asyncio.create_task(self._result_collector())
        asyncio.create_task(self._node_monitor())
        asyncio.create_task(self._metrics_collector())
        
        # Load existing nodes
        await self._load_nodes()
        
        logger.info("Edge orchestrator started")
    
    async def stop(self):
        """Stop edge orchestrator"""
        
        self.is_running = False
        
        # Stop communication
        if self.command_socket:
            self.command_socket.close()
        if self.result_socket:
            self.result_socket.close()
        
        self.zmq_context.term()
        
        logger.info("Edge orchestrator stopped")
    
    async def _start_communication(self):
        """Start ZMQ communication"""
        
        # Command socket (PULL)
        self.command_socket = self.zmq_context.socket(zmq.PULL)
        self.command_socket.bind("tcp://*:5555")
        
        # Result socket (PUSH)
        self.result_socket = self.zmq_context.socket(zmq.PUSH)
        self.result_socket.bind("tcp://*:5556")
        
        # Start command receiver
        asyncio.create_task(self._command_receiver())
    
    async def _command_receiver(self):
        """Receive commands via ZMQ"""
        
        while self.is_running:
            try:
                message = await self.command_socket.recv()
                command = msgpack.unpackb(message, raw=False)
                
                cmd_type = command.get('type')
                
                if cmd_type == 'register_node':
                    await self.register_node(EdgeNode(**command['node']))
                
                elif cmd_type == 'unregister_node':
                    await self.unregister_node(command['node_id'])
                
                elif cmd_type == 'submit_task':
                    await self.submit_task(EdgeTask(**command['task']))
                
                elif cmd_type == 'register_model':
                    await self.model_registry.register_model(
                        command['model_id'],
                        command['model_data'],
                        command['model_type'],
                        command['metadata']
                    )
                
            except Exception as e:
                logger.error(f"Command receiver error: {e}")
    
    async def register_node(self, node: EdgeNode) -> bool:
        """Register edge node"""
        
        session = self.Session()
        try:
            # Save to database
            db_node = EdgeNodeRegistry(
                id=node.id,
                name=node.name,
                type=node.type.value,
                location_lat=node.location.get('lat'),
                location_lon=node.location.get('lon'),
                location_region=node.location.get('region'),
                location_zone=node.location.get('zone'),
                capabilities=[c.value for c in node.capabilities],
                resources=node.resources,
                metadata=node.metadata,
                status='active'
            )
            
            session.add(db_node)
            session.commit()
            
            # Add to memory
            self.nodes[node.id] = node
            node.status = 'active'
            node.last_heartbeat = datetime.utcnow()
            
            # Create executor
            self.executors[node.id] = TaskExecutor(
                node.id,
                self.model_registry,
                self.resource_monitor
            )
            
            # Update scheduler
            self.scheduler.nodes[node.id] = node
            
            # Update metrics
            edge_nodes_active.labels(
                node.type.value,
                node.location.get('region', 'unknown')
            ).inc()
            
            logger.info(f"Registered edge node: {node.id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to register node: {e}")
            return False
        finally:
            session.close()
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister edge node"""
        
        if node_id not in self.nodes:
            return False
        
        session = self.Session()
        try:
            # Update database
            db_node = session.query(EdgeNodeRegistry).filter_by(
                id=node_id
            ).first()
            
            if db_node:
                db_node.status = 'inactive'
                session.commit()
            
            # Remove from memory
            node = self.nodes.pop(node_id)
            
            if node_id in self.executors:
                del self.executors[node_id]
            
            if node_id in self.scheduler.nodes:
                del self.scheduler.nodes[node_id]
            
            # Update metrics
            edge_nodes_active.labels(
                node.type.value,
                node.location.get('region', 'unknown')
            ).dec()
            
            logger.info(f"Unregistered edge node: {node_id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to unregister node: {e}")
            return False
        finally:
            session.close()
    
    async def submit_task(self, task: EdgeTask) -> str:
        """Submit task for execution"""
        
        # Schedule task
        node_id = await self.scheduler.schedule_task(task)
        
        if not node_id:
            # Queue task if no node available
            await self.task_queue.put(task)
            logger.warning(f"Task {task.id} queued - no available nodes")
            return task.id
        
        # Execute on selected node
        asyncio.create_task(self._execute_on_node(task, node_id))
        
        return task.id
    
    async def _execute_on_node(self, task: EdgeTask, node_id: str):
        """Execute task on specific node"""
        
        if node_id not in self.executors:
            logger.error(f"No executor for node: {node_id}")
            return
        
        executor = self.executors[node_id]
        node = self.nodes[node_id]
        
        # Update node stats
        node.tasks_running += 1
        
        try:
            # Execute task
            result = await executor.execute_task(task)
            
            # Store result
            await self._store_task_result(task, result)
            
            # Send result via ZMQ
            if self.result_socket:
                await self.result_socket.send(
                    msgpack.packb(result.to_dict())
                )
            
            # Put in result queue
            await self.result_queue.put(result)
            
            # Update scheduler model if ML-based
            if self.scheduler.strategy == SchedulingStrategy.ML_OPTIMIZED:
                self.scheduler.update_scheduler_model(
                    task.id,
                    node_id,
                    result.status == 'success'
                )
            
            # Update node stats
            node.tasks_completed += 1
            if result.status == 'failed':
                node.errors += 1
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            node.errors += 1
            edge_errors.labels(node_id, 'task_execution').inc()
        
        finally:
            node.tasks_running -= 1
    
    async def _store_task_result(self, task: EdgeTask, result: EdgeResult):
        """Store task result in database"""
        
        session = self.Session()
        try:
            history = EdgeTaskHistory(
                id=task.id,
                task_name=task.name,
                task_type=task.task_type,
                node_id=result.node_id,
                status=result.status,
                started_at=datetime.utcnow() - timedelta(seconds=result.processing_time),
                completed_at=datetime.utcnow(),
                processing_time=result.processing_time,
                input_size=len(str(task.input_data)),
                output_size=len(str(result.output_data)) if result.output_data else 0,
                resource_usage=result.resource_usage,
                error=result.error,
                metadata=result.metadata
            )
            
            session.add(history)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store task result: {e}")
        finally:
            session.close()
    
    async def _task_dispatcher(self):
        """Dispatch queued tasks"""
        
        while self.is_running:
            try:
                # Get queued task
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Try to schedule
                node_id = await self.scheduler.schedule_task(task)
                
                if node_id:
                    # Execute on node
                    asyncio.create_task(self._execute_on_node(task, node_id))
                else:
                    # Re-queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(5)  # Back off
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Task dispatcher error: {e}")
    
    async def _result_collector(self):
        """Collect and process results"""
        
        while self.is_running:
            try:
                result = await asyncio.wait_for(
                    self.result_queue.get(),
                    timeout=1.0
                )
                
                # Process result callbacks
                # In production, would handle callbacks
                
                logger.info(
                    f"Task {result.task_id} completed with status: {result.status}"
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Result collector error: {e}")
    
    async def _node_monitor(self):
        """Monitor node health"""
        
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for node_id, node in list(self.nodes.items()):
                    # Check heartbeat
                    if node.last_heartbeat:
                        time_since = (current_time - node.last_heartbeat).total_seconds()
                        
                        if time_since > 60 and node.status == 'active':
                            # Mark as inactive
                            node.status = 'inactive'
                            logger.warning(f"Node {node_id} marked inactive")
                            
                            # Update database
                            session = self.Session()
                            try:
                                db_node = session.query(EdgeNodeRegistry).filter_by(
                                    id=node_id
                                ).first()
                                if db_node:
                                    db_node.status = 'inactive'
                                    session.commit()
                            finally:
                                session.close()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Node monitor error: {e}")
    
    async def _metrics_collector(self):
        """Collect and update metrics"""
        
        while self.is_running:
            try:
                for node_id, node in self.nodes.items():
                    if node.status == 'active':
                        # Update resource usage metrics
                        if 'cpu' in node.resources:
                            edge_resource_usage.labels(
                                node_id,
                                'cpu'
                            ).set(node.resources['cpu'].get('percent', 0))
                        
                        if 'memory' in node.resources:
                            edge_resource_usage.labels(
                                node_id,
                                'memory'
                            ).set(node.resources['memory'].get('percent', 0))
                        
                        # Update task metrics
                        edge_tasks_running.labels(
                            node_id,
                            'all'
                        ).set(node.tasks_running)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
    
    async def _load_nodes(self):
        """Load nodes from database"""
        
        session = self.Session()
        try:
            db_nodes = session.query(EdgeNodeRegistry).filter_by(
                status='active'
            ).all()
            
            for db_node in db_nodes:
                node = EdgeNode(
                    id=db_node.id,
                    name=db_node.name,
                    type=EdgeNodeType(db_node.type),
                    location={
                        'lat': db_node.location_lat,
                        'lon': db_node.location_lon,
                        'region': db_node.location_region,
                        'zone': db_node.location_zone
                    },
                    capabilities=[EdgeCapability(c) for c in db_node.capabilities],
                    resources=db_node.resources or {},
                    metadata=db_node.metadata or {},
                    status=db_node.status,
                    last_heartbeat=db_node.last_heartbeat,
                    tasks_running=db_node.tasks_running,
                    tasks_completed=db_node.tasks_completed,
                    errors=db_node.errors
                )
                
                self.nodes[node.id] = node
                self.scheduler.nodes[node.id] = node
                
                # Create executor
                self.executors[node.id] = TaskExecutor(
                    node.id,
                    self.model_registry,
                    self.resource_monitor
                )
                
        finally:
            session.close()
    
    def get_cluster_topology(self) -> nx.Graph:
        """Get edge cluster topology"""
        
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(
                node_id,
                name=node.name,
                type=node.type.value,
                status=node.status,
                location=node.location,
                capabilities=[c.value for c in node.capabilities]
            )
        
        # Add edges based on location proximity
        for n1_id, n1 in self.nodes.items():
            for n2_id, n2 in self.nodes.items():
                if n1_id != n2_id:
                    # Same region
                    if n1.location.get('region') == n2.location.get('region'):
                        G.add_edge(n1_id, n2_id, weight=1, type='region')
                    
                    # Same zone
                    if n1.location.get('zone') == n2.location.get('zone'):
                        G.add_edge(n1_id, n2_id, weight=0.5, type='zone')
        
        return G
    
    async def deploy_application(
        self,
        app_id: str,
        deployment_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy distributed application across edge nodes"""
        
        results = {
            'app_id': app_id,
            'deployments': []
        }
        
        # Parse deployment spec
        components = deployment_spec.get('components', [])
        
        for component in components:
            # Find suitable nodes
            requirements = component.get('requirements', {})
            constraints = component.get('constraints', {})
            replicas = component.get('replicas', 1)
            
            # Create tasks for each replica
            for i in range(replicas):
                task = EdgeTask(
                    id=f"{app_id}_{component['name']}_{i}",
                    name=f"{component['name']} replica {i}",
                    task_type='deployment',
                    input_data={
                        'component': component,
                        'app_id': app_id
                    },
                    requirements=requirements,
                    constraints=constraints,
                    priority=8
                )
                
                # Submit task
                task_id = await self.submit_task(task)
                
                results['deployments'].append({
                    'component': component['name'],
                    'replica': i,
                    'task_id': task_id
                })
        
        return results
    
    async def federated_learning(
        self,
        model_id: str,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate federated learning across edge nodes"""
        
        # Get nodes with data
        data_nodes = [
            node for node in self.nodes.values()
            if node.status == 'active' and
               node.metadata.get('has_training_data', False)
        ]
        
        if not data_nodes:
            return {'error': 'No nodes with training data'}
        
        # Create training tasks
        training_tasks = []
        
        for node in data_nodes:
            task = EdgeTask(
                id=f"federated_{model_id}_{node.id}",
                name=f"Federated training on {node.name}",
                task_type='training',
                input_data={
                    'model_id': model_id,
                    'data_source': 'local'
                },
                requirements={
                    'model_config': training_config
                },
                constraints={
                    'node_id': node.id  # Pin to specific node
                }
            )
            
            training_tasks.append(task)
        
        # Submit training tasks
        task_ids = []
        for task in training_tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        
        # Wait for results
        results = []
        timeout = training_config.get('timeout', 3600)
        start_time = datetime.utcnow()
        
        while len(results) < len(task_ids):
            if (datetime.utcnow() - start_time).total_seconds() > timeout:
                break
            
            try:
                result = await asyncio.wait_for(
                    self.result_queue.get(),
                    timeout=1.0
                )
                
                if result.task_id in task_ids:
                    results.append(result)
                
            except asyncio.TimeoutError:
                continue
        
        # Aggregate model weights
        if results:
            aggregated_weights = self._aggregate_federated_weights(results)
            
            # Update global model
            # In production, would update model in registry
            
            return {
                'model_id': model_id,
                'nodes_trained': len(results),
                'aggregated_weights': 'stored',
                'metrics': {
                    'average_loss': np.mean([
                        r.output_data.get('metrics', {}).get('loss', 0)
                        for r in results
                        if r.status == 'success'
                    ])
                }
            }
        
        return {'error': 'No training results received'}
    
    def _aggregate_federated_weights(
        self,
        results: List[EdgeResult]
    ) -> List[np.ndarray]:
        """Aggregate federated learning weights"""
        
        # Extract weights from successful results
        all_weights = []
        sample_counts = []
        
        for result in results:
            if result.status == 'success' and result.output_data:
                weights = result.output_data.get('weights', [])
                samples = result.output_data.get('samples_trained', 1)
                
                if weights:
                    all_weights.append(weights)
                    sample_counts.append(samples)
        
        if not all_weights:
            return []
        
        # Weighted average based on sample count
        total_samples = sum(sample_counts)
        aggregated = []
        
        # Assume all models have same architecture
        num_layers = len(all_weights[0])
        
        for layer_idx in range(num_layers):
            layer_weights = []
            
            for i, weights in enumerate(all_weights):
                weight = np.array(weights[layer_idx])
                weighted = weight * (sample_counts[i] / total_samples)
                layer_weights.append(weighted)
            
            aggregated_layer = np.sum(layer_weights, axis=0)
            aggregated.append(aggregated_layer)
        
        return aggregated


# Example edge node implementation
class EdgeNodeAgent:
    """Edge node agent that runs on edge devices"""
    
    def __init__(
        self,
        node_config: Dict[str, Any],
        orchestrator_url: str
    ):
        self.node_config = node_config
        self.orchestrator_url = orchestrator_url
        self.node = None
        self.zmq_context = zmq.asyncio.Context()
        self.command_socket = None
        self.result_socket = None
        self.is_running = False
        
        # Local components
        self.resource_monitor = ResourceMonitor()
        self.model_registry = ModelRegistry(
            storage_path=node_config.get('model_path', '/edge/models')
        )
        self.executor = None
    
    async def start(self):
        """Start edge node agent"""
        
        self.is_running = True
        
        # Create node
        self.node = EdgeNode(
            id=self.node_config['id'],
            name=self.node_config['name'],
            type=EdgeNodeType(self.node_config['type']),
            location=self.node_config['location'],
            capabilities=[EdgeCapability(c) for c in self.node_config['capabilities']],
            resources=self.resource_monitor.get_system_resources(),
            metadata=self.node_config.get('metadata', {})
        )
        
        # Create executor
        self.executor = TaskExecutor(
            self.node.id,
            self.model_registry,
            self.resource_monitor
        )
        
        # Connect to orchestrator
        await self._connect_to_orchestrator()
        
        # Register node
        await self._register_node()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_sender())
        asyncio.create_task(self._resource_updater())
        asyncio.create_task(self._task_receiver())
        
        logger.info(f"Edge node agent {self.node.id} started")
    
    async def stop(self):
        """Stop edge node agent"""
        
        self.is_running = False
        
        # Unregister node
        await self._unregister_node()
        
        # Close sockets
        if self.command_socket:
            self.command_socket.close()
        if self.result_socket:
            self.result_socket.close()
        
        self.zmq_context.term()
        
        logger.info(f"Edge node agent {self.node.id} stopped")
    
    async def _connect_to_orchestrator(self):
        """Connect to orchestrator"""
        
        # Command socket (PUSH)
        self.command_socket = self.zmq_context.socket(zmq.PUSH)
        self.command_socket.connect(f"{self.orchestrator_url}:5555")
        
        # Result socket (PULL)
        self.result_socket = self.zmq_context.socket(zmq.PULL)
        self.result_socket.connect(f"{self.orchestrator_url}:5556")
    
    async def _register_node(self):
        """Register with orchestrator"""
        
        command = {
            'type': 'register_node',
            'node': self.node.to_dict()
        }
        
        await self.command_socket.send(msgpack.packb(command))
        logger.info(f"Registered node {self.node.id} with orchestrator")
    
    async def _unregister_node(self):
        """Unregister from orchestrator"""
        
        command = {
            'type': 'unregister_node',
            'node_id': self.node.id
        }
        
        await self.command_socket.send(msgpack.packb(command))
    
    async def _heartbeat_sender(self):
        """Send heartbeats to orchestrator"""
        
        while self.is_running:
            try:
                # Update heartbeat
                self.node.last_heartbeat = datetime.utcnow()
                
                # Send heartbeat
                command = {
                    'type': 'heartbeat',
                    'node_id': self.node.id,
                    'timestamp': self.node.last_heartbeat.isoformat(),
                    'status': self.node.status,
                    'tasks_running': self.node.tasks_running,
                    'resources': self.node.resources
                }
                
                await self.command_socket.send(msgpack.packb(command))
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _resource_updater(self):
        """Update resource information"""
        
        while self.is_running:
            try:
                # Get current resources
                self.node.resources = self.resource_monitor.get_system_resources()
                
                # Update metrics
                if 'cpu' in self.node.resources:
                    edge_resource_usage.labels(
                        self.node.id,
                        'cpu'
                    ).set(self.node.resources['cpu']['percent'])
                
                if 'memory' in self.node.resources:
                    edge_resource_usage.labels(
                        self.node.id,
                        'memory'
                    ).set(self.node.resources['memory']['percent'])
                
                await asyncio.sleep(10)  # Every 10 seconds
                
            except Exception as e:
                logger.error(f"Resource update error: {e}")
    
    async def _task_receiver(self):
        """Receive and execute tasks"""
        
        while self.is_running:
            try:
                # Check for tasks assigned to this node
                # In production, would receive tasks via message queue
                
                # Simulate task execution
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Task receiver error: {e}")


# Example usage
async def edge_demo():
    """Demo edge computing infrastructure"""
    
    # Initialize orchestrator
    orchestrator = EdgeOrchestrator(
        'postgresql://user:pass@localhost/edge_db'
    )
    await orchestrator.start()
    
    # Register edge nodes
    
    # Gateway node
    gateway_node = EdgeNode(
        id="edge_gateway_01",
        name="IoT Gateway",
        type=EdgeNodeType.GATEWAY,
        location={
            'lat': 37.7749,
            'lon': -122.4194,
            'region': 'us-west',
            'zone': 'us-west-1a'
        },
        capabilities=[
            EdgeCapability.CPU,
            EdgeCapability.NETWORK,
            EdgeCapability.STORAGE
        ],
        resources={
            'cpu': {'count': 4, 'frequency': 2400},
            'memory': {'total': 8 * 1024**3},  # 8GB
            'storage': {'total': 128 * 1024**3}  # 128GB
        },
        metadata={
            'device_model': 'EdgeBox-G1',
            'os': 'Ubuntu 20.04'
        }
    )
    
    await orchestrator.register_node(gateway_node)
    
    # GPU compute node
    gpu_node = EdgeNode(
        id="edge_gpu_01",
        name="GPU Compute Node",
        type=EdgeNodeType.COMPUTE,
        location={
            'lat': 37.7749,
            'lon': -122.4194,
            'region': 'us-west',
            'zone': 'us-west-1a'
        },
        capabilities=[
            EdgeCapability.CPU,
            EdgeCapability.GPU,
            EdgeCapability.NEURAL
        ],
        resources={
            'cpu': {'count': 8, 'frequency': 3200},
            'memory': {'total': 32 * 1024**3},  # 32GB
            'gpu': [{'model': 'RTX 3080', 'memory': 10 * 1024**3}]
        },
        metadata={
            'device_model': 'EdgeCompute-X1',
            'cuda_version': '11.4'
        }
    )
    
    await orchestrator.register_node(gpu_node)
    
    # Mobile edge node
    mobile_node = EdgeNode(
        id="edge_mobile_01",
        name="Mobile Edge Unit",
        type=EdgeNodeType.MOBILE,
        location={
            'lat': 37.7850,
            'lon': -122.4100,
            'region': 'us-west',
            'zone': 'mobile'
        },
        capabilities=[
            EdgeCapability.CPU,
            EdgeCapability.SENSOR,
            EdgeCapability.VIDEO
        ],
        resources={
            'cpu': {'count': 8, 'frequency': 2800},
            'memory': {'total': 8 * 1024**3},
            'battery': {'level': 85, 'capacity': 20000}  # mAh
        },
        metadata={
            'vehicle_id': 'MCP-VAN-001',
            'connectivity': '5G'
        }
    )
    
    await orchestrator.register_node(mobile_node)
    
    # Register ML models
    
    # Image classification model
    dummy_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        dummy_model.save(tmp.name)
        with open(tmp.name, 'rb') as f:
            model_data = f.read()
    
    await orchestrator.model_registry.register_model(
        'mobilenet_v2',
        model_data,
        'tensorflow',
        {
            'task': 'image_classification',
            'input_shape': [224, 224, 3],
            'classes': 1000
        }
    )
    
    # Optimize for edge
    optimized_id = await orchestrator.model_registry.optimize_model_for_edge(
        'mobilenet_v2',
        'cpu'
    )
    
    print(f"Optimized model ID: {optimized_id}")
    
    # Submit tasks
    
    # Inference task
    inference_task = EdgeTask(
        id="task_001",
        name="Image Classification",
        task_type="inference",
        input_data=np.random.rand(1, 224, 224, 3).astype(np.float32),
        output_format="json",
        requirements={
            'model_id': 'mobilenet_v2',
            'cpu': {'percent': 20}
        },
        constraints={
            'capabilities': ['cpu']
        },
        priority=7
    )
    
    task_id = await orchestrator.submit_task(inference_task)
    print(f"Submitted inference task: {task_id}")
    
    # Data processing task
    data_task = EdgeTask(
        id="task_002",
        name="Sensor Data Aggregation",
        task_type="data_processing",
        input_data=[
            {'timestamp': i, 'temperature': 20 + i * 0.1, 'humidity': 50 + i * 0.2}
            for i in range(100)
        ],
        output_format="json",
        requirements={
            'operation': 'aggregate',
            'group_by': [],
            'aggregations': {
                'temperature': ['mean', 'std'],
                'humidity': ['mean', 'std']
            }
        },
        priority=5
    )
    
    task_id = await orchestrator.submit_task(data_task)
    print(f"Submitted data processing task: {task_id}")
    
    # Stream processing task
    stream_task = EdgeTask(
        id="task_003",
        name="Stream Analytics",
        task_type="stream_processing",
        input_data=list(range(100)),
        output_format="json",
        requirements={
            'window_size': 10,
            'operation': 'mean'
        },
        priority=6
    )
    
    task_id = await orchestrator.submit_task(stream_task)
    print(f"Submitted stream processing task: {task_id}")
    
    # Deploy distributed application
    app_spec = {
        'components': [
            {
                'name': 'frontend',
                'requirements': {'cpu': {'count': 2}, 'memory': 2 * 1024**3},
                'constraints': {'capabilities': ['network']},
                'replicas': 2
            },
            {
                'name': 'backend',
                'requirements': {'cpu': {'count': 4}, 'memory': 4 * 1024**3},
                'constraints': {'capabilities': ['cpu']},
                'replicas': 3
            },
            {
                'name': 'ml-service',
                'requirements': {'gpu': True},
                'constraints': {'capabilities': ['gpu']},
                'replicas': 1
            }
        ]
    }
    
    deployment = await orchestrator.deploy_application('my_app', app_spec)
    print(f"Deployed application: {deployment}")
    
    # Wait for results
    print("\nWaiting for task results...")
    
    for _ in range(10):
        try:
            result = await asyncio.wait_for(
                orchestrator.result_queue.get(),
                timeout=5.0
            )
            
            print(f"\nTask {result.task_id} completed:")
            print(f"  Status: {result.status}")
            print(f"  Node: {result.node_id}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            
            if result.status == 'success' and result.output_data:
                print(f"  Output: {str(result.output_data)[:100]}...")
            
            if result.error:
                print(f"  Error: {result.error}")
            
        except asyncio.TimeoutError:
            print(".", end="", flush=True)
    
    # Get cluster topology
    topology = orchestrator.get_cluster_topology()
    print(f"\nCluster topology: {topology.number_of_nodes()} nodes, {topology.number_of_edges()} edges")
    
    # Federated learning example
    print("\nStarting federated learning...")
    
    # Mark nodes as having training data
    for node in orchestrator.nodes.values():
        node.metadata['has_training_data'] = True
    
    fl_result = await orchestrator.federated_learning(
        'mobilenet_v2',
        {
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    )
    
    print(f"Federated learning result: {fl_result}")
    
    # Stop orchestrator
    await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(edge_demo())