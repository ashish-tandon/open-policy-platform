"""
AR/VR Visualization System - 40by6
Immersive 3D data visualization for MCP Stack with AR/VR support
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata
import quaternion
import trimesh
import pyvista as pv
import vtk
from vispy import app, scene, visuals
import moderngl
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
import mediapipe as mp
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import DPTForDepthEstimation, DPTImageProcessor
import websockets
import json
import struct
import zlib
import msgpack
from collections import deque
import threading
import queue
import time
import math
import random
import colorsys
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from sqlalchemy import create_engine
import redis
from prometheus_client import Counter, Histogram, Gauge
import psutil
import GPUtil

logger = logging.getLogger(__name__)

# Metrics
vr_sessions = Gauge('vr_active_sessions', 'Active VR/AR sessions')
render_fps = Histogram('vr_render_fps', 'Rendering frames per second')
gesture_recognitions = Counter('vr_gesture_recognitions', 'Gesture recognitions', ['gesture_type'])
object_interactions = Counter('vr_object_interactions', 'Object interactions', ['interaction_type'])
render_time = Histogram('vr_render_time_seconds', 'Time to render frame')


class VisualizationType(Enum):
    """Types of 3D visualizations"""
    SCATTER_3D = "scatter_3d"
    SURFACE_3D = "surface_3d"
    VOLUME_3D = "volume_3d"
    NETWORK_GRAPH = "network_graph"
    TIME_SERIES_3D = "time_series_3d"
    HEATMAP_3D = "heatmap_3d"
    FLOW_FIELD = "flow_field"
    PARTICLE_SYSTEM = "particle_system"
    MOLECULAR = "molecular"
    TERRAIN = "terrain"
    GALAXY = "galaxy"
    DATA_SCULPTURE = "data_sculpture"


class InteractionMode(Enum):
    """VR/AR interaction modes"""
    GAZE = "gaze"  # Look at objects
    GESTURE = "gesture"  # Hand gestures
    CONTROLLER = "controller"  # VR controllers
    VOICE = "voice"  # Voice commands
    HAPTIC = "haptic"  # Touch feedback
    BRAIN = "brain"  # Brain-computer interface
    

class RenderingEngine(Enum):
    """3D rendering engines"""
    OPENGL = "opengl"
    WEBGL = "webgl"
    VULKAN = "vulkan"
    METAL = "metal"  # macOS
    DIRECTX = "directx"  # Windows
    WEBGPU = "webgpu"


class DeviceType(Enum):
    """AR/VR device types"""
    OCULUS_QUEST = "oculus_quest"
    HOLOLENS = "hololens"
    MAGIC_LEAP = "magic_leap"
    VIVE = "vive"
    PSVR = "psvr"
    MOBILE_AR = "mobile_ar"
    DESKTOP_VR = "desktop_vr"
    WEB_XR = "web_xr"


@dataclass
class Transform3D:
    """3D transformation"""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))  # Quaternion
    scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix"""
        # Create rotation matrix from quaternion
        q = quaternion.from_float_array(self.rotation)
        rot_matrix = quaternion.as_rotation_matrix(q)
        
        # Create transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rot_matrix * self.scale
        matrix[:3, 3] = self.position
        
        return matrix


@dataclass
class Material3D:
    """3D material properties"""
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0]))  # RGBA
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    texture: Optional[str] = None
    shader: str = "standard"
    transparent: bool = False
    double_sided: bool = False


@dataclass
class Light3D:
    """3D light source"""
    type: str = "directional"  # directional, point, spot, area
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 10.0, 0.0]))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    intensity: float = 1.0
    range: float = 100.0
    inner_cone: float = 30.0  # For spot lights
    outer_cone: float = 45.0  # For spot lights
    cast_shadows: bool = True


@dataclass
class Object3D:
    """3D object in scene"""
    id: str
    name: str
    mesh: Optional[Any] = None  # Mesh data
    transform: Transform3D = field(default_factory=Transform3D)
    material: Material3D = field(default_factory=Material3D)
    data: Dict[str, Any] = field(default_factory=dict)
    interactive: bool = True
    visible: bool = True
    children: List['Object3D'] = field(default_factory=list)
    animations: List[Dict[str, Any]] = field(default_factory=list)
    physics: Optional[Dict[str, Any]] = None
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box"""
        if self.mesh is None:
            return np.zeros(3), np.zeros(3)
        
        # Calculate bounds from mesh vertices
        vertices = self.mesh.get('vertices', [])
        if not vertices:
            return np.zeros(3), np.zeros(3)
        
        vertices = np.array(vertices)
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        return min_bounds, max_bounds


@dataclass
class Camera3D:
    """3D camera"""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 5.0, 10.0]))
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    fov: float = 60.0  # Field of view in degrees
    near: float = 0.1
    far: float = 1000.0
    projection: str = "perspective"  # perspective or orthographic
    
    def get_view_matrix(self) -> np.ndarray:
        """Get view matrix"""
        # Calculate camera basis vectors
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Create view matrix
        view = np.eye(4)
        view[:3, 0] = right
        view[:3, 1] = up
        view[:3, 2] = -forward
        view[:3, 3] = -self.position
        
        return view


@dataclass
class Scene3D:
    """3D scene container"""
    id: str
    name: str
    objects: Dict[str, Object3D] = field(default_factory=dict)
    lights: List[Light3D] = field(default_factory=list)
    camera: Camera3D = field(default_factory=Camera3D)
    background: Optional[Union[np.ndarray, str]] = None  # Color or skybox
    fog: Optional[Dict[str, Any]] = None
    post_processing: List[str] = field(default_factory=list)
    physics_enabled: bool = False
    
    def add_object(self, obj: Object3D):
        """Add object to scene"""
        self.objects[obj.id] = obj
    
    def remove_object(self, obj_id: str):
        """Remove object from scene"""
        if obj_id in self.objects:
            del self.objects[obj_id]
    
    def find_object(self, name: str) -> Optional[Object3D]:
        """Find object by name"""
        for obj in self.objects.values():
            if obj.name == name:
                return obj
        return None


class GestureRecognizer:
    """Hand gesture recognition for AR/VR"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.gesture_history = deque(maxlen=10)
    
    async def recognize_gesture(self, image: np.ndarray) -> Optional[str]:
        """Recognize hand gesture from image"""
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Get hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract key points
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            landmarks = np.array(landmarks)
            
            # Recognize gesture based on landmarks
            gesture = self._classify_gesture(landmarks)
            
            if gesture:
                self.gesture_history.append(gesture)
                gesture_recognitions.labels(gesture).inc()
                
                # Check for gesture sequences
                sequence = self._check_gesture_sequence()
                if sequence:
                    return sequence
                
                return gesture
        
        return None
    
    def _classify_gesture(self, landmarks: np.ndarray) -> Optional[str]:
        """Classify gesture from landmarks"""
        
        # Simple gesture classification based on finger positions
        # Thumb
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        # Index finger
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        
        # Middle finger
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]
        
        # Ring finger
        ring_tip = landmarks[16]
        ring_mcp = landmarks[13]
        
        # Pinky
        pinky_tip = landmarks[20]
        pinky_mcp = landmarks[17]
        
        # Check if fingers are extended
        thumb_extended = thumb_tip[1] < thumb_mcp[1]
        index_extended = index_tip[1] < index_mcp[1]
        middle_extended = middle_tip[1] < middle_mcp[1]
        ring_extended = ring_tip[1] < ring_mcp[1]
        pinky_extended = pinky_tip[1] < pinky_mcp[1]
        
        # Classify gestures
        extended_count = sum([
            thumb_extended, index_extended, middle_extended,
            ring_extended, pinky_extended
        ])
        
        if extended_count == 0:
            return "fist"
        elif extended_count == 1 and index_extended:
            return "point"
        elif extended_count == 2 and index_extended and middle_extended:
            return "peace"
        elif extended_count == 5:
            return "open_palm"
        elif thumb_extended and index_extended and not middle_extended:
            return "pinch"
        elif index_extended and pinky_extended and not middle_extended:
            return "rock"
        else:
            return "unknown"
    
    def _check_gesture_sequence(self) -> Optional[str]:
        """Check for gesture sequences"""
        
        if len(self.gesture_history) < 3:
            return None
        
        # Check for swipe gestures
        recent = list(self.gesture_history)[-3:]
        
        if recent == ["open_palm", "fist", "open_palm"]:
            return "grab_release"
        elif recent == ["point", "point", "fist"]:
            return "select"
        elif recent == ["open_palm", "open_palm", "fist"]:
            return "zoom_in"
        elif recent == ["fist", "open_palm", "open_palm"]:
            return "zoom_out"
        
        return None


class DataMeshGenerator:
    """Generate 3D meshes from data"""
    
    def __init__(self):
        self.mesh_cache = {}
    
    async def generate_scatter_mesh(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        size_col: Optional[str] = None,
        color_col: Optional[str] = None
    ) -> List[Object3D]:
        """Generate 3D scatter plot mesh"""
        
        objects = []
        
        # Normalize data
        x_data = data[x_col].values
        y_data = data[y_col].values
        z_data = data[z_col].values
        
        x_norm = (x_data - x_data.min()) / (x_data.max() - x_data.min()) * 10 - 5
        y_norm = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 10 - 5
        z_norm = (z_data - z_data.min()) / (z_data.max() - z_data.min()) * 10 - 5
        
        # Generate spheres for each point
        for i in range(len(data)):
            # Create sphere mesh
            sphere = self._create_sphere(radius=0.1 if size_col is None else data[size_col].iloc[i] * 0.1)
            
            # Set position
            transform = Transform3D(
                position=np.array([x_norm[i], y_norm[i], z_norm[i]])
            )
            
            # Set color
            if color_col:
                color_value = data[color_col].iloc[i]
                color = self._value_to_color(color_value)
            else:
                color = np.array([0.5, 0.5, 1.0, 1.0])
            
            material = Material3D(color=color, metallic=0.3, roughness=0.7)
            
            # Create object
            obj = Object3D(
                id=f"point_{i}",
                name=f"Data Point {i}",
                mesh=sphere,
                transform=transform,
                material=material,
                data={
                    x_col: data[x_col].iloc[i],
                    y_col: data[y_col].iloc[i],
                    z_col: data[z_col].iloc[i]
                }
            )
            
            objects.append(obj)
        
        return objects
    
    async def generate_surface_mesh(
        self,
        data: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float]
    ) -> Object3D:
        """Generate 3D surface mesh"""
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], data.shape[0])
        y = np.linspace(y_range[0], y_range[1], data.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Create vertices
        vertices = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                vertices.append([X[i, j], data[i, j], Y[i, j]])
        
        vertices = np.array(vertices)
        
        # Create faces (triangles)
        faces = []
        for i in range(data.shape[0] - 1):
            for j in range(data.shape[1] - 1):
                # Two triangles per grid cell
                idx = i * data.shape[1] + j
                faces.append([idx, idx + 1, idx + data.shape[1]])
                faces.append([idx + 1, idx + data.shape[1] + 1, idx + data.shape[1]])
        
        faces = np.array(faces)
        
        # Create mesh
        mesh = {
            'vertices': vertices,
            'faces': faces,
            'normals': self._calculate_normals(vertices, faces)
        }
        
        # Create material with gradient coloring
        material = Material3D(
            color=np.array([0.3, 0.7, 1.0, 1.0]),
            metallic=0.2,
            roughness=0.6
        )
        
        return Object3D(
            id="surface",
            name="Data Surface",
            mesh=mesh,
            material=material,
            data={'type': 'surface', 'dimensions': data.shape}
        )
    
    async def generate_network_mesh(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Tuple[int, int]],
        layout: str = "force"
    ) -> List[Object3D]:
        """Generate 3D network graph mesh"""
        
        objects = []
        
        # Calculate node positions
        if layout == "force":
            positions = self._force_directed_layout(nodes, edges)
        elif layout == "circular":
            positions = self._circular_layout(len(nodes))
        else:
            positions = self._random_layout(len(nodes))
        
        # Create node objects
        for i, node in enumerate(nodes):
            sphere = self._create_sphere(radius=0.2)
            
            transform = Transform3D(position=positions[i])
            
            # Color by node type or property
            color = self._value_to_color(node.get('value', i))
            material = Material3D(color=color, emissive=color * 0.2)
            
            obj = Object3D(
                id=f"node_{i}",
                name=node.get('name', f"Node {i}"),
                mesh=sphere,
                transform=transform,
                material=material,
                data=node
            )
            
            objects.append(obj)
        
        # Create edge objects
        for i, (start, end) in enumerate(edges):
            cylinder = self._create_cylinder(
                positions[start],
                positions[end],
                radius=0.02
            )
            
            material = Material3D(
                color=np.array([0.7, 0.7, 0.7, 0.5]),
                transparent=True
            )
            
            obj = Object3D(
                id=f"edge_{i}",
                name=f"Edge {start}-{end}",
                mesh=cylinder,
                material=material,
                interactive=False,
                data={'start': start, 'end': end}
            )
            
            objects.append(obj)
        
        return objects
    
    async def generate_volume_mesh(
        self,
        volume_data: np.ndarray,
        threshold: float = 0.5
    ) -> Object3D:
        """Generate 3D volume mesh using marching cubes"""
        
        try:
            import skimage.measure
            
            # Apply marching cubes
            verts, faces, normals, values = skimage.measure.marching_cubes(
                volume_data,
                level=threshold,
                step_size=1
            )
            
            # Normalize positions
            verts = verts / np.max(verts) * 10 - 5
            
            mesh = {
                'vertices': verts,
                'faces': faces,
                'normals': normals
            }
            
            material = Material3D(
                color=np.array([0.8, 0.3, 0.3, 0.8]),
                transparent=True,
                double_sided=True
            )
            
            return Object3D(
                id="volume",
                name="Volume Data",
                mesh=mesh,
                material=material,
                data={'type': 'volume', 'threshold': threshold}
            )
            
        except ImportError:
            logger.warning("scikit-image not available for marching cubes")
            return Object3D(id="volume", name="Volume Data")
    
    async def generate_particle_system(
        self,
        num_particles: int,
        emitter_shape: str = "sphere",
        velocity_field: Optional[Callable] = None
    ) -> Object3D:
        """Generate particle system"""
        
        # Generate initial particle positions
        if emitter_shape == "sphere":
            positions = self._generate_sphere_points(num_particles, radius=1.0)
        elif emitter_shape == "box":
            positions = np.random.uniform(-1, 1, (num_particles, 3))
        else:
            positions = np.random.randn(num_particles, 3)
        
        # Generate velocities
        if velocity_field:
            velocities = np.array([velocity_field(pos) for pos in positions])
        else:
            velocities = np.random.randn(num_particles, 3) * 0.1
        
        # Create particle mesh (point cloud)
        mesh = {
            'vertices': positions,
            'point_size': 2.0,
            'velocities': velocities
        }
        
        # Gradient colors based on speed
        speeds = np.linalg.norm(velocities, axis=1)
        colors = np.array([self._value_to_color(s) for s in speeds])
        
        material = Material3D(
            color=np.array([1.0, 1.0, 1.0, 1.0]),
            shader="particle"
        )
        
        return Object3D(
            id="particles",
            name="Particle System",
            mesh=mesh,
            material=material,
            data={
                'type': 'particles',
                'count': num_particles,
                'colors': colors
            },
            animations=[{
                'type': 'particle_update',
                'velocity_field': velocity_field
            }]
        )
    
    def _create_sphere(self, radius: float = 1.0, subdivisions: int = 2) -> Dict[str, Any]:
        """Create sphere mesh"""
        
        # Check cache
        cache_key = f"sphere_{radius}_{subdivisions}"
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key].copy()
        
        # Create icosphere
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        
        mesh = {
            'vertices': sphere.vertices,
            'faces': sphere.faces,
            'normals': sphere.vertex_normals
        }
        
        self.mesh_cache[cache_key] = mesh
        return mesh.copy()
    
    def _create_cylinder(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float = 0.1
    ) -> Dict[str, Any]:
        """Create cylinder mesh between two points"""
        
        # Calculate cylinder transform
        direction = end - start
        height = np.linalg.norm(direction)
        
        if height < 0.001:
            return {'vertices': [], 'faces': []}
        
        # Create cylinder along Y axis
        cylinder = trimesh.creation.cylinder(radius=radius, height=height)
        
        # Rotate to align with direction
        y_axis = np.array([0, 1, 0])
        direction_norm = direction / height
        
        if np.allclose(direction_norm, y_axis):
            rotation_matrix = np.eye(3)
        else:
            axis = np.cross(y_axis, direction_norm)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(y_axis, direction_norm))
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
        
        # Apply rotation
        vertices = cylinder.vertices @ rotation_matrix.T
        
        # Translate to position
        vertices += (start + end) / 2
        
        return {
            'vertices': vertices,
            'faces': cylinder.faces,
            'normals': (cylinder.vertex_normals @ rotation_matrix.T)
        }
    
    def _calculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate vertex normals"""
        
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-6)
            
            # Add to vertex normals
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-6)
        
        return normals
    
    def _value_to_color(self, value: float, cmap: str = "viridis") -> np.ndarray:
        """Convert scalar value to color"""
        
        # Normalize value to [0, 1]
        norm_value = (value - self._color_min) / (self._color_max - self._color_min + 1e-6)
        norm_value = np.clip(norm_value, 0, 1)
        
        # Use HSV for smooth gradients
        hue = norm_value * 0.8  # 0 to 0.8 (red to purple)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        
        return np.array([rgb[0], rgb[1], rgb[2], 1.0])
    
    def _force_directed_layout(
        self,
        nodes: List[Dict],
        edges: List[Tuple[int, int]],
        iterations: int = 100
    ) -> np.ndarray:
        """Calculate force-directed graph layout"""
        
        n = len(nodes)
        positions = np.random.randn(n, 3) * 5
        velocities = np.zeros((n, 3))
        
        # Parameters
        k_repel = 1.0
        k_attract = 0.1
        damping = 0.9
        
        for _ in range(iterations):
            forces = np.zeros((n, 3))
            
            # Repulsive forces between all nodes
            for i in range(n):
                for j in range(i + 1, n):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff) + 0.1
                    force = k_repel * diff / (dist ** 3)
                    forces[i] += force
                    forces[j] -= force
            
            # Attractive forces along edges
            for i, j in edges:
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff)
                force = k_attract * diff * dist
                forces[i] += force
                forces[j] -= force
            
            # Update positions
            velocities = velocities * damping + forces * 0.01
            positions += velocities
        
        return positions
    
    def _circular_layout(self, n: int) -> np.ndarray:
        """Generate circular layout"""
        
        positions = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = 5 * np.cos(angle)
            z = 5 * np.sin(angle)
            y = 0
            positions.append([x, y, z])
        
        return np.array(positions)
    
    def _random_layout(self, n: int) -> np.ndarray:
        """Generate random layout"""
        return np.random.uniform(-5, 5, (n, 3))
    
    def _generate_sphere_points(self, n: int, radius: float) -> np.ndarray:
        """Generate points uniformly distributed on sphere"""
        
        points = []
        for _ in range(n):
            # Generate random point on unit sphere
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            points.append([x, y, z])
        
        return np.array(points)


class VRRenderer:
    """VR/AR rendering engine"""
    
    def __init__(self, engine: RenderingEngine = RenderingEngine.OPENGL):
        self.engine = engine
        self.window = None
        self.context = None
        self.shaders = {}
        self.framebuffers = {}
        self.render_targets = {}
        self.initialized = False
        
    async def initialize(self, width: int = 1920, height: int = 1080):
        """Initialize rendering context"""
        
        if self.engine == RenderingEngine.OPENGL:
            await self._init_opengl(width, height)
        elif self.engine == RenderingEngine.WEBGL:
            await self._init_webgl(width, height)
        else:
            raise ValueError(f"Unsupported rendering engine: {self.engine}")
        
        # Load shaders
        await self._load_shaders()
        
        # Setup render targets
        await self._setup_render_targets(width, height)
        
        self.initialized = True
        logger.info(f"VR renderer initialized with {self.engine.value}")
    
    async def _init_opengl(self, width: int, height: int):
        """Initialize OpenGL context"""
        
        pygame.init()
        pygame.display.set_mode(
            (width, height),
            DOUBLEBUF | OPENGL
        )
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set viewport
        glViewport(0, 0, width, height)
        
        # Create moderngl context
        self.context = moderngl.create_context()
    
    async def _init_webgl(self, width: int, height: int):
        """Initialize WebGL context"""
        # This would be implemented for web-based rendering
        pass
    
    async def _load_shaders(self):
        """Load shader programs"""
        
        # Standard shader
        vertex_shader = '''
        #version 330
        
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        
        in vec3 a_position;
        in vec3 a_normal;
        in vec2 a_texcoord;
        
        out vec3 v_position;
        out vec3 v_normal;
        out vec2 v_texcoord;
        
        void main() {
            v_position = (u_model * vec4(a_position, 1.0)).xyz;
            v_normal = normalize((u_model * vec4(a_normal, 0.0)).xyz);
            v_texcoord = a_texcoord;
            
            gl_Position = u_projection * u_view * vec4(v_position, 1.0);
        }
        '''
        
        fragment_shader = '''
        #version 330
        
        uniform vec4 u_color;
        uniform float u_metallic;
        uniform float u_roughness;
        uniform vec3 u_light_pos;
        uniform vec3 u_camera_pos;
        
        in vec3 v_position;
        in vec3 v_normal;
        in vec2 v_texcoord;
        
        out vec4 f_color;
        
        void main() {
            vec3 normal = normalize(v_normal);
            vec3 light_dir = normalize(u_light_pos - v_position);
            vec3 view_dir = normalize(u_camera_pos - v_position);
            vec3 half_dir = normalize(light_dir + view_dir);
            
            // Simple lighting
            float NdotL = max(dot(normal, light_dir), 0.0);
            float NdotH = max(dot(normal, half_dir), 0.0);
            
            vec3 diffuse = u_color.rgb * NdotL;
            float spec = pow(NdotH, 32.0) * (1.0 - u_roughness);
            vec3 specular = vec3(spec) * u_metallic;
            
            f_color = vec4(diffuse + specular, u_color.a);
        }
        '''
        
        self.shaders['standard'] = self.context.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    async def _setup_render_targets(self, width: int, height: int):
        """Setup render targets for VR stereo rendering"""
        
        # Left eye render target
        self.render_targets['left_eye'] = self.context.texture(
            (width // 2, height),
            4
        )
        self.framebuffers['left_eye'] = self.context.framebuffer(
            color_attachments=[self.render_targets['left_eye']]
        )
        
        # Right eye render target
        self.render_targets['right_eye'] = self.context.texture(
            (width // 2, height),
            4
        )
        self.framebuffers['right_eye'] = self.context.framebuffer(
            color_attachments=[self.render_targets['right_eye']]
        )
    
    async def render_scene(
        self,
        scene: Scene3D,
        stereo: bool = True,
        eye_separation: float = 0.065
    ) -> Optional[np.ndarray]:
        """Render scene"""
        
        if not self.initialized:
            await self.initialize()
        
        with render_time.time():
            if stereo:
                # Render for VR (left and right eye)
                left_image = await self._render_eye(
                    scene,
                    eye_offset=-eye_separation / 2
                )
                right_image = await self._render_eye(
                    scene,
                    eye_offset=eye_separation / 2
                )
                
                # Combine images side by side
                return np.hstack([left_image, right_image])
            else:
                # Render single view
                return await self._render_eye(scene, eye_offset=0)
    
    async def _render_eye(
        self,
        scene: Scene3D,
        eye_offset: float
    ) -> np.ndarray:
        """Render scene from one eye perspective"""
        
        # Clear
        self.context.clear(0.1, 0.1, 0.2, 1.0)
        
        # Setup camera with eye offset
        camera = scene.camera
        eye_position = camera.position + np.array([eye_offset, 0, 0])
        
        # Calculate matrices
        view_matrix = self._calculate_view_matrix(
            eye_position,
            camera.target,
            camera.up
        )
        projection_matrix = self._calculate_projection_matrix(
            camera.fov,
            1.0,  # aspect ratio
            camera.near,
            camera.far
        )
        
        # Render each object
        for obj in scene.objects.values():
            if not obj.visible:
                continue
            
            await self._render_object(
                obj,
                view_matrix,
                projection_matrix,
                eye_position,
                scene.lights
            )
        
        # Read pixels
        pixels = self.context.screen.read()
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.context.screen.height,
            self.context.screen.width,
            3
        )
        
        # Track FPS
        render_fps.observe(60)  # Placeholder - would calculate actual FPS
        
        return image
    
    async def _render_object(
        self,
        obj: Object3D,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        camera_pos: np.ndarray,
        lights: List[Light3D]
    ):
        """Render single object"""
        
        if not obj.mesh:
            return
        
        # Get shader
        shader = self.shaders.get(obj.material.shader, self.shaders['standard'])
        
        # Set uniforms
        model_matrix = obj.transform.to_matrix()
        
        shader['u_model'].write(model_matrix.astype('f4').tobytes())
        shader['u_view'].write(view_matrix.astype('f4').tobytes())
        shader['u_projection'].write(projection_matrix.astype('f4').tobytes())
        
        shader['u_color'].value = tuple(obj.material.color)
        shader['u_metallic'].value = obj.material.metallic
        shader['u_roughness'].value = obj.material.roughness
        shader['u_camera_pos'].value = tuple(camera_pos)
        
        # Set light (use first light for now)
        if lights:
            shader['u_light_pos'].value = tuple(lights[0].position)
        
        # Create vertex buffer
        vertices = obj.mesh.get('vertices', [])
        faces = obj.mesh.get('faces', [])
        normals = obj.mesh.get('normals', [])
        
        if not vertices or not faces:
            return
        
        # Flatten data
        vertex_data = []
        for face in faces:
            for idx in face:
                vertex_data.extend(vertices[idx])
                if idx < len(normals):
                    vertex_data.extend(normals[idx])
                else:
                    vertex_data.extend([0, 1, 0])  # Default normal
                vertex_data.extend([0, 0])  # Texture coords
        
        vertex_buffer = self.context.buffer(
            np.array(vertex_data, dtype='f4').tobytes()
        )
        
        # Create vertex array
        vao = self.context.vertex_array(
            shader,
            [(vertex_buffer, '3f 3f 2f', 'a_position', 'a_normal', 'a_texcoord')]
        )
        
        # Render
        vao.render(moderngl.TRIANGLES)
        
        # Render children
        for child in obj.children:
            await self._render_object(
                child,
                view_matrix,
                projection_matrix,
                camera_pos,
                lights
            )
    
    def _calculate_view_matrix(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray
    ) -> np.ndarray:
        """Calculate view matrix"""
        
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.dot(view[:3, :3], eye)
        
        return view
    
    def _calculate_projection_matrix(
        self,
        fov: float,
        aspect: float,
        near: float,
        far: float
    ) -> np.ndarray:
        """Calculate projection matrix"""
        
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        projection = np.zeros((4, 4))
        projection[0, 0] = f / aspect
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = (2 * far * near) / (near - far)
        projection[3, 2] = -1
        
        return projection


class InteractionHandler:
    """Handle VR/AR interactions"""
    
    def __init__(self):
        self.gesture_recognizer = GestureRecognizer()
        self.selected_object = None
        self.interaction_mode = InteractionMode.GESTURE
        self.haptic_enabled = True
        self.voice_commands = {}
        self.gaze_timer = 0
        self.gaze_target = None
    
    async def process_input(
        self,
        input_data: Dict[str, Any],
        scene: Scene3D
    ) -> List[Dict[str, Any]]:
        """Process user input and return actions"""
        
        actions = []
        
        if self.interaction_mode == InteractionMode.GESTURE:
            actions.extend(await self._process_gesture(input_data, scene))
        
        elif self.interaction_mode == InteractionMode.GAZE:
            actions.extend(await self._process_gaze(input_data, scene))
        
        elif self.interaction_mode == InteractionMode.CONTROLLER:
            actions.extend(await self._process_controller(input_data, scene))
        
        elif self.interaction_mode == InteractionMode.VOICE:
            actions.extend(await self._process_voice(input_data, scene))
        
        # Track interactions
        for action in actions:
            object_interactions.labels(action['type']).inc()
        
        return actions
    
    async def _process_gesture(
        self,
        input_data: Dict[str, Any],
        scene: Scene3D
    ) -> List[Dict[str, Any]]:
        """Process hand gesture input"""
        
        actions = []
        
        # Get hand tracking data
        if 'hand_image' in input_data:
            gesture = await self.gesture_recognizer.recognize_gesture(
                input_data['hand_image']
            )
            
            if gesture == "pinch":
                # Ray cast from hand position
                if 'hand_position' in input_data:
                    hit_object = self._raycast(
                        input_data['hand_position'],
                        input_data.get('hand_direction', [0, 0, -1]),
                        scene
                    )
                    
                    if hit_object:
                        actions.append({
                            'type': 'select',
                            'object': hit_object.id,
                            'position': input_data['hand_position']
                        })
                        self.selected_object = hit_object
                        
                        if self.haptic_enabled:
                            actions.append({
                                'type': 'haptic',
                                'pattern': 'click',
                                'intensity': 0.7
                            })
            
            elif gesture == "grab_release" and self.selected_object:
                # Move selected object
                if 'hand_position' in input_data:
                    self.selected_object.transform.position = np.array(
                        input_data['hand_position']
                    )
                    
                    actions.append({
                        'type': 'move',
                        'object': self.selected_object.id,
                        'position': input_data['hand_position']
                    })
            
            elif gesture == "open_palm":
                # Deselect
                if self.selected_object:
                    actions.append({
                        'type': 'deselect',
                        'object': self.selected_object.id
                    })
                    self.selected_object = None
            
            elif gesture == "zoom_in":
                # Scale up selected object
                if self.selected_object:
                    self.selected_object.transform.scale *= 1.2
                    actions.append({
                        'type': 'scale',
                        'object': self.selected_object.id,
                        'scale': self.selected_object.transform.scale.tolist()
                    })
            
            elif gesture == "zoom_out":
                # Scale down selected object
                if self.selected_object:
                    self.selected_object.transform.scale *= 0.8
                    actions.append({
                        'type': 'scale',
                        'object': self.selected_object.id,
                        'scale': self.selected_object.transform.scale.tolist()
                    })
        
        return actions
    
    async def _process_gaze(
        self,
        input_data: Dict[str, Any],
        scene: Scene3D
    ) -> List[Dict[str, Any]]:
        """Process gaze-based input"""
        
        actions = []
        
        if 'gaze_origin' in input_data and 'gaze_direction' in input_data:
            # Raycast from gaze
            hit_object = self._raycast(
                input_data['gaze_origin'],
                input_data['gaze_direction'],
                scene
            )
            
            if hit_object:
                if hit_object == self.gaze_target:
                    # Continue gazing at same object
                    self.gaze_timer += input_data.get('delta_time', 0.016)
                    
                    # Select after 2 seconds of gazing
                    if self.gaze_timer >= 2.0:
                        actions.append({
                            'type': 'select',
                            'object': hit_object.id,
                            'method': 'gaze'
                        })
                        self.selected_object = hit_object
                        self.gaze_timer = 0
                else:
                    # New gaze target
                    self.gaze_target = hit_object
                    self.gaze_timer = 0
                    
                    actions.append({
                        'type': 'hover',
                        'object': hit_object.id
                    })
            else:
                # Not looking at any object
                self.gaze_target = None
                self.gaze_timer = 0
        
        return actions
    
    async def _process_controller(
        self,
        input_data: Dict[str, Any],
        scene: Scene3D
    ) -> List[Dict[str, Any]]:
        """Process VR controller input"""
        
        actions = []
        
        # Trigger button
        if input_data.get('trigger_pressed'):
            if 'controller_position' in input_data:
                hit_object = self._raycast(
                    input_data['controller_position'],
                    input_data.get('controller_forward', [0, 0, -1]),
                    scene
                )
                
                if hit_object:
                    if self.selected_object == hit_object:
                        # Deselect
                        actions.append({
                            'type': 'deselect',
                            'object': hit_object.id
                        })
                        self.selected_object = None
                    else:
                        # Select
                        actions.append({
                            'type': 'select',
                            'object': hit_object.id
                        })
                        self.selected_object = hit_object
        
        # Grip button - grab object
        if input_data.get('grip_pressed') and self.selected_object:
            self.selected_object.transform.position = np.array(
                input_data.get('controller_position', [0, 0, 0])
            )
            
            actions.append({
                'type': 'grab',
                'object': self.selected_object.id,
                'position': input_data.get('controller_position')
            })
        
        # Thumbstick - teleport or rotate
        thumbstick = input_data.get('thumbstick', [0, 0])
        if abs(thumbstick[0]) > 0.5 or abs(thumbstick[1]) > 0.5:
            if input_data.get('thumbstick_pressed'):
                # Teleport
                teleport_pos = input_data.get('controller_position', [0, 0, 0])
                teleport_pos[0] += thumbstick[0] * 2
                teleport_pos[2] += thumbstick[1] * 2
                
                actions.append({
                    'type': 'teleport',
                    'position': teleport_pos
                })
            else:
                # Rotate view
                actions.append({
                    'type': 'rotate_view',
                    'rotation': [thumbstick[0] * 45, thumbstick[1] * 45]
                })
        
        # Menu button
        if input_data.get('menu_pressed'):
            actions.append({
                'type': 'open_menu'
            })
        
        return actions
    
    async def _process_voice(
        self,
        input_data: Dict[str, Any],
        scene: Scene3D
    ) -> List[Dict[str, Any]]:
        """Process voice commands"""
        
        actions = []
        
        if 'voice_command' in input_data:
            command = input_data['voice_command'].lower()
            
            # Parse voice commands
            if "select" in command:
                # Extract object name
                for obj in scene.objects.values():
                    if obj.name.lower() in command:
                        actions.append({
                            'type': 'select',
                            'object': obj.id,
                            'method': 'voice'
                        })
                        self.selected_object = obj
                        break
            
            elif "show" in command:
                if "all" in command:
                    for obj in scene.objects.values():
                        obj.visible = True
                    actions.append({'type': 'show_all'})
                else:
                    # Show specific object
                    for obj in scene.objects.values():
                        if obj.name.lower() in command:
                            obj.visible = True
                            actions.append({
                                'type': 'show',
                                'object': obj.id
                            })
            
            elif "hide" in command:
                if "all" in command:
                    for obj in scene.objects.values():
                        obj.visible = False
                    actions.append({'type': 'hide_all'})
                elif self.selected_object:
                    self.selected_object.visible = False
                    actions.append({
                        'type': 'hide',
                        'object': self.selected_object.id
                    })
            
            elif "rotate" in command:
                if self.selected_object:
                    # Extract rotation amount
                    angle = 45  # Default
                    if "90" in command:
                        angle = 90
                    elif "180" in command:
                        angle = 180
                    
                    axis = [0, 1, 0]  # Default Y axis
                    if "x" in command:
                        axis = [1, 0, 0]
                    elif "z" in command:
                        axis = [0, 0, 1]
                    
                    actions.append({
                        'type': 'rotate',
                        'object': self.selected_object.id,
                        'angle': angle,
                        'axis': axis
                    })
            
            elif "color" in command or "colour" in command:
                if self.selected_object:
                    # Extract color
                    color = [1, 1, 1, 1]
                    if "red" in command:
                        color = [1, 0, 0, 1]
                    elif "green" in command:
                        color = [0, 1, 0, 1]
                    elif "blue" in command:
                        color = [0, 0, 1, 1]
                    elif "yellow" in command:
                        color = [1, 1, 0, 1]
                    
                    self.selected_object.material.color = np.array(color)
                    actions.append({
                        'type': 'change_color',
                        'object': self.selected_object.id,
                        'color': color
                    })
        
        return actions
    
    def _raycast(
        self,
        origin: List[float],
        direction: List[float],
        scene: Scene3D,
        max_distance: float = 100.0
    ) -> Optional[Object3D]:
        """Perform raycast to find hit object"""
        
        origin = np.array(origin)
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        
        closest_hit = None
        closest_distance = max_distance
        
        for obj in scene.objects.values():
            if not obj.visible or not obj.interactive:
                continue
            
            # Simple sphere intersection test
            # In production, would use proper mesh intersection
            min_bounds, max_bounds = obj.get_bounds()
            center = (min_bounds + max_bounds) / 2
            radius = np.linalg.norm(max_bounds - min_bounds) / 2
            
            # Transform center to world space
            center = obj.transform.to_matrix()[:3, :3] @ center + obj.transform.position
            
            # Ray-sphere intersection
            oc = origin - center
            a = np.dot(direction, direction)
            b = 2.0 * np.dot(oc, direction)
            c = np.dot(oc, oc) - radius * radius
            
            discriminant = b * b - 4 * a * c
            
            if discriminant > 0:
                # Hit! Calculate distance
                t = (-b - np.sqrt(discriminant)) / (2.0 * a)
                
                if 0 < t < closest_distance:
                    closest_hit = obj
                    closest_distance = t
        
        return closest_hit


class ARVRVisualizationSystem:
    """Main AR/VR Visualization System for MCP Stack"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.mesh_generator = DataMeshGenerator()
        self.renderer = VRRenderer()
        self.interaction_handler = InteractionHandler()
        self.scenes: Dict[str, Scene3D] = {}
        self.active_scene = None
        self.websocket_clients = set()
        self.device_type = DeviceType.DESKTOP_VR
        self.render_thread = None
        self.is_running = False
    
    async def create_visualization(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        viz_type: VisualizationType,
        options: Dict[str, Any] = {}
    ) -> str:
        """Create new visualization from data"""
        
        scene_id = f"viz_{datetime.utcnow().timestamp()}"
        scene = Scene3D(id=scene_id, name=options.get('name', 'Data Visualization'))
        
        # Generate appropriate mesh based on type
        if viz_type == VisualizationType.SCATTER_3D:
            objects = await self.mesh_generator.generate_scatter_mesh(
                data,
                options.get('x', 'x'),
                options.get('y', 'y'),
                options.get('z', 'z'),
                options.get('size'),
                options.get('color')
            )
            for obj in objects:
                scene.add_object(obj)
        
        elif viz_type == VisualizationType.SURFACE_3D:
            surface = await self.mesh_generator.generate_surface_mesh(
                data,
                options.get('x_range', (-5, 5)),
                options.get('y_range', (-5, 5))
            )
            scene.add_object(surface)
        
        elif viz_type == VisualizationType.NETWORK_GRAPH:
            objects = await self.mesh_generator.generate_network_mesh(
                options.get('nodes', []),
                options.get('edges', []),
                options.get('layout', 'force')
            )
            for obj in objects:
                scene.add_object(obj)
        
        elif viz_type == VisualizationType.VOLUME_3D:
            volume = await self.mesh_generator.generate_volume_mesh(
                data,
                options.get('threshold', 0.5)
            )
            scene.add_object(volume)
        
        elif viz_type == VisualizationType.PARTICLE_SYSTEM:
            particles = await self.mesh_generator.generate_particle_system(
                options.get('num_particles', 1000),
                options.get('emitter_shape', 'sphere'),
                options.get('velocity_field')
            )
            scene.add_object(particles)
        
        # Add default lighting
        scene.lights.append(
            Light3D(
                type="directional",
                position=np.array([5, 10, 5]),
                direction=np.array([-1, -2, -1]),
                intensity=0.8
            )
        )
        
        scene.lights.append(
            Light3D(
                type="ambient",
                color=np.array([0.2, 0.2, 0.3]),
                intensity=0.3
            )
        )
        
        # Setup camera
        scene.camera = Camera3D(
            position=np.array([0, 5, 10]),
            target=np.array([0, 0, 0])
        )
        
        # Add to scenes
        self.scenes[scene_id] = scene
        
        # Set as active if first scene
        if not self.active_scene:
            self.active_scene = scene
        
        logger.info(f"Created {viz_type.value} visualization: {scene_id}")
        
        return scene_id
    
    async def start_session(
        self,
        device: DeviceType = DeviceType.DESKTOP_VR,
        render_resolution: Tuple[int, int] = (1920, 1080)
    ):
        """Start AR/VR session"""
        
        self.device_type = device
        
        # Initialize renderer
        await self.renderer.initialize(
            width=render_resolution[0],
            height=render_resolution[1]
        )
        
        # Start render loop
        self.is_running = True
        self.render_thread = threading.Thread(
            target=lambda: asyncio.run(self._render_loop())
        )
        self.render_thread.start()
        
        # Start WebSocket server for remote connections
        if device == DeviceType.WEB_XR:
            asyncio.create_task(self._start_websocket_server())
        
        # Increment session counter
        vr_sessions.inc()
        
        logger.info(f"Started AR/VR session for {device.value}")
    
    async def stop_session(self):
        """Stop AR/VR session"""
        
        self.is_running = False
        
        if self.render_thread:
            self.render_thread.join()
        
        # Close WebSocket connections
        for client in self.websocket_clients:
            await client.close()
        
        # Decrement session counter
        vr_sessions.dec()
        
        logger.info("Stopped AR/VR session")
    
    async def _render_loop(self):
        """Main render loop"""
        
        clock = pygame.time.Clock()
        frame_count = 0
        fps_timer = time.time()
        
        while self.is_running:
            # Check for input
            input_data = await self._get_input()
            
            # Process interactions
            if self.active_scene and input_data:
                actions = await self.interaction_handler.process_input(
                    input_data,
                    self.active_scene
                )
                
                # Apply actions
                for action in actions:
                    await self._apply_action(action)
            
            # Update animations
            if self.active_scene:
                await self._update_animations(1/60.0)
            
            # Render scene
            if self.active_scene:
                frame = await self.renderer.render_scene(
                    self.active_scene,
                    stereo=(self.device_type != DeviceType.MOBILE_AR)
                )
                
                # Send frame to connected clients
                if frame is not None and self.websocket_clients:
                    await self._broadcast_frame(frame)
            
            # Update FPS
            frame_count += 1
            current_time = time.time()
            if current_time - fps_timer >= 1.0:
                fps = frame_count / (current_time - fps_timer)
                render_fps.observe(fps)
                frame_count = 0
                fps_timer = current_time
            
            # Cap at 90 FPS for VR
            clock.tick(90)
    
    async def _get_input(self) -> Dict[str, Any]:
        """Get input from device"""
        
        input_data = {}
        
        # Get pygame events
        for event in pygame.event.get():
            if event.type == QUIT:
                self.is_running = False
            
            elif event.type == MOUSEMOTION:
                # Convert mouse to gaze for desktop
                if self.device_type == DeviceType.DESKTOP_VR:
                    x, y = event.pos
                    # Convert to normalized device coordinates
                    width, height = pygame.display.get_surface().get_size()
                    ndc_x = (2.0 * x / width) - 1.0
                    ndc_y = 1.0 - (2.0 * y / height)
                    
                    # Calculate gaze direction
                    input_data['gaze_origin'] = self.active_scene.camera.position.tolist()
                    input_data['gaze_direction'] = [ndc_x, ndc_y, -1.0]
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    input_data['trigger_pressed'] = True
                elif event.button == 3:  # Right click
                    input_data['menu_pressed'] = True
            
            elif event.type == KEYDOWN:
                # Keyboard shortcuts
                if event.key == K_SPACE:
                    input_data['thumbstick_pressed'] = True
                elif event.key == K_g:
                    input_data['grip_pressed'] = True
                elif event.key == K_v:
                    # Switch to voice mode
                    self.interaction_handler.interaction_mode = InteractionMode.VOICE
                elif event.key == K_h:
                    # Toggle haptic
                    self.interaction_handler.haptic_enabled = not self.interaction_handler.haptic_enabled
        
        # Get hand tracking if available
        if self.device_type in [DeviceType.HOLOLENS, DeviceType.MAGIC_LEAP]:
            # Would get actual hand tracking data here
            pass
        
        return input_data
    
    async def _update_animations(self, delta_time: float):
        """Update object animations"""
        
        for obj in self.active_scene.objects.values():
            for anim in obj.animations:
                if anim['type'] == 'rotate':
                    # Rotate object
                    angle = anim.get('speed', 1.0) * delta_time
                    axis = np.array(anim.get('axis', [0, 1, 0]))
                    
                    # Create rotation quaternion
                    q = quaternion.from_rotation_vector(axis * angle)
                    current_q = quaternion.from_float_array(obj.transform.rotation)
                    new_q = q * current_q
                    
                    obj.transform.rotation = quaternion.as_float_array(new_q)
                
                elif anim['type'] == 'particle_update':
                    # Update particle positions
                    if 'vertices' in obj.mesh and 'velocities' in obj.mesh:
                        vertices = obj.mesh['vertices']
                        velocities = obj.mesh['velocities']
                        
                        # Update positions
                        vertices += velocities * delta_time
                        
                        # Apply velocity field if provided
                        if 'velocity_field' in anim and anim['velocity_field']:
                            for i, pos in enumerate(vertices):
                                velocities[i] = anim['velocity_field'](pos)
                        
                        # Respawn particles that go out of bounds
                        out_of_bounds = np.abs(vertices) > 10
                        vertices[out_of_bounds] = np.random.uniform(-1, 1, vertices[out_of_bounds].shape)
    
    async def _apply_action(self, action: Dict[str, Any]):
        """Apply interaction action"""
        
        action_type = action.get('type')
        
        if action_type == 'select':
            # Highlight selected object
            obj_id = action.get('object')
            if obj_id in self.active_scene.objects:
                obj = self.active_scene.objects[obj_id]
                # Add glow effect
                obj.material.emissive = obj.material.color[:3] * 0.3
        
        elif action_type == 'deselect':
            # Remove highlight
            obj_id = action.get('object')
            if obj_id in self.active_scene.objects:
                obj = self.active_scene.objects[obj_id]
                obj.material.emissive = np.array([0, 0, 0])
        
        elif action_type == 'move':
            # Object already moved in interaction handler
            pass
        
        elif action_type == 'rotate':
            # Rotate object
            obj_id = action.get('object')
            if obj_id in self.active_scene.objects:
                obj = self.active_scene.objects[obj_id]
                angle = np.radians(action.get('angle', 45))
                axis = np.array(action.get('axis', [0, 1, 0]))
                
                q = quaternion.from_rotation_vector(axis * angle)
                current_q = quaternion.from_float_array(obj.transform.rotation)
                new_q = q * current_q
                
                obj.transform.rotation = quaternion.as_float_array(new_q)
        
        elif action_type == 'scale':
            # Scale object
            obj_id = action.get('object')
            if obj_id in self.active_scene.objects:
                obj = self.active_scene.objects[obj_id]
                obj.transform.scale = np.array(action.get('scale', [1, 1, 1]))
        
        elif action_type == 'teleport':
            # Move camera
            self.active_scene.camera.position = np.array(action.get('position', [0, 0, 0]))
        
        elif action_type == 'haptic':
            # Trigger haptic feedback
            await self._trigger_haptic(
                action.get('pattern', 'click'),
                action.get('intensity', 0.5)
            )
        
        # Broadcast action to connected clients
        if self.websocket_clients:
            await self._broadcast_action(action)
    
    async def _trigger_haptic(self, pattern: str, intensity: float):
        """Trigger haptic feedback"""
        
        # This would interface with actual haptic devices
        logger.info(f"Haptic feedback: {pattern} at {intensity}")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for WebXR"""
        
        async def handle_client(websocket, path):
            """Handle WebSocket client"""
            self.websocket_clients.add(websocket)
            
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data['type'] == 'input':
                        # Process remote input
                        input_data = data['data']
                        actions = await self.interaction_handler.process_input(
                            input_data,
                            self.active_scene
                        )
                        
                        for action in actions:
                            await self._apply_action(action)
                    
                    elif data['type'] == 'get_scene':
                        # Send scene data
                        scene_data = self._serialize_scene(self.active_scene)
                        await websocket.send(json.dumps({
                            'type': 'scene',
                            'data': scene_data
                        }))
                    
            finally:
                self.websocket_clients.remove(websocket)
        
        # Start server
        await websockets.serve(handle_client, "localhost", 8765)
        logger.info("WebSocket server started on ws://localhost:8765")
    
    async def _broadcast_frame(self, frame: np.ndarray):
        """Broadcast frame to WebSocket clients"""
        
        # Compress frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_data = buffer.tobytes()
        
        message = {
            'type': 'frame',
            'data': frame_data.hex(),
            'timestamp': time.time()
        }
        
        # Send to all clients
        if self.websocket_clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.websocket_clients]
            )
    
    async def _broadcast_action(self, action: Dict[str, Any]):
        """Broadcast action to WebSocket clients"""
        
        message = {
            'type': 'action',
            'data': action,
            'timestamp': time.time()
        }
        
        if self.websocket_clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.websocket_clients]
            )
    
    def _serialize_scene(self, scene: Scene3D) -> Dict[str, Any]:
        """Serialize scene for transmission"""
        
        return {
            'id': scene.id,
            'name': scene.name,
            'objects': [
                {
                    'id': obj.id,
                    'name': obj.name,
                    'transform': {
                        'position': obj.transform.position.tolist(),
                        'rotation': obj.transform.rotation.tolist(),
                        'scale': obj.transform.scale.tolist()
                    },
                    'material': {
                        'color': obj.material.color.tolist(),
                        'metallic': obj.material.metallic,
                        'roughness': obj.material.roughness
                    },
                    'visible': obj.visible
                }
                for obj in scene.objects.values()
            ],
            'camera': {
                'position': scene.camera.position.tolist(),
                'target': scene.camera.target.tolist(),
                'fov': scene.camera.fov
            }
        }
    
    async def export_scene(
        self,
        scene_id: str,
        format: str = "gltf"
    ) -> bytes:
        """Export scene to standard format"""
        
        if scene_id not in self.scenes:
            raise ValueError(f"Scene {scene_id} not found")
        
        scene = self.scenes[scene_id]
        
        if format == "gltf":
            # Export to glTF format
            # This would use a library like trimesh or custom exporter
            pass
        
        elif format == "usd":
            # Export to Universal Scene Description
            pass
        
        elif format == "fbx":
            # Export to FBX
            pass
        
        return b""  # Placeholder


# Example usage
async def arvr_demo():
    """Demo AR/VR visualization"""
    
    # Initialize system
    arvr = ARVRVisualizationSystem('postgresql://user:pass@localhost/arvr_db')
    
    # Create sample data
    # 3D scatter plot
    n_points = 100
    scatter_data = pd.DataFrame({
        'x': np.random.randn(n_points),
        'y': np.random.randn(n_points),
        'z': np.random.randn(n_points),
        'value': np.random.rand(n_points),
        'category': np.random.choice(['A', 'B', 'C'], n_points)
    })
    
    scene_id = await arvr.create_visualization(
        scatter_data,
        VisualizationType.SCATTER_3D,
        {
            'x': 'x',
            'y': 'y',
            'z': 'z',
            'color': 'value',
            'name': '3D Scatter Visualization'
        }
    )
    
    print(f"Created scatter visualization: {scene_id}")
    
    # Network graph
    nodes = [
        {'name': f'Node {i}', 'value': np.random.rand()}
        for i in range(20)
    ]
    edges = [(i, (i + 1) % 20) for i in range(20)]
    edges.extend([(i, (i + 5) % 20) for i in range(0, 20, 3)])
    
    network_id = await arvr.create_visualization(
        None,
        VisualizationType.NETWORK_GRAPH,
        {
            'nodes': nodes,
            'edges': edges,
            'layout': 'force',
            'name': 'Network Visualization'
        }
    )
    
    print(f"Created network visualization: {network_id}")
    
    # Surface plot
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    surface_id = await arvr.create_visualization(
        Z,
        VisualizationType.SURFACE_3D,
        {
            'x_range': (-5, 5),
            'y_range': (-5, 5),
            'name': 'Surface Visualization'
        }
    )
    
    print(f"Created surface visualization: {surface_id}")
    
    # Particle system
    particle_id = await arvr.create_visualization(
        None,
        VisualizationType.PARTICLE_SYSTEM,
        {
            'num_particles': 500,
            'emitter_shape': 'sphere',
            'velocity_field': lambda pos: np.array([
                -pos[1] * 0.1,
                pos[0] * 0.1,
                np.sin(pos[2]) * 0.05
            ]),
            'name': 'Particle System'
        }
    )
    
    print(f"Created particle visualization: {particle_id}")
    
    # Start VR session
    await arvr.start_session(DeviceType.DESKTOP_VR)
    
    print("\nVR Session Started!")
    print("Controls:")
    print("- Mouse: Look around / Gaze selection")
    print("- Left Click: Select object")
    print("- Right Click: Open menu")
    print("- V: Switch to voice mode")
    print("- H: Toggle haptic feedback")
    print("- ESC: Exit")
    
    # Keep running
    try:
        await asyncio.sleep(60)  # Run for 60 seconds
    except KeyboardInterrupt:
        pass
    
    # Stop session
    await arvr.stop_session()
    print("\nVR Session ended")


if __name__ == "__main__":
    # Set up data range for color mapping
    DataMeshGenerator._color_min = 0
    DataMeshGenerator._color_max = 1
    
    asyncio.run(arvr_demo())