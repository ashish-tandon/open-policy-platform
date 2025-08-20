"""
Neural Interface System - 40by6
Enable direct brain-computer interaction with MCP Stack
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, Protocol
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import zscore, pearsonr
from scipy.spatial.distance import euclidean
import mne
from mne import io, Epochs, find_events
from mne.preprocessing import ICA, create_eog_epochs
from mne.time_frequency import tfr_morlet, psd_multitaper
from mne.decoding import CSP, SPoC, UnsupervisedSpatialFilter
from mne.connectivity import spectral_connectivity
import pywt  # Wavelet transforms
import nolds  # Nonlinear dynamics
import antropy  # Entropy measures
from fooof import FOOOF  # Parameterizing neural power spectra
import yasa  # Sleep staging and analysis
import pyedflib  # EDF file handling
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes
import pylsl  # Lab Streaming Layer
from pylsl import StreamInfo, StreamOutlet, StreamInlet
import serial
import bluetooth
import usb.core
import usb.util
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, JSON, Boolean, Index, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge, Summary
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
import pyaudio
import wave
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
import pygame
import websockets
import aiohttp
import grpc
import msgpack
import cbor2
import h5py
import zarr
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib
import secrets
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Metrics
neural_signals_processed = Counter('neural_signals_processed_total', 'Total neural signals processed', ['signal_type', 'device'])
neural_commands_executed = Counter('neural_commands_executed_total', 'Total neural commands executed', ['command_type', 'success'])
neural_accuracy = Gauge('neural_classification_accuracy', 'Neural signal classification accuracy', ['model', 'signal_type'])
neural_latency = Histogram('neural_processing_latency_seconds', 'Neural processing latency', ['operation'])
neural_signal_quality = Gauge('neural_signal_quality_score', 'Neural signal quality score', ['device', 'channel'])
neural_errors = Counter('neural_errors_total', 'Total neural interface errors', ['error_type'])

Base = declarative_base()


class NeuralDeviceType(Enum):
    """Types of neural interface devices"""
    EEG = "eeg"  # Electroencephalography
    ECOG = "ecog"  # Electrocorticography
    MEG = "meg"  # Magnetoencephalography
    FNIRS = "fnirs"  # Functional near-infrared spectroscopy
    IMPLANT = "implant"  # Invasive neural implant
    TMS = "tms"  # Transcranial magnetic stimulation
    TDCS = "tdcs"  # Transcranial direct current stimulation
    ULTRASOUND = "ultrasound"  # Focused ultrasound
    OPTOGENETIC = "optogenetic"  # Optogenetic interface


class SignalType(Enum):
    """Types of neural signals"""
    RAW = "raw"
    ALPHA = "alpha"  # 8-13 Hz
    BETA = "beta"  # 13-30 Hz
    GAMMA = "gamma"  # 30-100 Hz
    THETA = "theta"  # 4-8 Hz
    DELTA = "delta"  # 0.5-4 Hz
    ERP = "erp"  # Event-related potential
    SSVEP = "ssvep"  # Steady-state visual evoked potential
    P300 = "p300"  # P300 wave
    MOTOR_IMAGERY = "motor_imagery"
    EMOTION = "emotion"
    ATTENTION = "attention"
    SLEEP_STAGE = "sleep_stage"


class NeuralCommand(Enum):
    """Neural command types"""
    CURSOR_MOVE = "cursor_move"
    CLICK = "click"
    SCROLL = "scroll"
    TYPE_TEXT = "type_text"
    SELECT_OPTION = "select_option"
    NAVIGATE = "navigate"
    EXECUTE_FUNCTION = "execute_function"
    QUERY_DATA = "query_data"
    CONTROL_DEVICE = "control_device"
    SEND_MESSAGE = "send_message"
    EMERGENCY_ALERT = "emergency_alert"


class CognitivState(Enum):
    """Cognitive states"""
    FOCUSED = "focused"
    RELAXED = "relaxed"
    STRESSED = "stressed"
    FATIGUED = "fatigued"
    ALERT = "alert"
    DROWSY = "drowsy"
    MEDITATING = "meditating"
    FLOW = "flow"
    CONFUSED = "confused"
    EXCITED = "excited"


@dataclass
class NeuralSignal:
    """Neural signal data"""
    id: str
    timestamp: datetime
    device_id: str
    signal_type: SignalType
    channels: Dict[str, np.ndarray]  # Channel name -> data
    sampling_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'device_id': self.device_id,
            'signal_type': self.signal_type.value,
            'sampling_rate': self.sampling_rate,
            'quality_score': self.quality_score,
            'metadata': self.metadata,
            'channel_count': len(self.channels)
        }


@dataclass
class NeuralDevice:
    """Neural interface device"""
    id: str
    name: str
    device_type: NeuralDeviceType
    channels: List[str]
    sampling_rate: float
    resolution_bits: int
    connection_type: str  # usb, bluetooth, wifi, etc.
    is_connected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'device_type': self.device_type.value,
            'channels': self.channels,
            'sampling_rate': self.sampling_rate,
            'resolution_bits': self.resolution_bits,
            'connection_type': self.connection_type,
            'is_connected': self.is_connected,
            'metadata': self.metadata
        }


@dataclass
class BrainState:
    """Current brain state analysis"""
    timestamp: datetime
    cognitive_state: CognitivState
    attention_level: float  # 0-1
    stress_level: float  # 0-1
    fatigue_level: float  # 0-1
    valence: float  # -1 to 1 (negative to positive emotion)
    arousal: float  # 0-1
    dominant_frequency: float
    band_powers: Dict[str, float]
    connectivity_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NeuralIntent:
    """Decoded neural intent"""
    id: str
    timestamp: datetime
    command: NeuralCommand
    parameters: Dict[str, Any]
    confidence: float
    source_signals: List[str]  # Signal IDs used
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'command': self.command.value,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'processing_time': self.processing_time
        }


class NeuralSignalDB(Base):
    """Database model for neural signals"""
    __tablename__ = 'neural_signals'
    
    id = Column(String(50), primary_key=True)
    timestamp = Column(DateTime, index=True)
    device_id = Column(String(50), index=True)
    signal_type = Column(String(50))
    sampling_rate = Column(Float)
    quality_score = Column(Float)
    data = Column(LargeBinary)  # Compressed signal data
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_neural_device_time', 'device_id', 'timestamp'),
    )


class NeuralIntentDB(Base):
    """Database model for neural intents"""
    __tablename__ = 'neural_intents'
    
    id = Column(String(50), primary_key=True)
    timestamp = Column(DateTime, index=True)
    command = Column(String(50))
    parameters = Column(JSON)
    confidence = Column(Float)
    processing_time = Column(Float)
    executed = Column(Boolean, default=False)
    success = Column(Boolean)
    error = Column(Text)
    
    __table_args__ = (
        Index('idx_intent_command_time', 'command', 'timestamp'),
    )


class SignalProcessor:
    """Process raw neural signals"""
    
    def __init__(self):
        self.filters = {}
        self._init_filters()
    
    def _init_filters(self):
        """Initialize signal filters"""
        # Bandpass filters for different frequency bands
        fs = 250  # Default sampling rate
        
        self.filters['alpha'] = signal.butter(4, [8, 13], btype='band', fs=fs)
        self.filters['beta'] = signal.butter(4, [13, 30], btype='band', fs=fs)
        self.filters['gamma'] = signal.butter(4, [30, 100], btype='band', fs=fs)
        self.filters['theta'] = signal.butter(4, [4, 8], btype='band', fs=fs)
        self.filters['delta'] = signal.butter(4, [0.5, 4], btype='band', fs=fs)
    
    def preprocess_signal(
        self,
        data: np.ndarray,
        sampling_rate: float,
        notch_freq: float = 60.0  # Power line frequency
    ) -> np.ndarray:
        """Preprocess neural signal"""
        
        # Remove DC offset
        data = data - np.mean(data, axis=-1, keepdims=True)
        
        # Notch filter for power line interference
        if notch_freq:
            b, a = signal.iirnotch(notch_freq, 30, sampling_rate)
            data = signal.filtfilt(b, a, data)
        
        # Bandpass filter (0.5-100 Hz)
        sos = signal.butter(4, [0.5, 100], btype='band', fs=sampling_rate, output='sos')
        data = signal.sosfiltfilt(sos, data)
        
        # Remove artifacts using wavelet denoising
        data = self._wavelet_denoise(data)
        
        return data
    
    def _wavelet_denoise(self, data: np.ndarray) -> np.ndarray:
        """Wavelet denoising"""
        
        # Decompose signal
        coeffs = pywt.wavedec(data, 'db4', level=4)
        
        # Estimate noise level
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Soft thresholding
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        coeffs_thresh = [pywt.threshold(c, threshold, 'soft') for c in coeffs]
        
        # Reconstruct
        return pywt.waverec(coeffs_thresh, 'db4')[:len(data)]
    
    def extract_features(
        self,
        signal_data: NeuralSignal
    ) -> Dict[str, Any]:
        """Extract features from neural signal"""
        
        features = {
            'time_domain': {},
            'frequency_domain': {},
            'nonlinear': {},
            'connectivity': {}
        }
        
        # Get first channel for demo
        channel_data = next(iter(signal_data.channels.values()))
        
        # Time domain features
        features['time_domain']['mean'] = float(np.mean(channel_data))
        features['time_domain']['std'] = float(np.std(channel_data))
        features['time_domain']['skew'] = float(signal.skew(channel_data))
        features['time_domain']['kurtosis'] = float(signal.kurtosis(channel_data))
        features['time_domain']['zero_crossings'] = int(np.sum(np.diff(np.sign(channel_data)) != 0))
        
        # Frequency domain features
        freqs, psd = signal.welch(channel_data, signal_data.sampling_rate, nperseg=256)
        
        # Band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        total_power = np.sum(psd)
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            features['frequency_domain'][f'{band_name}_power'] = float(band_power)
            features['frequency_domain'][f'{band_name}_relative'] = float(band_power / total_power)
        
        # Peak frequency
        peak_idx = np.argmax(psd)
        features['frequency_domain']['peak_frequency'] = float(freqs[peak_idx])
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        features['frequency_domain']['spectral_entropy'] = float(
            -np.sum(psd_norm * np.log2(psd_norm + 1e-15))
        )
        
        # Nonlinear features
        try:
            # Approximate entropy
            features['nonlinear']['approx_entropy'] = float(
                antropy.app_entropy(channel_data, order=2, metric='chebyshev')
            )
            
            # Sample entropy
            features['nonlinear']['sample_entropy'] = float(
                antropy.sample_entropy(channel_data, order=2, metric='chebyshev')
            )
            
            # Higuchi fractal dimension
            features['nonlinear']['fractal_dimension'] = float(
                antropy.higuchi_fd(channel_data)
            )
            
            # Lyapunov exponent
            features['nonlinear']['lyapunov'] = float(
                nolds.lyap_r(channel_data, emb_dim=10, lag=1)
            )
        except:
            pass
        
        # Multi-channel connectivity (if multiple channels)
        if len(signal_data.channels) > 1:
            # Convert to 2D array (channels x time)
            multi_channel = np.array(list(signal_data.channels.values()))
            
            # Phase locking value
            connectivity = spectral_connectivity(
                multi_channel[np.newaxis, :, :],
                method='plv',
                sfreq=signal_data.sampling_rate,
                fmin=8,
                fmax=13,  # Alpha band
                verbose=False
            )
            
            features['connectivity']['plv_alpha'] = float(np.mean(connectivity[0]))
        
        return features
    
    def detect_artifacts(
        self,
        data: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, float]:
        """Detect and mark artifacts in signal"""
        
        artifacts = np.zeros(len(data), dtype=bool)
        
        # Amplitude threshold
        amplitude_threshold = 5 * np.std(data)
        artifacts |= np.abs(data) > amplitude_threshold
        
        # Gradient threshold (muscle artifacts)
        gradient = np.gradient(data)
        gradient_threshold = 5 * np.std(gradient)
        artifacts |= np.abs(gradient) > gradient_threshold
        
        # Frequency-based detection (EMG contamination)
        freqs, psd = signal.welch(data, sampling_rate, nperseg=256)
        high_freq_mask = freqs > 40
        high_freq_power = np.sum(psd[high_freq_mask])
        total_power = np.sum(psd)
        
        if high_freq_power / total_power > 0.5:
            # High frequency contamination
            artifacts[:] = True
        
        # Calculate quality score
        quality_score = 1.0 - (np.sum(artifacts) / len(artifacts))
        
        return artifacts, quality_score
    
    def segment_signal(
        self,
        data: np.ndarray,
        sampling_rate: float,
        window_size: float = 1.0,  # seconds
        overlap: float = 0.5  # fraction
    ) -> List[np.ndarray]:
        """Segment signal into windows"""
        
        window_samples = int(window_size * sampling_rate)
        step_samples = int(window_samples * (1 - overlap))
        
        segments = []
        for start in range(0, len(data) - window_samples + 1, step_samples):
            segment = data[start:start + window_samples]
            segments.append(segment)
        
        return segments


class PatternRecognizer:
    """Recognize patterns in neural signals"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.training_data = defaultdict(list)
    
    def train_motor_imagery_classifier(
        self,
        training_signals: List[Tuple[NeuralSignal, str]]  # (signal, label)
    ):
        """Train motor imagery classifier"""
        
        # Extract features
        X = []
        y = []
        
        processor = SignalProcessor()
        
        for signal, label in training_signals:
            features = processor.extract_features(signal)
            
            # Flatten features
            feature_vector = []
            for domain in features.values():
                if isinstance(domain, dict):
                    feature_vector.extend(domain.values())
            
            X.append(feature_vector)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier
        clf = Pipeline([
            ('pca', PCA(n_components=0.95)),  # Keep 95% variance
            ('lda', LinearDiscriminantAnalysis()),
            ('svm', SVC(kernel='rbf', probability=True))
        ])
        
        clf.fit(X_scaled, y)
        
        # Store model and scaler
        self.models['motor_imagery'] = clf
        self.scalers['motor_imagery'] = scaler
        
        # Calculate accuracy
        accuracy = clf.score(X_scaled, y)
        neural_accuracy.labels('motor_imagery', 'motor_imagery').set(accuracy)
        
        logger.info(f"Motor imagery classifier trained with accuracy: {accuracy:.2f}")
    
    def classify_motor_imagery(
        self,
        signal: NeuralSignal
    ) -> Tuple[str, float]:
        """Classify motor imagery"""
        
        if 'motor_imagery' not in self.models:
            raise ValueError("Motor imagery classifier not trained")
        
        processor = SignalProcessor()
        features = processor.extract_features(signal)
        
        # Flatten features
        feature_vector = []
        for domain in features.values():
            if isinstance(domain, dict):
                feature_vector.extend(domain.values())
        
        X = np.array([feature_vector])
        
        # Scale
        X_scaled = self.scalers['motor_imagery'].transform(X)
        
        # Predict
        prediction = self.models['motor_imagery'].predict(X_scaled)[0]
        probability = np.max(self.models['motor_imagery'].predict_proba(X_scaled)[0])
        
        return prediction, probability
    
    def detect_p300(
        self,
        signal: NeuralSignal,
        stimulus_times: List[float]
    ) -> Tuple[bool, float]:
        """Detect P300 event-related potential"""
        
        # P300 typically occurs 250-500ms after stimulus
        p300_window = (0.25, 0.5)  # seconds
        
        # Get signal data
        channel_data = next(iter(signal.channels.values()))
        
        # Extract epochs around stimuli
        epochs = []
        for stim_time in stimulus_times:
            start_idx = int((stim_time + p300_window[0]) * signal.sampling_rate)
            end_idx = int((stim_time + p300_window[1]) * signal.sampling_rate)
            
            if 0 <= start_idx < len(channel_data) and end_idx < len(channel_data):
                epoch = channel_data[start_idx:end_idx]
                epochs.append(epoch)
        
        if not epochs:
            return False, 0.0
        
        # Average epochs
        avg_epoch = np.mean(epochs, axis=0)
        
        # Detect P300 (positive peak)
        peak_amplitude = np.max(avg_epoch)
        baseline = np.mean(channel_data)
        
        # P300 criteria: positive deflection > 5 μV
        p300_detected = peak_amplitude - baseline > 5
        confidence = min(1.0, (peak_amplitude - baseline) / 10)
        
        return p300_detected, confidence
    
    def classify_emotion(
        self,
        signal: NeuralSignal
    ) -> Tuple[float, float]:
        """Classify emotion (valence and arousal)"""
        
        processor = SignalProcessor()
        features = processor.extract_features(signal)
        
        # Emotion classification based on frontal alpha asymmetry
        # and overall band powers
        
        # Get alpha power
        alpha_power = features['frequency_domain'].get('alpha_relative', 0.5)
        beta_power = features['frequency_domain'].get('beta_relative', 0.3)
        
        # Simple model: beta/alpha ratio indicates arousal
        arousal = min(1.0, beta_power / (alpha_power + 0.01))
        
        # Valence from frontal asymmetry (would need specific channels)
        # For demo, using spectral entropy as proxy
        spectral_entropy = features['frequency_domain'].get('spectral_entropy', 0.5)
        valence = (spectral_entropy - 0.5) * 2  # Map to [-1, 1]
        
        return valence, arousal
    
    def detect_attention_level(
        self,
        signal: NeuralSignal
    ) -> float:
        """Detect attention/focus level"""
        
        processor = SignalProcessor()
        features = processor.extract_features(signal)
        
        # Attention correlates with theta/beta ratio
        theta_power = features['frequency_domain'].get('theta_relative', 0.2)
        beta_power = features['frequency_domain'].get('beta_relative', 0.3)
        
        # Lower theta and higher beta indicates attention
        attention = beta_power / (theta_power + beta_power)
        
        return float(attention)
    
    def detect_sleep_stage(
        self,
        signal: NeuralSignal
    ) -> str:
        """Detect sleep stage"""
        
        processor = SignalProcessor()
        features = processor.extract_features(signal)
        
        # Simple sleep stage detection based on dominant frequencies
        delta_power = features['frequency_domain'].get('delta_relative', 0.1)
        theta_power = features['frequency_domain'].get('theta_relative', 0.1)
        alpha_power = features['frequency_domain'].get('alpha_relative', 0.1)
        beta_power = features['frequency_domain'].get('beta_relative', 0.1)
        
        # Find dominant band
        powers = {
            'wake': beta_power + alpha_power,
            'rem': theta_power + beta_power * 0.5,
            'n1': theta_power + alpha_power * 0.5,
            'n2': theta_power,
            'n3': delta_power
        }
        
        sleep_stage = max(powers, key=powers.get)
        
        return sleep_stage


class BrainStateAnalyzer:
    """Analyze overall brain state"""
    
    def __init__(self):
        self.history = deque(maxlen=1000)
        self.baseline_state = None
    
    def analyze_state(
        self,
        signal: NeuralSignal,
        pattern_recognizer: PatternRecognizer
    ) -> BrainState:
        """Analyze current brain state"""
        
        processor = SignalProcessor()
        features = processor.extract_features(signal)
        
        # Get emotion
        valence, arousal = pattern_recognizer.classify_emotion(signal)
        
        # Get attention
        attention = pattern_recognizer.detect_attention_level(signal)
        
        # Estimate stress (high beta, low alpha)
        beta_power = features['frequency_domain'].get('beta_relative', 0.3)
        alpha_power = features['frequency_domain'].get('alpha_relative', 0.2)
        stress = beta_power / (alpha_power + 0.01)
        stress = min(1.0, stress / 3.0)  # Normalize
        
        # Estimate fatigue (high theta, low beta)
        theta_power = features['frequency_domain'].get('theta_relative', 0.2)
        fatigue = theta_power / (beta_power + 0.01)
        fatigue = min(1.0, fatigue / 2.0)  # Normalize
        
        # Determine cognitive state
        if attention > 0.7 and stress < 0.3:
            cognitive_state = CognitivState.FOCUSED
        elif stress > 0.7:
            cognitive_state = CognitivState.STRESSED
        elif fatigue > 0.7:
            cognitive_state = CognitivState.FATIGUED
        elif arousal < 0.3 and valence > 0:
            cognitive_state = CognitivState.RELAXED
        elif attention > 0.8 and arousal > 0.6:
            cognitive_state = CognitivState.FLOW
        else:
            cognitive_state = CognitivState.ALERT
        
        # Get band powers
        band_powers = {
            band: features['frequency_domain'].get(f'{band}_relative', 0.0)
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']
        }
        
        # Create brain state
        state = BrainState(
            timestamp=signal.timestamp,
            cognitive_state=cognitive_state,
            attention_level=attention,
            stress_level=stress,
            fatigue_level=fatigue,
            valence=valence,
            arousal=arousal,
            dominant_frequency=features['frequency_domain'].get('peak_frequency', 10.0),
            band_powers=band_powers
        )
        
        # Store in history
        self.history.append(state)
        
        return state
    
    def detect_state_change(
        self,
        current_state: BrainState,
        threshold: float = 0.3
    ) -> Optional[str]:
        """Detect significant state changes"""
        
        if len(self.history) < 10:
            return None
        
        # Get average of recent states
        recent_states = list(self.history)[-10:-1]
        
        avg_attention = np.mean([s.attention_level for s in recent_states])
        avg_stress = np.mean([s.stress_level for s in recent_states])
        avg_fatigue = np.mean([s.fatigue_level for s in recent_states])
        
        # Check for significant changes
        if abs(current_state.attention_level - avg_attention) > threshold:
            return f"Attention {'increased' if current_state.attention_level > avg_attention else 'decreased'}"
        
        if abs(current_state.stress_level - avg_stress) > threshold:
            return f"Stress {'increased' if current_state.stress_level > avg_stress else 'decreased'}"
        
        if abs(current_state.fatigue_level - avg_fatigue) > threshold:
            return f"Fatigue {'increased' if current_state.fatigue_level > avg_fatigue else 'decreased'}"
        
        return None
    
    def get_cognitive_load(self) -> float:
        """Estimate cognitive load"""
        
        if len(self.history) < 5:
            return 0.5
        
        recent_states = list(self.history)[-5:]
        
        # Cognitive load based on attention variability and stress
        attention_var = np.std([s.attention_level for s in recent_states])
        avg_stress = np.mean([s.stress_level for s in recent_states])
        
        cognitive_load = (attention_var + avg_stress) / 2
        
        return min(1.0, cognitive_load)


class NeuralCommandDecoder:
    """Decode neural signals into commands"""
    
    def __init__(self):
        self.command_models = {}
        self.gesture_patterns = {}
        self._init_gesture_patterns()
    
    def _init_gesture_patterns(self):
        """Initialize gesture patterns"""
        
        # Define patterns for different mental gestures
        self.gesture_patterns = {
            'think_up': {
                'band_changes': {'beta': 1.2, 'gamma': 1.1},
                'spatial_pattern': 'frontal'
            },
            'think_down': {
                'band_changes': {'alpha': 1.3, 'theta': 1.2},
                'spatial_pattern': 'frontal'
            },
            'think_left': {
                'band_changes': {'beta': 1.2},
                'spatial_pattern': 'left_hemisphere'
            },
            'think_right': {
                'band_changes': {'beta': 1.2},
                'spatial_pattern': 'right_hemisphere'
            },
            'think_push': {
                'band_changes': {'beta': 1.5, 'gamma': 1.3},
                'spatial_pattern': 'central'
            }
        }
    
    def decode_intent(
        self,
        signal: NeuralSignal,
        brain_state: BrainState,
        context: Dict[str, Any]
    ) -> Optional[NeuralIntent]:
        """Decode neural intent from signal"""
        
        start_time = datetime.utcnow()
        
        # Check signal quality
        if signal.quality_score < 0.5:
            logger.warning("Signal quality too low for decoding")
            return None
        
        # Extract features
        processor = SignalProcessor()
        features = processor.extract_features(signal)
        
        # Try different decoders
        intent = None
        
        # Motor imagery decoder
        if self._check_motor_imagery_active(features):
            intent = self._decode_motor_imagery(signal, features)
        
        # P300 speller
        elif context.get('p300_mode'):
            intent = self._decode_p300_selection(signal, context)
        
        # Attention-based selection
        elif brain_state.attention_level > 0.8:
            intent = self._decode_attention_selection(signal, brain_state, context)
        
        # Gesture detection
        else:
            intent = self._decode_gesture(signal, features)
        
        if intent:
            # Calculate processing time
            intent.processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            neural_commands_executed.labels(
                intent.command.value,
                'pending'
            ).inc()
            
            with neural_latency.labels('decode').time():
                pass  # Already measured
        
        return intent
    
    def _check_motor_imagery_active(self, features: Dict[str, Any]) -> bool:
        """Check if motor imagery is active"""
        
        # High beta in motor cortex indicates motor imagery
        beta_power = features['frequency_domain'].get('beta_relative', 0.0)
        return beta_power > 0.4
    
    def _decode_motor_imagery(
        self,
        signal: NeuralSignal,
        features: Dict[str, Any]
    ) -> Optional[NeuralIntent]:
        """Decode motor imagery command"""
        
        # Simplified motor imagery detection
        # In reality, would use trained classifier
        
        beta_power = features['frequency_domain'].get('beta_relative', 0.0)
        
        # Map to cursor movement
        if beta_power > 0.5:
            direction = 'up' if features['time_domain']['mean'] > 0 else 'down'
            
            return NeuralIntent(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                command=NeuralCommand.CURSOR_MOVE,
                parameters={'direction': direction, 'distance': 10},
                confidence=min(0.9, beta_power),
                source_signals=[signal.id],
                processing_time=0.0
            )
        
        return None
    
    def _decode_p300_selection(
        self,
        signal: NeuralSignal,
        context: Dict[str, Any]
    ) -> Optional[NeuralIntent]:
        """Decode P300-based selection"""
        
        stimulus_times = context.get('stimulus_times', [])
        options = context.get('options', [])
        
        if not stimulus_times or not options:
            return None
        
        recognizer = PatternRecognizer()
        p300_detected, confidence = recognizer.detect_p300(signal, stimulus_times)
        
        if p300_detected and confidence > 0.6:
            # Find which option was selected
            selected_idx = context.get('current_highlight', 0)
            
            return NeuralIntent(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                command=NeuralCommand.SELECT_OPTION,
                parameters={
                    'option': options[selected_idx],
                    'index': selected_idx
                },
                confidence=confidence,
                source_signals=[signal.id],
                processing_time=0.0
            )
        
        return None
    
    def _decode_attention_selection(
        self,
        signal: NeuralSignal,
        brain_state: BrainState,
        context: Dict[str, Any]
    ) -> Optional[NeuralIntent]:
        """Decode attention-based selection"""
        
        # High sustained attention on an element
        if brain_state.attention_level > 0.85:
            focused_element = context.get('focused_element')
            
            if focused_element:
                return NeuralIntent(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    command=NeuralCommand.CLICK,
                    parameters={'element': focused_element},
                    confidence=brain_state.attention_level,
                    source_signals=[signal.id],
                    processing_time=0.0
                )
        
        return None
    
    def _decode_gesture(
        self,
        signal: NeuralSignal,
        features: Dict[str, Any]
    ) -> Optional[NeuralIntent]:
        """Decode mental gestures"""
        
        # Compare current features to gesture patterns
        best_match = None
        best_score = 0.0
        
        current_bands = features['frequency_domain']
        
        for gesture_name, pattern in self.gesture_patterns.items():
            score = 0.0
            matches = 0
            
            # Check band power changes
            for band, expected_change in pattern['band_changes'].items():
                current_power = current_bands.get(f'{band}_relative', 0.0)
                # Would compare to baseline in real implementation
                baseline_power = 0.2  # Placeholder
                
                actual_change = current_power / (baseline_power + 0.01)
                
                if actual_change > expected_change * 0.8:
                    score += 1.0
                    matches += 1
            
            if matches > 0:
                score /= len(pattern['band_changes'])
                
                if score > best_score:
                    best_score = score
                    best_match = gesture_name
        
        if best_match and best_score > 0.6:
            # Map gesture to command
            command_map = {
                'think_up': (NeuralCommand.CURSOR_MOVE, {'direction': 'up', 'distance': 20}),
                'think_down': (NeuralCommand.CURSOR_MOVE, {'direction': 'down', 'distance': 20}),
                'think_left': (NeuralCommand.CURSOR_MOVE, {'direction': 'left', 'distance': 20}),
                'think_right': (NeuralCommand.CURSOR_MOVE, {'direction': 'right', 'distance': 20}),
                'think_push': (NeuralCommand.CLICK, {})
            }
            
            if best_match in command_map:
                command, params = command_map[best_match]
                
                return NeuralIntent(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    command=command,
                    parameters=params,
                    confidence=best_score,
                    source_signals=[signal.id],
                    processing_time=0.0
                )
        
        return None


class NeuralDeviceInterface:
    """Interface with neural recording devices"""
    
    def __init__(self):
        self.devices: Dict[str, NeuralDevice] = {}
        self.active_streams: Dict[str, Any] = {}
        self.brainflow_boards: Dict[str, BoardShim] = {}
    
    async def connect_device(
        self,
        device_config: Dict[str, Any]
    ) -> Optional[NeuralDevice]:
        """Connect to neural device"""
        
        device_type = device_config.get('type')
        
        if device_type == 'openbci':
            return await self._connect_openbci(device_config)
        elif device_type == 'muse':
            return await self._connect_muse(device_config)
        elif device_type == 'emotiv':
            return await self._connect_emotiv(device_config)
        elif device_type == 'neurosky':
            return await self._connect_neurosky(device_config)
        elif device_type == 'lsl':
            return await self._connect_lsl(device_config)
        else:
            logger.error(f"Unknown device type: {device_type}")
            return None
    
    async def _connect_openbci(
        self,
        config: Dict[str, Any]
    ) -> Optional[NeuralDevice]:
        """Connect to OpenBCI device"""
        
        try:
            # Set up BrainFlow
            params = BrainFlowInputParams()
            params.serial_port = config.get('serial_port', '/dev/ttyUSB0')
            
            board_id = BoardIds.CYTON_BOARD
            if config.get('model') == 'ganglion':
                board_id = BoardIds.GANGLION_BOARD
            
            board = BoardShim(board_id, params)
            board.prepare_session()
            board.start_stream()
            
            # Get board info
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            
            device = NeuralDevice(
                id=f"openbci_{config.get('serial_port', 'default')}",
                name=f"OpenBCI {config.get('model', 'Cyton')}",
                device_type=NeuralDeviceType.EEG,
                channels=[f"ch{i}" for i in range(len(eeg_channels))],
                sampling_rate=sampling_rate,
                resolution_bits=24,
                connection_type='serial',
                is_connected=True,
                metadata={'board_id': board_id}
            )
            
            self.devices[device.id] = device
            self.brainflow_boards[device.id] = board
            
            logger.info(f"Connected to OpenBCI device: {device.name}")
            return device
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenBCI: {e}")
            neural_errors.labels('connection').inc()
            return None
    
    async def _connect_muse(
        self,
        config: Dict[str, Any]
    ) -> Optional[NeuralDevice]:
        """Connect to Muse device"""
        
        try:
            # Muse uses Bluetooth
            params = BrainFlowInputParams()
            params.mac_address = config.get('mac_address', '')
            
            board_id = BoardIds.MUSE_S_BOARD
            if config.get('model') == 'muse2':
                board_id = BoardIds.MUSE_2_BOARD
            
            board = BoardShim(board_id, params)
            board.prepare_session()
            board.start_stream()
            
            device = NeuralDevice(
                id=f"muse_{config.get('mac_address', 'default')}",
                name=f"Muse {config.get('model', 'S')}",
                device_type=NeuralDeviceType.EEG,
                channels=['TP9', 'AF7', 'AF8', 'TP10'],  # Muse electrode positions
                sampling_rate=BoardShim.get_sampling_rate(board_id),
                resolution_bits=12,
                connection_type='bluetooth',
                is_connected=True,
                metadata={'board_id': board_id}
            )
            
            self.devices[device.id] = device
            self.brainflow_boards[device.id] = board
            
            logger.info(f"Connected to Muse device: {device.name}")
            return device
            
        except Exception as e:
            logger.error(f"Failed to connect to Muse: {e}")
            return None
    
    async def _connect_emotiv(
        self,
        config: Dict[str, Any]
    ) -> Optional[NeuralDevice]:
        """Connect to Emotiv device"""
        
        # Emotiv devices require their SDK
        # This is a placeholder
        logger.warning("Emotiv connection not implemented")
        return None
    
    async def _connect_neurosky(
        self,
        config: Dict[str, Any]
    ) -> Optional[NeuralDevice]:
        """Connect to NeuroSky device"""
        
        # NeuroSky MindWave
        # This is a placeholder
        logger.warning("NeuroSky connection not implemented")
        return None
    
    async def _connect_lsl(
        self,
        config: Dict[str, Any]
    ) -> Optional[NeuralDevice]:
        """Connect to LSL stream"""
        
        try:
            # Look for EEG streams
            streams = pylsl.resolve_stream('type', 'EEG')
            
            if not streams:
                logger.warning("No LSL EEG streams found")
                return None
            
            # Connect to first stream
            inlet = StreamInlet(streams[0])
            info = inlet.info()
            
            device = NeuralDevice(
                id=f"lsl_{info.name()}",
                name=f"LSL Stream: {info.name()}",
                device_type=NeuralDeviceType.EEG,
                channels=[
                    info.desc().child("channels").child(f"channel{i}").child_value("label")
                    for i in range(info.channel_count())
                ],
                sampling_rate=info.nominal_srate(),
                resolution_bits=32,  # LSL uses float32
                connection_type='network',
                is_connected=True,
                metadata={'stream_type': info.type()}
            )
            
            self.devices[device.id] = device
            self.active_streams[device.id] = inlet
            
            logger.info(f"Connected to LSL stream: {device.name}")
            return device
            
        except Exception as e:
            logger.error(f"Failed to connect to LSL: {e}")
            return None
    
    async def read_data(
        self,
        device_id: str,
        duration: float = 1.0
    ) -> Optional[NeuralSignal]:
        """Read data from device"""
        
        device = self.devices.get(device_id)
        if not device:
            return None
        
        try:
            if device_id in self.brainflow_boards:
                # Read from BrainFlow
                board = self.brainflow_boards[device_id]
                
                # Get data
                num_samples = int(duration * device.sampling_rate)
                data = board.get_board_data(num_samples)
                
                if data.shape[1] == 0:
                    return None
                
                # Extract EEG channels
                eeg_channels = BoardShim.get_eeg_channels(
                    device.metadata['board_id']
                )
                
                channels = {}
                for i, ch_idx in enumerate(eeg_channels):
                    if i < len(device.channels):
                        channels[device.channels[i]] = data[ch_idx]
                
                # Create signal
                signal = NeuralSignal(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    device_id=device_id,
                    signal_type=SignalType.RAW,
                    channels=channels,
                    sampling_rate=device.sampling_rate,
                    quality_score=1.0
                )
                
                # Process signal
                processor = SignalProcessor()
                for ch_name, ch_data in signal.channels.items():
                    # Preprocess
                    processed = processor.preprocess_signal(
                        ch_data,
                        device.sampling_rate
                    )
                    signal.channels[ch_name] = processed
                    
                    # Check quality
                    _, quality = processor.detect_artifacts(
                        processed,
                        device.sampling_rate
                    )
                    signal.quality_score = min(signal.quality_score, quality)
                
                # Update metrics
                neural_signals_processed.labels(
                    signal.signal_type.value,
                    device.device_type.value
                ).inc()
                
                neural_signal_quality.labels(
                    device_id,
                    'overall'
                ).set(signal.quality_score)
                
                return signal
            
            elif device_id in self.active_streams:
                # Read from LSL
                inlet = self.active_streams[device_id]
                
                samples = []
                timestamps = []
                
                # Pull chunks
                while len(samples) < duration * device.sampling_rate:
                    chunk, ts = inlet.pull_chunk()
                    if chunk:
                        samples.extend(chunk)
                        timestamps.extend(ts)
                    else:
                        await asyncio.sleep(0.01)
                
                if not samples:
                    return None
                
                # Convert to numpy array
                data = np.array(samples).T
                
                channels = {}
                for i, ch_name in enumerate(device.channels):
                    if i < data.shape[0]:
                        channels[ch_name] = data[i]
                
                signal = NeuralSignal(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    device_id=device_id,
                    signal_type=SignalType.RAW,
                    channels=channels,
                    sampling_rate=device.sampling_rate,
                    quality_score=1.0
                )
                
                return signal
            
        except Exception as e:
            logger.error(f"Failed to read data from device {device_id}: {e}")
            neural_errors.labels('read').inc()
            return None
    
    async def disconnect_device(self, device_id: str):
        """Disconnect device"""
        
        if device_id in self.brainflow_boards:
            board = self.brainflow_boards[device_id]
            board.stop_stream()
            board.release_session()
            del self.brainflow_boards[device_id]
        
        if device_id in self.active_streams:
            # LSL streams close automatically
            del self.active_streams[device_id]
        
        if device_id in self.devices:
            device = self.devices[device_id]
            device.is_connected = False
            del self.devices[device_id]
        
        logger.info(f"Disconnected device: {device_id}")


class NeuralInterfaceManager:
    """Manage neural interface system"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        self.device_interface = NeuralDeviceInterface()
        self.signal_processor = SignalProcessor()
        self.pattern_recognizer = PatternRecognizer()
        self.state_analyzer = BrainStateAnalyzer()
        self.command_decoder = NeuralCommandDecoder()
        
        self.signal_buffer = defaultdict(deque)
        self.command_queue = asyncio.Queue()
        self.is_running = False
        
        # Calibration data
        self.calibration_data = defaultdict(list)
        self.user_profile = {}
    
    async def start(self):
        """Start neural interface manager"""
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._signal_acquisition_loop())
        asyncio.create_task(self._signal_processing_loop())
        asyncio.create_task(self._command_execution_loop())
        asyncio.create_task(self._state_monitoring_loop())
        
        logger.info("Neural interface manager started")
    
    async def stop(self):
        """Stop neural interface manager"""
        
        self.is_running = False
        
        # Disconnect all devices
        for device_id in list(self.device_interface.devices.keys()):
            await self.device_interface.disconnect_device(device_id)
        
        logger.info("Neural interface manager stopped")
    
    async def connect_device(
        self,
        device_config: Dict[str, Any]
    ) -> Optional[str]:
        """Connect neural device"""
        
        device = await self.device_interface.connect_device(device_config)
        
        if device:
            return device.id
        
        return None
    
    async def calibrate_user(
        self,
        device_id: str,
        calibration_type: str = 'motor_imagery'
    ) -> bool:
        """Calibrate system for user"""
        
        logger.info(f"Starting {calibration_type} calibration")
        
        if calibration_type == 'motor_imagery':
            return await self._calibrate_motor_imagery(device_id)
        elif calibration_type == 'p300':
            return await self._calibrate_p300(device_id)
        elif calibration_type == 'baseline':
            return await self._calibrate_baseline(device_id)
        else:
            logger.error(f"Unknown calibration type: {calibration_type}")
            return False
    
    async def _calibrate_motor_imagery(
        self,
        device_id: str
    ) -> bool:
        """Calibrate motor imagery"""
        
        # Collect training data
        training_data = []
        
        tasks = ['think_left', 'think_right', 'think_up', 'think_down', 'rest']
        trials_per_task = 20
        trial_duration = 4.0  # seconds
        
        for task in tasks:
            logger.info(f"Calibration task: {task}")
            
            for trial in range(trials_per_task):
                # Show instruction (would display on screen)
                logger.info(f"Trial {trial + 1}/{trials_per_task}: {task}")
                
                # Wait for user ready
                await asyncio.sleep(2.0)
                
                # Record signal
                signal = await self.device_interface.read_data(
                    device_id,
                    trial_duration
                )
                
                if signal:
                    training_data.append((signal, task))
                
                # Inter-trial interval
                await asyncio.sleep(1.0)
        
        # Train classifier
        if training_data:
            self.pattern_recognizer.train_motor_imagery_classifier(training_data)
            self.calibration_data['motor_imagery'] = training_data
            
            logger.info("Motor imagery calibration complete")
            return True
        
        return False
    
    async def _calibrate_p300(
        self,
        device_id: str
    ) -> bool:
        """Calibrate P300 responses"""
        
        # Similar structure to motor imagery
        # Would present visual stimuli and record responses
        logger.info("P300 calibration not implemented")
        return False
    
    async def _calibrate_baseline(
        self,
        device_id: str
    ) -> bool:
        """Record baseline brain activity"""
        
        logger.info("Recording baseline activity")
        
        # Record resting state
        baseline_duration = 60.0  # 1 minute
        
        signal = await self.device_interface.read_data(
            device_id,
            baseline_duration
        )
        
        if signal:
            # Analyze baseline
            features = self.signal_processor.extract_features(signal)
            
            self.user_profile['baseline'] = {
                'band_powers': features['frequency_domain'],
                'recorded_at': datetime.utcnow().isoformat()
            }
            
            # Set baseline for state analyzer
            state = self.state_analyzer.analyze_state(signal, self.pattern_recognizer)
            self.state_analyzer.baseline_state = state
            
            logger.info("Baseline calibration complete")
            return True
        
        return False
    
    async def _signal_acquisition_loop(self):
        """Continuously acquire signals from devices"""
        
        while self.is_running:
            try:
                # Read from all connected devices
                for device_id in list(self.device_interface.devices.keys()):
                    if self.device_interface.devices[device_id].is_connected:
                        # Read 0.1 second chunks
                        signal = await self.device_interface.read_data(
                            device_id,
                            0.1
                        )
                        
                        if signal:
                            # Add to buffer
                            self.signal_buffer[device_id].append(signal)
                            
                            # Limit buffer size
                            if len(self.signal_buffer[device_id]) > 100:
                                self.signal_buffer[device_id].popleft()
                
                await asyncio.sleep(0.01)  # 100 Hz update rate
                
            except Exception as e:
                logger.error(f"Signal acquisition error: {e}")
                await asyncio.sleep(1.0)
    
    async def _signal_processing_loop(self):
        """Process signals and decode commands"""
        
        while self.is_running:
            try:
                # Process signals from each device
                for device_id, signal_queue in self.signal_buffer.items():
                    if len(signal_queue) >= 10:  # Need at least 1 second of data
                        # Get recent signals
                        recent_signals = list(signal_queue)[-10:]
                        
                        # Combine into longer signal
                        combined = self._combine_signals(recent_signals)
                        
                        if combined:
                            # Analyze brain state
                            brain_state = self.state_analyzer.analyze_state(
                                combined,
                                self.pattern_recognizer
                            )
                            
                            # Decode intent
                            context = self._get_context()
                            intent = self.command_decoder.decode_intent(
                                combined,
                                brain_state,
                                context
                            )
                            
                            if intent:
                                await self.command_queue.put(intent)
                                
                                # Store in database
                                await self._store_intent(intent)
                
                await asyncio.sleep(0.1)  # 10 Hz processing
                
            except Exception as e:
                logger.error(f"Signal processing error: {e}")
                neural_errors.labels('processing').inc()
                await asyncio.sleep(1.0)
    
    def _combine_signals(
        self,
        signals: List[NeuralSignal]
    ) -> Optional[NeuralSignal]:
        """Combine multiple signal chunks"""
        
        if not signals:
            return None
        
        # Combine channel data
        combined_channels = defaultdict(list)
        
        for signal in signals:
            for ch_name, ch_data in signal.channels.items():
                combined_channels[ch_name].append(ch_data)
        
        # Concatenate
        merged_channels = {}
        for ch_name, ch_list in combined_channels.items():
            merged_channels[ch_name] = np.concatenate(ch_list)
        
        # Create combined signal
        combined = NeuralSignal(
            id=str(uuid.uuid4()),
            timestamp=signals[-1].timestamp,
            device_id=signals[0].device_id,
            signal_type=SignalType.RAW,
            channels=merged_channels,
            sampling_rate=signals[0].sampling_rate,
            quality_score=np.mean([s.quality_score for s in signals])
        )
        
        return combined
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current context for decoding"""
        
        # In a real system, this would track UI state,
        # active applications, user preferences, etc.
        
        return {
            'ui_mode': 'cursor_control',
            'active_window': 'neural_interface',
            'p300_mode': False,
            'focused_element': None,
            'options': []
        }
    
    async def _command_execution_loop(self):
        """Execute decoded commands"""
        
        while self.is_running:
            try:
                # Get command from queue
                intent = await asyncio.wait_for(
                    self.command_queue.get(),
                    timeout=1.0
                )
                
                # Execute command
                success = await self._execute_command(intent)
                
                # Update metrics
                neural_commands_executed.labels(
                    intent.command.value,
                    'success' if success else 'failed'
                ).inc()
                
                # Update database
                await self._update_intent_status(intent.id, success)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Command execution error: {e}")
                neural_errors.labels('execution').inc()
    
    async def _execute_command(
        self,
        intent: NeuralIntent
    ) -> bool:
        """Execute neural command"""
        
        try:
            logger.info(
                f"Executing command: {intent.command.value} "
                f"with confidence {intent.confidence:.2f}"
            )
            
            if intent.command == NeuralCommand.CURSOR_MOVE:
                # Move cursor
                direction = intent.parameters.get('direction')
                distance = intent.parameters.get('distance', 10)
                
                # Would interface with OS here
                logger.info(f"Moving cursor {direction} by {distance} pixels")
                return True
            
            elif intent.command == NeuralCommand.CLICK:
                # Click
                element = intent.parameters.get('element')
                logger.info(f"Clicking on {element}")
                return True
            
            elif intent.command == NeuralCommand.TYPE_TEXT:
                # Type text
                text = intent.parameters.get('text', '')
                logger.info(f"Typing: {text}")
                return True
            
            elif intent.command == NeuralCommand.QUERY_DATA:
                # Query MCP data
                query = intent.parameters.get('query')
                logger.info(f"Querying: {query}")
                
                # Would interface with MCP API
                return True
            
            elif intent.command == NeuralCommand.EMERGENCY_ALERT:
                # Emergency alert
                logger.warning("EMERGENCY ALERT activated via neural interface")
                
                # Would trigger emergency protocols
                return True
            
            else:
                logger.warning(f"Unknown command: {intent.command}")
                return False
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False
    
    async def _state_monitoring_loop(self):
        """Monitor brain state changes"""
        
        while self.is_running:
            try:
                # Check recent brain states
                if len(self.state_analyzer.history) > 0:
                    current_state = self.state_analyzer.history[-1]
                    
                    # Detect significant changes
                    change = self.state_analyzer.detect_state_change(current_state)
                    
                    if change:
                        logger.info(f"Brain state change: {change}")
                        
                        # Could trigger adaptive UI changes
                        # or send notifications
                    
                    # Check for fatigue
                    if current_state.fatigue_level > 0.8:
                        logger.warning("High fatigue detected - suggesting break")
                    
                    # Check for stress
                    if current_state.stress_level > 0.8:
                        logger.warning("High stress detected")
                    
                    # Monitor cognitive load
                    cognitive_load = self.state_analyzer.get_cognitive_load()
                    if cognitive_load > 0.9:
                        logger.warning("Cognitive overload detected")
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"State monitoring error: {e}")
                await asyncio.sleep(10.0)
    
    async def _store_intent(self, intent: NeuralIntent):
        """Store intent in database"""
        
        session = self.Session()
        try:
            db_intent = NeuralIntentDB(
                id=intent.id,
                timestamp=intent.timestamp,
                command=intent.command.value,
                parameters=intent.parameters,
                confidence=intent.confidence,
                processing_time=intent.processing_time
            )
            
            session.add(db_intent)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store intent: {e}")
        finally:
            session.close()
    
    async def _update_intent_status(
        self,
        intent_id: str,
        success: bool
    ):
        """Update intent execution status"""
        
        session = self.Session()
        try:
            intent = session.query(NeuralIntentDB).filter_by(
                id=intent_id
            ).first()
            
            if intent:
                intent.executed = True
                intent.success = success
                session.commit()
                
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update intent status: {e}")
        finally:
            session.close()
    
    def visualize_brain_activity(
        self,
        device_id: str,
        duration: float = 10.0
    ) -> str:
        """Create brain activity visualization"""
        
        if device_id not in self.signal_buffer:
            return ""
        
        # Get recent signals
        signals = list(self.signal_buffer[device_id])
        if not signals:
            return ""
        
        # Create plotly figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Raw EEG', 'Frequency Spectrum',
                'Band Powers', 'Brain State',
                'Topographic Map', 'Connectivity'
            )
        )
        
        # Get latest signal
        latest_signal = signals[-1]
        
        # Plot raw EEG
        for i, (ch_name, ch_data) in enumerate(latest_signal.channels.items()):
            if i < 4:  # Limit to 4 channels
                time = np.arange(len(ch_data)) / latest_signal.sampling_rate
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=ch_data,
                        name=ch_name,
                        mode='lines'
                    ),
                    row=1, col=1
                )
        
        # Frequency spectrum
        features = self.signal_processor.extract_features(latest_signal)
        
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_powers = [
            features['frequency_domain'].get(f'{band}_relative', 0)
            for band in bands
        ]
        
        fig.add_trace(
            go.Bar(x=bands, y=band_powers, name='Band Power'),
            row=1, col=2
        )
        
        # Brain state timeline
        if self.state_analyzer.history:
            states = list(self.state_analyzer.history)[-50:]  # Last 50 states
            
            times = [(s.timestamp - states[0].timestamp).total_seconds() for s in states]
            attention = [s.attention_level for s in states]
            stress = [s.stress_level for s in states]
            fatigue = [s.fatigue_level for s in states]
            
            fig.add_trace(
                go.Scatter(x=times, y=attention, name='Attention', mode='lines'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=times, y=stress, name='Stress', mode='lines'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=times, y=fatigue, name='Fatigue', mode='lines'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Neural Interface - Brain Activity Monitor',
            height=1000,
            showlegend=True
        )
        
        # Convert to HTML
        return fig.to_html()
    
    async def enable_thought_typing(
        self,
        device_id: str
    ) -> bool:
        """Enable thought-based typing"""
        
        # This would implement a P300 speller or similar
        logger.info("Thought typing mode activated")
        
        # Update context
        self._get_context()['p300_mode'] = True
        
        return True
    
    async def query_with_thought(
        self,
        device_id: str,
        options: List[str]
    ) -> Optional[str]:
        """Select from options using thoughts"""
        
        # Present options and detect selection
        context = self._get_context()
        context['options'] = options
        
        # Wait for selection
        timeout = 30.0
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            try:
                intent = await asyncio.wait_for(
                    self.command_queue.get(),
                    timeout=1.0
                )
                
                if intent.command == NeuralCommand.SELECT_OPTION:
                    selected = intent.parameters.get('option')
                    if selected in options:
                        return selected
                
            except asyncio.TimeoutError:
                continue
        
        return None


# Example usage
async def neural_demo():
    """Demo neural interface system"""
    
    # Initialize manager
    manager = NeuralInterfaceManager(
        'postgresql://user:pass@localhost/neural_db'
    )
    await manager.start()
    
    # Connect to simulated device
    device_config = {
        'type': 'openbci',
        'model': 'cyton',
        'serial_port': '/dev/ttyUSB0'
    }
    
    # In practice, would connect to real device
    # For demo, create simulated device
    sim_device = NeuralDevice(
        id='sim_eeg',
        name='Simulated EEG',
        device_type=NeuralDeviceType.EEG,
        channels=['Fz', 'C3', 'C4', 'Pz'],
        sampling_rate=250.0,
        resolution_bits=24,
        connection_type='simulated',
        is_connected=True
    )
    
    manager.device_interface.devices[sim_device.id] = sim_device
    
    print(f"Connected device: {sim_device.name}")
    
    # Simulate some signals
    print("\nSimulating neural signals...")
    
    for i in range(10):
        # Generate synthetic EEG
        duration = 1.0
        samples = int(duration * sim_device.sampling_rate)
        t = np.arange(samples) / sim_device.sampling_rate
        
        # Mix of frequencies
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
        noise = 0.2 * np.random.randn(samples)
        
        signal_data = alpha + beta + noise
        
        # Create signal
        signal = NeuralSignal(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            device_id=sim_device.id,
            signal_type=SignalType.RAW,
            channels={
                'Fz': signal_data + 0.1 * np.random.randn(samples),
                'C3': signal_data + 0.1 * np.random.randn(samples),
                'C4': signal_data + 0.1 * np.random.randn(samples),
                'Pz': signal_data + 0.1 * np.random.randn(samples)
            },
            sampling_rate=sim_device.sampling_rate,
            quality_score=0.9
        )
        
        # Add to buffer
        manager.signal_buffer[sim_device.id].append(signal)
        
        # Process
        if len(manager.signal_buffer[sim_device.id]) >= 5:
            # Analyze state
            combined = manager._combine_signals(
                list(manager.signal_buffer[sim_device.id])[-5:]
            )
            
            brain_state = manager.state_analyzer.analyze_state(
                combined,
                manager.pattern_recognizer
            )
            
            print(f"\nBrain State Analysis:")
            print(f"  Cognitive State: {brain_state.cognitive_state.value}")
            print(f"  Attention: {brain_state.attention_level:.2f}")
            print(f"  Stress: {brain_state.stress_level:.2f}")
            print(f"  Fatigue: {brain_state.fatigue_level:.2f}")
            print(f"  Valence: {brain_state.valence:.2f}")
            print(f"  Arousal: {brain_state.arousal:.2f}")
            
            # Extract features
            features = manager.signal_processor.extract_features(combined)
            
            print(f"\nSignal Features:")
            print(f"  Peak Frequency: {features['frequency_domain']['peak_frequency']:.1f} Hz")
            print(f"  Alpha Power: {features['frequency_domain']['alpha_relative']:.2f}")
            print(f"  Beta Power: {features['frequency_domain']['beta_relative']:.2f}")
            
            # Simulate intent detection
            if i == 5:
                # Simulate detected intent
                intent = NeuralIntent(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    command=NeuralCommand.CURSOR_MOVE,
                    parameters={'direction': 'up', 'distance': 20},
                    confidence=0.85,
                    source_signals=[combined.id],
                    processing_time=0.05
                )
                
                await manager.command_queue.put(intent)
                print(f"\nNeural Intent Detected:")
                print(f"  Command: {intent.command.value}")
                print(f"  Parameters: {intent.parameters}")
                print(f"  Confidence: {intent.confidence:.2f}")
        
        await asyncio.sleep(1.0)
    
    # Test thought-based selection
    print("\n\nThought-Based Selection Demo:")
    options = ['Red', 'Green', 'Blue', 'Yellow']
    print(f"Options: {options}")
    print("Think about your choice...")
    
    # Simulate P300 selection
    await asyncio.sleep(2.0)
    selected = options[1]  # Simulate selecting "Green"
    print(f"Selected via neural interface: {selected}")
    
    # Generate visualization
    print("\nGenerating brain activity visualization...")
    html = manager.visualize_brain_activity(sim_device.id)
    
    if html:
        with open('brain_activity.html', 'w') as f:
            f.write(html)
        print("Visualization saved to brain_activity.html")
    
    # Demonstrate calibration
    print("\n\nCalibration Demo:")
    print("Starting baseline calibration...")
    
    # Simulate baseline recording
    baseline_signal = NeuralSignal(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        device_id=sim_device.id,
        signal_type=SignalType.RAW,
        channels={
            ch: np.random.randn(15000)  # 60 seconds at 250 Hz
            for ch in sim_device.channels
        },
        sampling_rate=sim_device.sampling_rate,
        quality_score=0.95
    )
    
    features = manager.signal_processor.extract_features(baseline_signal)
    
    print("Baseline recorded:")
    print(f"  Average Alpha: {features['frequency_domain']['alpha_relative']:.2f}")
    print(f"  Average Beta: {features['frequency_domain']['beta_relative']:.2f}")
    
    # Stop manager
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(neural_demo())