"""
Satellite Communication System - 40by6
Enable global connectivity for MCP Stack via satellite networks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import math
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import struct
import socket
import ssl
import serial
import usb.core
import usb.util
from skyfield.api import load, Topos, EarthSatellite
from skyfield.timelib import Time
from skyfield.units import Angle
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time as AstropyTime
import ephem
import orbital
from orbital import earth, KeplerianElements
import sgp4
from sgp4.api import Satrec, WGS72
import gnuradio
from gnuradio import gr, blocks, analog, digital, channels
from gnuradio import fft, filter as gr_filter
import osmosdr
import rtlsdr
import limesdr
import uhd
import iridium
import inmarsat
import thuraya
import globalstar
import orbcomm
import swarm
import starlink
from spacex.starlink import StarlinkAPI
import oneweb
import kuiper
import telesat
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, JSON, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge, Summary
import paho.mqtt.client as mqtt
import aiohttp
import websockets
import grpc
import msgpack
import cbor2
import protobuf
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import hashlib
import secrets
import cv2
import pyaudio
import wave
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter, spectrogram
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import networkx as nx
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import folium
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import xarray as xr
import cartopy
import cartopy.crs as ccrs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ray
from ray import serve
import tensorflow as tf
import torch
import onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Metrics
sat_connections_active = Gauge('satellite_connections_active', 'Active satellite connections', ['constellation', 'band'])
sat_messages_sent = Counter('satellite_messages_sent_total', 'Total satellite messages sent', ['constellation', 'message_type'])
sat_messages_received = Counter('satellite_messages_received_total', 'Total satellite messages received', ['constellation', 'message_type'])
sat_data_transferred = Counter('satellite_data_transferred_bytes', 'Satellite data transferred', ['direction', 'constellation'])
sat_link_quality = Gauge('satellite_link_quality_db', 'Satellite link quality in dB', ['satellite_id', 'metric'])
sat_latency = Histogram('satellite_latency_seconds', 'Satellite communication latency', ['constellation', 'service'])
sat_errors = Counter('satellite_errors_total', 'Total satellite communication errors', ['error_type', 'constellation'])

Base = declarative_base()


class SatelliteConstellation(Enum):
    """Satellite constellations"""
    STARLINK = "starlink"
    ONEWEB = "oneweb"
    KUIPER = "kuiper"
    IRIDIUM = "iridium"
    GLOBALSTAR = "globalstar"
    INMARSAT = "inmarsat"
    THURAYA = "thuraya"
    ORBCOMM = "orbcomm"
    SWARM = "swarm"
    TELESAT = "telesat"
    GPS = "gps"
    GALILEO = "galileo"
    GLONASS = "glonass"
    BEIDOU = "beidou"
    CUSTOM = "custom"


class FrequencyBand(Enum):
    """Satellite frequency bands"""
    L_BAND = "L"  # 1-2 GHz
    S_BAND = "S"  # 2-4 GHz
    C_BAND = "C"  # 4-8 GHz
    X_BAND = "X"  # 8-12 GHz
    KU_BAND = "Ku"  # 12-18 GHz
    K_BAND = "K"  # 18-27 GHz
    KA_BAND = "Ka"  # 27-40 GHz
    V_BAND = "V"  # 40-75 GHz
    W_BAND = "W"  # 75-110 GHz
    Q_BAND = "Q"  # 33-50 GHz
    U_BAND = "U"  # 40-60 GHz


class ModulationType(Enum):
    """Modulation types"""
    BPSK = "bpsk"
    QPSK = "qpsk"
    QAM16 = "16qam"
    QAM64 = "64qam"
    QAM256 = "256qam"
    FSK = "fsk"
    MSK = "msk"
    OFDM = "ofdm"
    CDMA = "cdma"
    TDMA = "tdma"
    FDMA = "fdma"


class ServiceType(Enum):
    """Satellite service types"""
    DATA = "data"
    VOICE = "voice"
    VIDEO = "video"
    IOT = "iot"
    EMERGENCY = "emergency"
    BROADCAST = "broadcast"
    NAVIGATION = "navigation"
    TIMING = "timing"
    SENSING = "sensing"


@dataclass
class Satellite:
    """Satellite representation"""
    id: str
    name: str
    constellation: SatelliteConstellation
    tle_line1: str
    tle_line2: str
    frequency_bands: List[FrequencyBand]
    services: List[ServiceType]
    bandwidth_mbps: float
    power_watts: float
    antenna_gain_dbi: float
    noise_figure_db: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'constellation': self.constellation.value,
            'tle': [self.tle_line1, self.tle_line2],
            'frequency_bands': [b.value for b in self.frequency_bands],
            'services': [s.value for s in self.services],
            'bandwidth_mbps': self.bandwidth_mbps,
            'power_watts': self.power_watts,
            'antenna_gain_dbi': self.antenna_gain_dbi,
            'noise_figure_db': self.noise_figure_db,
            'metadata': self.metadata
        }


@dataclass
class GroundStation:
    """Ground station representation"""
    id: str
    name: str
    location: EarthLocation
    antenna_diameter_m: float
    antenna_gain_dbi: float
    transmit_power_watts: float
    noise_temperature_k: float
    frequency_bands: List[FrequencyBand]
    elevation_mask_deg: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'location': {
                'lat': self.location.lat.deg,
                'lon': self.location.lon.deg,
                'alt': self.location.height.value
            },
            'antenna_diameter_m': self.antenna_diameter_m,
            'antenna_gain_dbi': self.antenna_gain_dbi,
            'transmit_power_watts': self.transmit_power_watts,
            'noise_temperature_k': self.noise_temperature_k,
            'frequency_bands': [b.value for b in self.frequency_bands],
            'elevation_mask_deg': self.elevation_mask_deg,
            'metadata': self.metadata
        }


@dataclass
class SatelliteLink:
    """Satellite communication link"""
    satellite: Satellite
    ground_station: GroundStation
    timestamp: datetime
    azimuth_deg: float
    elevation_deg: float
    range_km: float
    doppler_shift_hz: float
    path_loss_db: float
    link_margin_db: float
    data_rate_mbps: float
    ber: float  # Bit error rate
    is_visible: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'satellite_id': self.satellite.id,
            'ground_station_id': self.ground_station.id,
            'timestamp': self.timestamp.isoformat(),
            'azimuth_deg': self.azimuth_deg,
            'elevation_deg': self.elevation_deg,
            'range_km': self.range_km,
            'doppler_shift_hz': self.doppler_shift_hz,
            'path_loss_db': self.path_loss_db,
            'link_margin_db': self.link_margin_db,
            'data_rate_mbps': self.data_rate_mbps,
            'ber': self.ber,
            'is_visible': self.is_visible
        }


@dataclass
class SatelliteMessage:
    """Satellite message"""
    id: str
    timestamp: datetime
    source: str
    destination: str
    message_type: str
    payload: bytes
    constellation: SatelliteConstellation
    satellite_id: Optional[str] = None
    frequency_hz: Optional[float] = None
    modulation: Optional[ModulationType] = None
    encryption: bool = True
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


class SatelliteTracker:
    """Track satellite positions and visibility"""
    
    def __init__(self):
        self.ts = load.timescale()
        self.satellites: Dict[str, EarthSatellite] = {}
        self.orbital_models: Dict[str, Satrec] = {}
    
    def add_satellite(self, satellite: Satellite):
        """Add satellite to tracker"""
        
        # Create Skyfield satellite
        sat_skyfield = EarthSatellite(
            satellite.tle_line1,
            satellite.tle_line2,
            satellite.name,
            self.ts
        )
        self.satellites[satellite.id] = sat_skyfield
        
        # Create SGP4 model
        sat_sgp4 = Satrec.twoline2rv(
            satellite.tle_line1,
            satellite.tle_line2,
            WGS72
        )
        self.orbital_models[satellite.id] = sat_sgp4
    
    def get_position(
        self,
        satellite_id: str,
        timestamp: Optional[datetime] = None
    ) -> Tuple[float, float, float]:
        """Get satellite position (lat, lon, alt_km)"""
        
        if satellite_id not in self.satellites:
            raise ValueError(f"Unknown satellite: {satellite_id}")
        
        sat = self.satellites[satellite_id]
        
        # Get time
        if timestamp:
            t = self.ts.from_datetime(timestamp)
        else:
            t = self.ts.now()
        
        # Get position
        geocentric = sat.at(t)
        subpoint = geocentric.subpoint()
        
        return (
            subpoint.latitude.degrees,
            subpoint.longitude.degrees,
            subpoint.elevation.km
        )
    
    def get_look_angles(
        self,
        satellite_id: str,
        ground_station: GroundStation,
        timestamp: Optional[datetime] = None
    ) -> Tuple[float, float, float, bool]:
        """Get azimuth, elevation, range, and visibility"""
        
        if satellite_id not in self.satellites:
            raise ValueError(f"Unknown satellite: {satellite_id}")
        
        sat = self.satellites[satellite_id]
        
        # Get time
        if timestamp:
            t = self.ts.from_datetime(timestamp)
        else:
            t = self.ts.now()
        
        # Create observer
        observer = Topos(
            latitude_degrees=ground_station.location.lat.deg,
            longitude_degrees=ground_station.location.lon.deg,
            elevation_m=ground_station.location.height.value
        )
        
        # Calculate position
        difference = sat - observer
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        
        # Check visibility
        is_visible = alt.degrees > ground_station.elevation_mask_deg
        
        return (
            az.degrees,
            alt.degrees,
            distance.km,
            is_visible
        )
    
    def calculate_passes(
        self,
        satellite_id: str,
        ground_station: GroundStation,
        start_time: datetime,
        duration_hours: float = 24
    ) -> List[Dict[str, Any]]:
        """Calculate satellite passes over ground station"""
        
        if satellite_id not in self.satellites:
            raise ValueError(f"Unknown satellite: {satellite_id}")
        
        sat = self.satellites[satellite_id]
        
        # Create observer
        observer = Topos(
            latitude_degrees=ground_station.location.lat.deg,
            longitude_degrees=ground_station.location.lon.deg,
            elevation_m=ground_station.location.height.value
        )
        
        # Time range
        t0 = self.ts.from_datetime(start_time)
        t1 = self.ts.from_datetime(start_time + timedelta(hours=duration_hours))
        
        # Find events
        t, events = sat.find_events(observer, t0, t1, altitude_degrees=ground_station.elevation_mask_deg)
        
        # Parse passes
        passes = []
        i = 0
        while i < len(events):
            if events[i] == 0:  # Rise
                rise_time = t[i]
                
                # Find culmination and set
                culmination_time = None
                set_time = None
                
                for j in range(i + 1, len(events)):
                    if events[j] == 1:  # Culmination
                        culmination_time = t[j]
                    elif events[j] == 2:  # Set
                        set_time = t[j]
                        break
                
                if rise_time and set_time:
                    # Calculate max elevation
                    if culmination_time:
                        difference = sat - observer
                        topocentric = difference.at(culmination_time)
                        alt, az, distance = topocentric.altaz()
                        max_elevation = alt.degrees
                    else:
                        max_elevation = ground_station.elevation_mask_deg
                    
                    passes.append({
                        'rise_time': rise_time.utc_datetime(),
                        'set_time': set_time.utc_datetime(),
                        'duration_seconds': (set_time - rise_time) * 86400,
                        'max_elevation_deg': max_elevation
                    })
                
                i = j + 1 if set_time else i + 1
            else:
                i += 1
        
        return passes
    
    def get_doppler_shift(
        self,
        satellite_id: str,
        ground_station: GroundStation,
        frequency_hz: float,
        timestamp: Optional[datetime] = None
    ) -> float:
        """Calculate Doppler shift"""
        
        if satellite_id not in self.orbital_models:
            raise ValueError(f"Unknown satellite: {satellite_id}")
        
        # Get satellite velocity
        if timestamp:
            jd = self.ts.from_datetime(timestamp).tt
        else:
            jd = self.ts.now().tt
        
        sat_sgp4 = self.orbital_models[satellite_id]
        
        # Get position and velocity
        e, r, v = sat_sgp4.sgp4(jd, 0.0)
        
        if e != 0:
            return 0.0
        
        # Calculate radial velocity component
        # Simplified - in production would be more accurate
        radial_velocity_kmps = np.linalg.norm(v) * 0.1  # Rough estimate
        
        # Doppler shift
        c = 299792.458  # Speed of light km/s
        doppler_shift = frequency_hz * (radial_velocity_kmps / c)
        
        return doppler_shift
    
    def predict_coverage(
        self,
        satellite_ids: List[str],
        region: Polygon,
        timestamp: Optional[datetime] = None,
        min_elevation_deg: float = 10.0
    ) -> Dict[str, Any]:
        """Predict coverage for a region"""
        
        if timestamp:
            t = self.ts.from_datetime(timestamp)
        else:
            t = self.ts.now()
        
        # Sample points in region
        minx, miny, maxx, maxy = region.bounds
        lat_points = np.linspace(miny, maxy, 20)
        lon_points = np.linspace(minx, maxx, 20)
        
        coverage_map = np.zeros((len(lat_points), len(lon_points)))
        
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                point = Point(lon, lat)
                
                if region.contains(point):
                    # Check visibility from this point
                    observer = Topos(
                        latitude_degrees=lat,
                        longitude_degrees=lon
                    )
                    
                    visible_sats = 0
                    for sat_id in satellite_ids:
                        if sat_id in self.satellites:
                            sat = self.satellites[sat_id]
                            difference = sat - observer
                            topocentric = difference.at(t)
                            alt, _, _ = topocentric.altaz()
                            
                            if alt.degrees > min_elevation_deg:
                                visible_sats += 1
                    
                    coverage_map[i, j] = visible_sats
        
        return {
            'timestamp': timestamp or datetime.utcnow(),
            'coverage_map': coverage_map,
            'lat_grid': lat_points,
            'lon_grid': lon_points,
            'max_satellites': int(np.max(coverage_map)),
            'avg_satellites': float(np.mean(coverage_map[coverage_map > 0]))
        }


class LinkBudgetCalculator:
    """Calculate satellite link budgets"""
    
    @staticmethod
    def calculate_path_loss(frequency_hz: float, distance_km: float) -> float:
        """Calculate free space path loss"""
        
        # FSPL = 20*log10(d) + 20*log10(f) + 32.44
        # d in km, f in MHz
        frequency_mhz = frequency_hz / 1e6
        path_loss_db = 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.44
        
        return path_loss_db
    
    @staticmethod
    def calculate_atmospheric_loss(
        frequency_hz: float,
        elevation_deg: float,
        weather: str = "clear"
    ) -> float:
        """Calculate atmospheric losses"""
        
        # Simplified atmospheric loss model
        frequency_ghz = frequency_hz / 1e9
        
        # Base losses
        if weather == "clear":
            loss_per_km = 0.01 * frequency_ghz  # dB/km
        elif weather == "rain":
            loss_per_km = 0.1 * frequency_ghz
        else:
            loss_per_km = 0.05 * frequency_ghz
        
        # Path through atmosphere
        if elevation_deg > 0:
            path_length_km = 10 / np.sin(np.radians(elevation_deg))
        else:
            path_length_km = 1000  # Horizon
        
        return loss_per_km * path_length_km
    
    @staticmethod
    def calculate_link_margin(
        satellite: Satellite,
        ground_station: GroundStation,
        distance_km: float,
        frequency_hz: float,
        elevation_deg: float,
        required_eb_no_db: float = 10.0,
        weather: str = "clear"
    ) -> Dict[str, float]:
        """Calculate complete link budget"""
        
        # Transmit power
        tx_power_dbw = 10 * np.log10(satellite.power_watts)
        
        # Antenna gains
        tx_gain_dbi = satellite.antenna_gain_dbi
        rx_gain_dbi = ground_station.antenna_gain_dbi
        
        # Losses
        path_loss_db = LinkBudgetCalculator.calculate_path_loss(frequency_hz, distance_km)
        atmospheric_loss_db = LinkBudgetCalculator.calculate_atmospheric_loss(
            frequency_hz, elevation_deg, weather
        )
        pointing_loss_db = 0.5  # Typical pointing loss
        polarization_loss_db = 0.5
        
        # Total losses
        total_loss_db = path_loss_db + atmospheric_loss_db + pointing_loss_db + polarization_loss_db
        
        # Received power
        rx_power_dbw = tx_power_dbw + tx_gain_dbi + rx_gain_dbi - total_loss_db
        
        # Noise calculations
        k_boltzmann_dbw = -228.6  # Boltzmann constant in dBW/K/Hz
        noise_temp_k = ground_station.noise_temperature_k + 290 * (10**(satellite.noise_figure_db/10) - 1)
        noise_temp_db = 10 * np.log10(noise_temp_k)
        
        # G/T (figure of merit)
        g_over_t_db = rx_gain_dbi - noise_temp_db
        
        # C/N0 (carrier to noise density)
        c_over_n0_db = rx_power_dbw - k_boltzmann_dbw - noise_temp_db
        
        # Data rate calculation
        bandwidth_hz = satellite.bandwidth_mbps * 1e6
        data_rate_bps = bandwidth_hz * np.log2(1 + 10**(c_over_n0_db/10))
        data_rate_mbps = data_rate_bps / 1e6
        
        # Link margin
        bandwidth_db = 10 * np.log10(bandwidth_hz)
        required_c_over_n_db = required_eb_no_db + bandwidth_db
        actual_c_over_n_db = c_over_n0_db - bandwidth_db
        link_margin_db = actual_c_over_n_db - required_c_over_n_db
        
        return {
            'tx_power_dbw': tx_power_dbw,
            'tx_gain_dbi': tx_gain_dbi,
            'rx_gain_dbi': rx_gain_dbi,
            'path_loss_db': path_loss_db,
            'atmospheric_loss_db': atmospheric_loss_db,
            'total_loss_db': total_loss_db,
            'rx_power_dbw': rx_power_dbw,
            'noise_temp_k': noise_temp_k,
            'g_over_t_db': g_over_t_db,
            'c_over_n0_db': c_over_n0_db,
            'data_rate_mbps': data_rate_mbps,
            'link_margin_db': link_margin_db
        }


class SatelliteModem:
    """Software-defined satellite modem"""
    
    def __init__(
        self,
        sdr_type: str = "rtlsdr",  # rtlsdr, hackrf, limesdr, usrp
        device_index: int = 0
    ):
        self.sdr_type = sdr_type
        self.device_index = device_index
        self.sdr = None
        self.sample_rate = 2.048e6
        self.center_freq = 1.57542e9  # L-band default
        self._init_sdr()
    
    def _init_sdr(self):
        """Initialize SDR hardware"""
        
        if self.sdr_type == "rtlsdr":
            self.sdr = rtlsdr.RtlSdr(device_index=self.device_index)
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            self.sdr.gain = 'auto'
        
        elif self.sdr_type == "limesdr":
            # LimeSDR initialization
            pass
        
        elif self.sdr_type == "usrp":
            # USRP initialization
            self.sdr = uhd.usrp.MultiUSRP()
            self.sdr.set_rx_rate(self.sample_rate)
            self.sdr.set_rx_freq(self.center_freq)
        
    def receive_samples(self, num_samples: int) -> np.ndarray:
        """Receive IQ samples"""
        
        if self.sdr_type == "rtlsdr":
            return self.sdr.read_samples(num_samples)
        
        elif self.sdr_type == "usrp":
            # USRP receive
            metadata = uhd.types.RXMetadata()
            samples = np.zeros(num_samples, dtype=np.complex64)
            self.sdr.recv_num_samps(
                samples,
                num_samples,
                metadata,
                [0],
                0.1
            )
            return samples
        
        return np.zeros(num_samples, dtype=np.complex64)
    
    def demodulate_bpsk(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate BPSK signal"""
        
        # Carrier recovery
        carrier_freq = self._estimate_carrier_frequency(samples)
        
        # Generate local oscillator
        t = np.arange(len(samples)) / self.sample_rate
        lo = np.exp(-2j * np.pi * carrier_freq * t)
        
        # Mix down to baseband
        baseband = samples * lo
        
        # Low-pass filter
        cutoff = 10e3  # 10 kHz
        b, a = butter(5, cutoff / (self.sample_rate / 2), 'low')
        filtered = lfilter(b, a, baseband)
        
        # Symbol timing recovery
        symbols = self._symbol_timing_recovery(filtered)
        
        # Decision
        bits = (np.real(symbols) > 0).astype(int)
        
        return bits
    
    def demodulate_qpsk(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate QPSK signal"""
        
        # Similar to BPSK but with I/Q decision
        carrier_freq = self._estimate_carrier_frequency(samples)
        
        t = np.arange(len(samples)) / self.sample_rate
        lo = np.exp(-2j * np.pi * carrier_freq * t)
        
        baseband = samples * lo
        
        cutoff = 10e3
        b, a = butter(5, cutoff / (self.sample_rate / 2), 'low')
        filtered = lfilter(b, a, baseband)
        
        symbols = self._symbol_timing_recovery(filtered)
        
        # QPSK decision
        bits = []
        for symbol in symbols:
            i_bit = 1 if np.real(symbol) > 0 else 0
            q_bit = 1 if np.imag(symbol) > 0 else 0
            bits.extend([i_bit, q_bit])
        
        return np.array(bits)
    
    def _estimate_carrier_frequency(self, samples: np.ndarray) -> float:
        """Estimate carrier frequency offset"""
        
        # FFT-based frequency estimation
        fft_size = 8192
        window = np.hanning(fft_size)
        
        fft_result = np.fft.fft(samples[:fft_size] * window)
        freqs = np.fft.fftfreq(fft_size, 1/self.sample_rate)
        
        # Find peak
        peak_idx = np.argmax(np.abs(fft_result))
        carrier_offset = freqs[peak_idx]
        
        return carrier_offset
    
    def _symbol_timing_recovery(
        self,
        samples: np.ndarray,
        samples_per_symbol: int = 8
    ) -> np.ndarray:
        """Gardner timing recovery"""
        
        symbols = []
        mu = 0.0  # Fractional sample offset
        
        for i in range(0, len(samples) - 2 * samples_per_symbol, samples_per_symbol):
            # Interpolate sample at optimal timing
            index = int(i + mu)
            if index + 1 < len(samples):
                # Linear interpolation
                sample = samples[index] * (1 - mu) + samples[index + 1] * mu
                symbols.append(sample)
                
                # Timing error detector (Gardner)
                if index + samples_per_symbol < len(samples):
                    midpoint = samples[index + samples_per_symbol // 2]
                    next_sample = samples[index + samples_per_symbol]
                    
                    error = np.real(np.conj(midpoint) * (next_sample - sample))
                    
                    # Loop filter
                    mu += 0.01 * error  # Simple proportional control
                    mu = np.clip(mu, -0.5, 0.5)
        
        return np.array(symbols)
    
    def transmit_bpsk(
        self,
        bits: np.ndarray,
        symbol_rate: float = 1200
    ) -> np.ndarray:
        """Generate BPSK signal"""
        
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        
        # Map bits to symbols
        symbols = 2 * bits - 1  # 0 -> -1, 1 -> 1
        
        # Upsample
        upsampled = np.repeat(symbols, samples_per_symbol)
        
        # Pulse shaping (raised cosine)
        beta = 0.35
        span = 10
        sps = samples_per_symbol
        
        h = self._rcosfilter(span * sps, beta, 1/symbol_rate, self.sample_rate)
        shaped = np.convolve(upsampled, h, mode='same')
        
        # Modulate to carrier
        t = np.arange(len(shaped)) / self.sample_rate
        carrier = np.exp(2j * np.pi * self.center_freq * t)
        
        modulated = np.real(shaped * carrier)
        
        return modulated
    
    def _rcosfilter(self, N, beta, Ts, Fs):
        """Raised cosine filter"""
        
        T_delta = Ts / float(Fs)
        time_idx = ((np.arange(N) - N/2)) * T_delta
        sample_num = np.arange(N)
        
        h_rrc = np.zeros(N, dtype=float)
        
        for x in sample_num:
            t = (x - N/2) * T_delta
            if t == 0.0:
                h_rrc[x] = 1.0
            elif beta != 0 and np.abs(t) == Ts/(2.0*beta):
                h_rrc[x] = (np.pi/4.0) * np.sinc(1.0/(2.0*beta))
            else:
                h_rrc[x] = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1.0 - (2.0*beta*t/Ts)**2)
        
        return h_rrc / np.sqrt(np.sum(h_rrc**2))


class ConstellationProtocolHandler:
    """Handle protocols for different satellite constellations"""
    
    def __init__(self):
        self.handlers = {
            SatelliteConstellation.STARLINK: self._handle_starlink,
            SatelliteConstellation.IRIDIUM: self._handle_iridium,
            SatelliteConstellation.GLOBALSTAR: self._handle_globalstar,
            SatelliteConstellation.ORBCOMM: self._handle_orbcomm,
            SatelliteConstellation.SWARM: self._handle_swarm
        }
    
    async def send_message(
        self,
        message: SatelliteMessage,
        satellite: Satellite
    ) -> bool:
        """Send message via satellite"""
        
        handler = self.handlers.get(satellite.constellation)
        if handler:
            return await handler(message, 'send')
        
        # Generic handler
        return await self._handle_generic(message, satellite, 'send')
    
    async def receive_message(
        self,
        satellite: Satellite
    ) -> Optional[SatelliteMessage]:
        """Receive message from satellite"""
        
        handler = self.handlers.get(satellite.constellation)
        if handler:
            return await handler(None, 'receive')
        
        # Generic handler
        return await self._handle_generic(None, satellite, 'receive')
    
    async def _handle_starlink(
        self,
        message: Optional[SatelliteMessage],
        operation: str
    ) -> Union[bool, Optional[SatelliteMessage]]:
        """Handle Starlink protocol"""
        
        try:
            # Initialize Starlink API (hypothetical)
            api = StarlinkAPI()
            
            if operation == 'send':
                # Authenticate
                await api.authenticate()
                
                # Prepare packet
                packet = {
                    'destination': message.destination,
                    'payload': message.payload.hex(),
                    'priority': message.priority,
                    'service_type': message.message_type
                }
                
                # Send via Starlink
                result = await api.send_data(packet)
                
                sat_messages_sent.labels('starlink', message.message_type).inc()
                sat_data_transferred.labels('sent', 'starlink').inc(len(message.payload))
                
                return result.success
            
            elif operation == 'receive':
                # Poll for messages
                data = await api.receive_data(timeout=1.0)
                
                if data:
                    message = SatelliteMessage(
                        id=str(uuid.uuid4()),
                        timestamp=datetime.utcnow(),
                        source=data.source,
                        destination=data.destination,
                        message_type=data.service_type,
                        payload=bytes.fromhex(data.payload),
                        constellation=SatelliteConstellation.STARLINK,
                        satellite_id=data.satellite_id
                    )
                    
                    sat_messages_received.labels('starlink', message.message_type).inc()
                    sat_data_transferred.labels('received', 'starlink').inc(len(message.payload))
                    
                    return message
                
                return None
            
        except Exception as e:
            logger.error(f"Starlink handler error: {e}")
            sat_errors.labels('protocol', 'starlink').inc()
            return False if operation == 'send' else None
    
    async def _handle_iridium(
        self,
        message: Optional[SatelliteMessage],
        operation: str
    ) -> Union[bool, Optional[SatelliteMessage]]:
        """Handle Iridium protocol"""
        
        try:
            # Iridium SBD (Short Burst Data)
            if operation == 'send':
                # Connect to Iridium modem
                modem = serial.Serial('/dev/ttyUSB0', 19200, timeout=1)
                
                # AT commands
                modem.write(b'AT\r')
                response = modem.readline()
                
                if b'OK' in response:
                    # Write message to buffer
                    modem.write(f'AT+SBDWB={len(message.payload)}\r'.encode())
                    modem.write(message.payload)
                    
                    # Initiate SBD session
                    modem.write(b'AT+SBDIX\r')
                    
                    # Wait for response
                    response = modem.readline()
                    
                    modem.close()
                    
                    sat_messages_sent.labels('iridium', message.message_type).inc()
                    
                    return b'+SBDIX: 0' in response
                
                modem.close()
                return False
            
            elif operation == 'receive':
                # Check for incoming messages
                modem = serial.Serial('/dev/ttyUSB0', 19200, timeout=1)
                
                # Check mailbox
                modem.write(b'AT+SBDIXA\r')
                response = modem.readline()
                
                if b'+SBDIX: 1' in response:  # Message available
                    # Read message
                    modem.write(b'AT+SBDRB\r')
                    data = modem.read(340)  # Max Iridium SBD size
                    
                    modem.close()
                    
                    if data:
                        message = SatelliteMessage(
                            id=str(uuid.uuid4()),
                            timestamp=datetime.utcnow(),
                            source='iridium',
                            destination='ground',
                            message_type='sbd',
                            payload=data,
                            constellation=SatelliteConstellation.IRIDIUM
                        )
                        
                        sat_messages_received.labels('iridium', 'sbd').inc()
                        
                        return message
                
                modem.close()
                return None
            
        except Exception as e:
            logger.error(f"Iridium handler error: {e}")
            sat_errors.labels('protocol', 'iridium').inc()
            return False if operation == 'send' else None
    
    async def _handle_globalstar(
        self,
        message: Optional[SatelliteMessage],
        operation: str
    ) -> Union[bool, Optional[SatelliteMessage]]:
        """Handle Globalstar protocol"""
        
        # Globalstar implementation
        # Similar structure to Iridium
        pass
    
    async def _handle_orbcomm(
        self,
        message: Optional[SatelliteMessage],
        operation: str
    ) -> Union[bool, Optional[SatelliteMessage]]:
        """Handle Orbcomm protocol"""
        
        # Orbcomm implementation
        pass
    
    async def _handle_swarm(
        self,
        message: Optional[SatelliteMessage],
        operation: str
    ) -> Union[bool, Optional[SatelliteMessage]]:
        """Handle Swarm protocol"""
        
        # Swarm implementation
        pass
    
    async def _handle_generic(
        self,
        message: Optional[SatelliteMessage],
        satellite: Satellite,
        operation: str
    ) -> Union[bool, Optional[SatelliteMessage]]:
        """Generic satellite protocol handler"""
        
        # Generic implementation using SDR
        modem = SatelliteModem()
        
        if operation == 'send':
            # Convert message to bits
            bits = np.unpackbits(np.frombuffer(message.payload, dtype=np.uint8))
            
            # Modulate
            if satellite.metadata.get('modulation') == 'qpsk':
                signal = modem.transmit_qpsk(bits)
            else:
                signal = modem.transmit_bpsk(bits)
            
            # Transmit would happen here
            # In simulation, just return success
            return True
        
        elif operation == 'receive':
            # Receive samples
            samples = modem.receive_samples(int(modem.sample_rate))
            
            # Demodulate
            if satellite.metadata.get('modulation') == 'qpsk':
                bits = modem.demodulate_qpsk(samples)
            else:
                bits = modem.demodulate_bpsk(samples)
            
            # Pack bits to bytes
            if len(bits) > 0:
                bytes_data = np.packbits(bits).tobytes()
                
                message = SatelliteMessage(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    source=satellite.id,
                    destination='ground',
                    message_type='data',
                    payload=bytes_data,
                    constellation=satellite.constellation,
                    satellite_id=satellite.id
                )
                
                return message
        
        return None


class SatelliteCommunicationManager:
    """Manage satellite communications"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        self.tracker = SatelliteTracker()
        self.link_calculator = LinkBudgetCalculator()
        self.protocol_handler = ConstellationProtocolHandler()
        
        self.satellites: Dict[str, Satellite] = {}
        self.ground_stations: Dict[str, GroundStation] = {}
        self.active_links: Dict[str, SatelliteLink] = {}
        
        self.message_queue = asyncio.Queue()
        self.is_running = False
    
    async def start(self):
        """Start satellite communication manager"""
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._link_monitor())
        asyncio.create_task(self._message_processor())
        asyncio.create_task(self._coverage_analyzer())
        
        # Load satellites and ground stations
        await self._load_satellites()
        await self._load_ground_stations()
        
        logger.info("Satellite communication manager started")
    
    async def stop(self):
        """Stop satellite communication manager"""
        
        self.is_running = False
        logger.info("Satellite communication manager stopped")
    
    async def add_satellite(self, satellite: Satellite) -> bool:
        """Add satellite to system"""
        
        try:
            # Add to tracker
            self.tracker.add_satellite(satellite)
            
            # Store in memory
            self.satellites[satellite.id] = satellite
            
            # Store in database
            # Implementation would store satellite data
            
            logger.info(f"Added satellite: {satellite.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add satellite: {e}")
            return False
    
    async def add_ground_station(self, ground_station: GroundStation) -> bool:
        """Add ground station"""
        
        try:
            self.ground_stations[ground_station.id] = ground_station
            
            logger.info(f"Added ground station: {ground_station.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add ground station: {e}")
            return False
    
    async def send_message(
        self,
        message: SatelliteMessage,
        ground_station_id: Optional[str] = None
    ) -> str:
        """Send message via satellite"""
        
        # Find best satellite
        satellite = await self._select_best_satellite(
            message.constellation,
            ground_station_id
        )
        
        if not satellite:
            raise ValueError("No suitable satellite available")
        
        # Check link quality
        link = self.active_links.get(f"{satellite.id}_{ground_station_id}")
        if not link or link.link_margin_db < 3.0:
            raise ValueError("Insufficient link margin")
        
        # Update message with satellite info
        message.satellite_id = satellite.id
        message.frequency_hz = self._get_frequency(satellite)
        
        # Send via protocol handler
        success = await self.protocol_handler.send_message(message, satellite)
        
        if success:
            # Update metrics
            with sat_latency.labels(
                satellite.constellation.value,
                message.message_type
            ).time():
                # Simulated processing
                await asyncio.sleep(0.1)
            
            logger.info(f"Message {message.id} sent via {satellite.name}")
            return message.id
        else:
            raise RuntimeError("Failed to send message")
    
    async def _select_best_satellite(
        self,
        constellation: SatelliteConstellation,
        ground_station_id: Optional[str] = None
    ) -> Optional[Satellite]:
        """Select best available satellite"""
        
        candidates = []
        
        # Get ground station
        if ground_station_id:
            gs = self.ground_stations.get(ground_station_id)
        else:
            # Use first available
            gs = next(iter(self.ground_stations.values())) if self.ground_stations else None
        
        if not gs:
            return None
        
        # Find visible satellites
        for sat_id, satellite in self.satellites.items():
            if constellation and satellite.constellation != constellation:
                continue
            
            # Check visibility
            az, el, range_km, visible = self.tracker.get_look_angles(
                sat_id,
                gs
            )
            
            if visible:
                # Calculate link budget
                frequency = self._get_frequency(satellite)
                link_budget = self.link_calculator.calculate_link_margin(
                    satellite,
                    gs,
                    range_km,
                    frequency,
                    el
                )
                
                if link_budget['link_margin_db'] > 3.0:  # 3 dB minimum
                    candidates.append((
                        satellite,
                        link_budget['link_margin_db'],
                        el
                    ))
        
        if not candidates:
            return None
        
        # Select best (highest elevation for now)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates[0][0]
    
    def _get_frequency(self, satellite: Satellite) -> float:
        """Get operating frequency for satellite"""
        
        # Map frequency bands to center frequencies
        freq_map = {
            FrequencyBand.L_BAND: 1.5e9,
            FrequencyBand.S_BAND: 2.2e9,
            FrequencyBand.C_BAND: 6e9,
            FrequencyBand.X_BAND: 10e9,
            FrequencyBand.KU_BAND: 14e9,
            FrequencyBand.K_BAND: 20e9,
            FrequencyBand.KA_BAND: 30e9,
            FrequencyBand.V_BAND: 50e9
        }
        
        if satellite.frequency_bands:
            return freq_map.get(satellite.frequency_bands[0], 1.5e9)
        
        return 1.5e9  # Default L-band
    
    async def _link_monitor(self):
        """Monitor satellite links"""
        
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for gs_id, gs in self.ground_stations.items():
                    for sat_id, satellite in self.satellites.items():
                        # Get link parameters
                        az, el, range_km, visible = self.tracker.get_look_angles(
                            sat_id,
                            gs,
                            current_time
                        )
                        
                        link_id = f"{sat_id}_{gs_id}"
                        
                        if visible:
                            # Calculate link budget
                            frequency = self._get_frequency(satellite)
                            link_budget = self.link_calculator.calculate_link_margin(
                                satellite,
                                gs,
                                range_km,
                                frequency,
                                el
                            )
                            
                            # Calculate Doppler
                            doppler = self.tracker.get_doppler_shift(
                                sat_id,
                                gs,
                                frequency,
                                current_time
                            )
                            
                            # Create/update link
                            link = SatelliteLink(
                                satellite=satellite,
                                ground_station=gs,
                                timestamp=current_time,
                                azimuth_deg=az,
                                elevation_deg=el,
                                range_km=range_km,
                                doppler_shift_hz=doppler,
                                path_loss_db=link_budget['path_loss_db'],
                                link_margin_db=link_budget['link_margin_db'],
                                data_rate_mbps=link_budget['data_rate_mbps'],
                                ber=10**(-link_budget['link_margin_db']/10),  # Simplified
                                is_visible=True
                            )
                            
                            self.active_links[link_id] = link
                            
                            # Update metrics
                            sat_link_quality.labels(
                                sat_id,
                                'link_margin'
                            ).set(link_budget['link_margin_db'])
                            
                            sat_link_quality.labels(
                                sat_id,
                                'elevation'
                            ).set(el)
                            
                        else:
                            # Remove inactive link
                            if link_id in self.active_links:
                                del self.active_links[link_id]
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Link monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _message_processor(self):
        """Process incoming messages"""
        
        while self.is_running:
            try:
                # Check each constellation for messages
                for constellation in SatelliteConstellation:
                    # Find satellites of this constellation
                    sats = [
                        s for s in self.satellites.values()
                        if s.constellation == constellation
                    ]
                    
                    if sats:
                        # Try to receive from first available
                        message = await self.protocol_handler.receive_message(sats[0])
                        
                        if message:
                            await self.message_queue.put(message)
                            logger.info(
                                f"Received message {message.id} from {constellation.value}"
                            )
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _coverage_analyzer(self):
        """Analyze coverage patterns"""
        
        while self.is_running:
            try:
                # Analyze coverage every 5 minutes
                await asyncio.sleep(300)
                
                # Get all satellite IDs
                sat_ids = list(self.satellites.keys())
                
                if sat_ids:
                    # Define region of interest (example: continental US)
                    from shapely.geometry import box
                    region = box(-125, 24, -66, 49)  # West, South, East, North
                    
                    # Calculate coverage
                    coverage = self.tracker.predict_coverage(
                        sat_ids,
                        region,
                        datetime.utcnow()
                    )
                    
                    logger.info(
                        f"Coverage analysis: max {coverage['max_satellites']} satellites, "
                        f"avg {coverage['avg_satellites']:.1f}"
                    )
                    
                    # Could store coverage maps for visualization
                
            except Exception as e:
                logger.error(f"Coverage analyzer error: {e}")
    
    async def _load_satellites(self):
        """Load satellite catalog"""
        
        # Example satellites - in production would load from database/API
        
        # Starlink satellites (example subset)
        starlink_tles = [
            ("STARLINK-1007",
             "1 44713U 19074A   23365.50000000  .00001234  00000-0  12345-3 0  9999",
             "2 44713  53.0536 123.4567 0001234  86.1234 274.1234 15.06391234123456"),
            ("STARLINK-1008",
             "1 44714U 19074B   23365.50000000  .00001234  00000-0  12345-3 0  9999",
             "2 44714  53.0536 123.4567 0001234  86.1234 274.1234 15.06391234123456"),
        ]
        
        for name, tle1, tle2 in starlink_tles:
            satellite = Satellite(
                id=name.lower().replace('-', '_'),
                name=name,
                constellation=SatelliteConstellation.STARLINK,
                tle_line1=tle1,
                tle_line2=tle2,
                frequency_bands=[FrequencyBand.KU_BAND, FrequencyBand.KA_BAND],
                services=[ServiceType.DATA, ServiceType.VIDEO],
                bandwidth_mbps=100,
                power_watts=100,
                antenna_gain_dbi=33,
                noise_figure_db=1.5,
                metadata={'generation': 'v1.5'}
            )
            
            await self.add_satellite(satellite)
        
        # Iridium satellites
        iridium_tle = (
            "IRIDIUM 103",
            "1 41917U 17003A   23365.50000000  .00000123  00000-0  12345-4 0  9999",
            "2 41917  86.3978 123.4567 0001234 123.4567  23.4567 14.34212345123456"
        )
        
        iridium_sat = Satellite(
            id="iridium_103",
            name=iridium_tle[0],
            constellation=SatelliteConstellation.IRIDIUM,
            tle_line1=iridium_tle[1],
            tle_line2=iridium_tle[2],
            frequency_bands=[FrequencyBand.L_BAND],
            services=[ServiceType.DATA, ServiceType.VOICE, ServiceType.IOT],
            bandwidth_mbps=2.4,
            power_watts=50,
            antenna_gain_dbi=25,
            noise_figure_db=2.0,
            metadata={'generation': 'NEXT'}
        )
        
        await self.add_satellite(iridium_sat)
    
    async def _load_ground_stations(self):
        """Load ground stations"""
        
        # Example ground stations
        stations = [
            {
                'id': 'gs_seattle',
                'name': 'Seattle Ground Station',
                'lat': 47.6062,
                'lon': -122.3321,
                'alt': 100,
                'diameter': 3.7,
                'gain': 45,
                'power': 100,
                'temp': 150,
                'bands': [FrequencyBand.KU_BAND, FrequencyBand.KA_BAND]
            },
            {
                'id': 'gs_denver',
                'name': 'Denver Ground Station',
                'lat': 39.7392,
                'lon': -104.9903,
                'alt': 1600,
                'diameter': 5.0,
                'gain': 48,
                'power': 200,
                'temp': 120,
                'bands': [FrequencyBand.KU_BAND, FrequencyBand.KA_BAND, FrequencyBand.X_BAND]
            }
        ]
        
        for config in stations:
            gs = GroundStation(
                id=config['id'],
                name=config['name'],
                location=EarthLocation(
                    lat=config['lat'] * u.deg,
                    lon=config['lon'] * u.deg,
                    height=config['alt'] * u.m
                ),
                antenna_diameter_m=config['diameter'],
                antenna_gain_dbi=config['gain'],
                transmit_power_watts=config['power'],
                noise_temperature_k=config['temp'],
                frequency_bands=config['bands']
            )
            
            await self.add_ground_station(gs)
    
    def visualize_coverage(
        self,
        output_file: str = "coverage_map.html"
    ) -> str:
        """Create interactive coverage visualization"""
        
        # Create folium map
        m = folium.Map(
            location=[40.0, -95.0],  # Center of US
            zoom_start=4
        )
        
        # Add ground stations
        for gs_id, gs in self.ground_stations.items():
            folium.Marker(
                [gs.location.lat.deg, gs.location.lon.deg],
                popup=f"{gs.name}<br>Bands: {[b.value for b in gs.frequency_bands]}",
                icon=folium.Icon(color='green', icon='broadcast-tower', prefix='fa')
            ).add_to(m)
        
        # Add satellite positions and footprints
        for sat_id, satellite in self.satellites.items():
            try:
                lat, lon, alt_km = self.tracker.get_position(sat_id)
                
                # Satellite marker
                folium.Marker(
                    [lat, lon],
                    popup=f"{satellite.name}<br>Alt: {alt_km:.0f} km<br>Constellation: {satellite.constellation.value}",
                    icon=folium.Icon(color='red', icon='satellite', prefix='fa')
                ).add_to(m)
                
                # Footprint (simplified circular)
                # Calculate footprint radius based on altitude and min elevation
                earth_radius_km = 6371
                min_elevation_rad = np.radians(10)  # 10 degree mask
                
                # Spherical geometry
                horizon_angle = np.arccos(
                    earth_radius_km / (earth_radius_km + alt_km)
                ) - min_elevation_rad
                
                footprint_radius_km = earth_radius_km * horizon_angle
                
                # Draw footprint
                folium.Circle(
                    [lat, lon],
                    radius=footprint_radius_km * 1000,  # Convert to meters
                    color='blue',
                    fill=True,
                    fillOpacity=0.1,
                    popup=f"{satellite.name} footprint"
                ).add_to(m)
                
            except Exception as e:
                logger.error(f"Error plotting satellite {sat_id}: {e}")
        
        # Add active links
        for link_id, link in self.active_links.items():
            if link.is_visible:
                sat_lat, sat_lon, _ = self.tracker.get_position(link.satellite.id)
                gs_lat = link.ground_station.location.lat.deg
                gs_lon = link.ground_station.location.lon.deg
                
                # Color based on link quality
                if link.link_margin_db > 10:
                    color = 'green'
                elif link.link_margin_db > 5:
                    color = 'yellow'
                else:
                    color = 'red'
                
                folium.PolyLine(
                    [[gs_lat, gs_lon], [sat_lat, sat_lon]],
                    color=color,
                    weight=2,
                    opacity=0.8,
                    popup=f"Link margin: {link.link_margin_db:.1f} dB<br>Data rate: {link.data_rate_mbps:.1f} Mbps"
                ).add_to(m)
        
        # Save map
        m.save(output_file)
        
        return output_file
    
    async def emergency_broadcast(
        self,
        message: str,
        region: Optional[Polygon] = None
    ) -> Dict[str, Any]:
        """Send emergency broadcast via multiple satellites"""
        
        results = {
            'broadcast_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow(),
            'satellites_used': [],
            'coverage_area': None,
            'estimated_recipients': 0
        }
        
        # Find all available satellites
        available_sats = []
        
        for sat_id, satellite in self.satellites.items():
            # Check if satellite supports emergency service
            if ServiceType.EMERGENCY not in satellite.services:
                continue
            
            # Check visibility from any ground station
            for gs in self.ground_stations.values():
                _, _, _, visible = self.tracker.get_look_angles(sat_id, gs)
                if visible:
                    available_sats.append(satellite)
                    break
        
        # Send via all available satellites
        for satellite in available_sats:
            try:
                emergency_msg = SatelliteMessage(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    source='emergency_system',
                    destination='broadcast',
                    message_type='emergency',
                    payload=message.encode('utf-8'),
                    constellation=satellite.constellation,
                    priority=10  # Highest priority
                )
                
                success = await self.protocol_handler.send_message(
                    emergency_msg,
                    satellite
                )
                
                if success:
                    results['satellites_used'].append({
                        'satellite_id': satellite.id,
                        'constellation': satellite.constellation.value,
                        'status': 'success'
                    })
                
            except Exception as e:
                logger.error(f"Emergency broadcast error for {satellite.id}: {e}")
                results['satellites_used'].append({
                    'satellite_id': satellite.id,
                    'constellation': satellite.constellation.value,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Calculate coverage area
        if results['satellites_used']:
            # Simplified - union of all satellite footprints
            footprints = []
            
            for sat_info in results['satellites_used']:
                if sat_info['status'] == 'success':
                    sat_id = sat_info['satellite_id']
                    lat, lon, alt_km = self.tracker.get_position(sat_id)
                    
                    # Calculate footprint
                    earth_radius_km = 6371
                    horizon_angle = np.arccos(earth_radius_km / (earth_radius_km + alt_km))
                    footprint_radius_deg = np.degrees(horizon_angle)
                    
                    # Create circular footprint
                    footprint = Point(lon, lat).buffer(footprint_radius_deg)
                    footprints.append(footprint)
            
            if footprints:
                coverage = unary_union(footprints)
                results['coverage_area'] = coverage.area
                
                # Estimate recipients (very rough)
                # Assume 100 people per square degree in populated areas
                results['estimated_recipients'] = int(coverage.area * 100)
        
        return results


# Example usage
async def satellite_demo():
    """Demo satellite communication system"""
    
    # Initialize manager
    manager = SatelliteCommunicationManager(
        'postgresql://user:pass@localhost/satellite_db'
    )
    await manager.start()
    
    # Wait for satellites to be loaded
    await asyncio.sleep(2)
    
    # Send a message via Starlink
    message = SatelliteMessage(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        source='mcp_node_1',
        destination='mcp_cloud',
        message_type='telemetry',
        payload=json.dumps({
            'temperature': 25.5,
            'humidity': 60.2,
            'status': 'operational'
        }).encode('utf-8'),
        constellation=SatelliteConstellation.STARLINK,
        priority=5
    )
    
    try:
        message_id = await manager.send_message(message, 'gs_seattle')
        print(f"Message sent successfully: {message_id}")
    except Exception as e:
        print(f"Failed to send message: {e}")
    
    # Check satellite visibility
    print("\nSatellite Visibility:")
    for gs_id, gs in manager.ground_stations.items():
        print(f"\nFrom {gs.name}:")
        
        for sat_id, satellite in manager.satellites.items():
            az, el, range_km, visible = manager.tracker.get_look_angles(
                sat_id,
                gs
            )
            
            if visible:
                print(f"  {satellite.name}: Az={az:.1f}° El={el:.1f}° Range={range_km:.0f}km")
    
    # Calculate passes for next 24 hours
    print("\nUpcoming Starlink Passes (Seattle):")
    
    starlink_sats = [
        s for s in manager.satellites.values()
        if s.constellation == SatelliteConstellation.STARLINK
    ]
    
    if starlink_sats:
        passes = manager.tracker.calculate_passes(
            starlink_sats[0].id,
            manager.ground_stations['gs_seattle'],
            datetime.utcnow(),
            24
        )
        
        for pass_info in passes[:5]:  # First 5 passes
            duration = pass_info['duration_seconds']
            print(f"  {pass_info['rise_time']} - Duration: {duration:.0f}s, Max El: {pass_info['max_elevation_deg']:.1f}°")
    
    # Test emergency broadcast
    print("\nSending emergency broadcast...")
    
    broadcast_result = await manager.emergency_broadcast(
        "This is a test emergency message from MCP satellite system"
    )
    
    print(f"Broadcast ID: {broadcast_result['broadcast_id']}")
    print(f"Satellites used: {len(broadcast_result['satellites_used'])}")
    print(f"Estimated coverage: {broadcast_result['estimated_recipients']} recipients")
    
    # Create coverage visualization
    print("\nGenerating coverage map...")
    map_file = manager.visualize_coverage()
    print(f"Coverage map saved to: {map_file}")
    
    # Monitor for incoming messages
    print("\nMonitoring for incoming messages...")
    
    timeout = 10  # seconds
    start_time = datetime.utcnow()
    
    while (datetime.utcnow() - start_time).total_seconds() < timeout:
        try:
            incoming = await asyncio.wait_for(
                manager.message_queue.get(),
                timeout=1.0
            )
            
            print(f"\nReceived message:")
            print(f"  ID: {incoming.id}")
            print(f"  From: {incoming.source}")
            print(f"  Type: {incoming.message_type}")
            print(f"  Constellation: {incoming.constellation.value}")
            print(f"  Payload: {incoming.payload.decode('utf-8', errors='ignore')[:100]}...")
            
        except asyncio.TimeoutError:
            print(".", end="", flush=True)
    
    # Get active links
    print(f"\n\nActive satellite links: {len(manager.active_links)}")
    
    for link_id, link in manager.active_links.items():
        print(f"\n{link_id}:")
        print(f"  Satellite: {link.satellite.name}")
        print(f"  Ground Station: {link.ground_station.name}")
        print(f"  Elevation: {link.elevation_deg:.1f}°")
        print(f"  Range: {link.range_km:.0f} km")
        print(f"  Link Margin: {link.link_margin_db:.1f} dB")
        print(f"  Data Rate: {link.data_rate_mbps:.1f} Mbps")
        print(f"  Doppler: {link.doppler_shift_hz:.0f} Hz")
    
    # Predict coverage for a region
    from shapely.geometry import box
    california = box(-124.5, 32.5, -114.0, 42.0)  # California bounding box
    
    coverage = manager.tracker.predict_coverage(
        list(manager.satellites.keys()),
        california,
        datetime.utcnow()
    )
    
    print(f"\nCalifornia coverage analysis:")
    print(f"  Max simultaneous satellites: {coverage['max_satellites']}")
    print(f"  Average satellite visibility: {coverage['avg_satellites']:.1f}")
    
    # Stop manager
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(satellite_demo())