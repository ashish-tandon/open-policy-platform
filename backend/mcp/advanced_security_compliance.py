"""
Advanced Security and Compliance System - 40by6
Enterprise-grade security, audit trails, and compliance features
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import hmac
import secrets
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, JSON, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis.asyncio as redis
from pathlib import Path
import yaml
import pyotp
import qrcode
from io import BytesIO
import aiohttp
from email_validator import validate_email
import ipaddress
from collections import defaultdict
import geoip2.database
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

Base = declarative_base()


class SecurityEventType(Enum):
    """Types of security events"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    CONFIG_CHANGE = "config_change"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"


class ComplianceStandard(Enum):
    """Compliance standards supported"""
    GDPR = "gdpr"  # General Data Protection Regulation
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"  # Information Security Management
    NIST = "nist"  # NIST Cybersecurity Framework
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    id: str
    name: str
    description: str
    rules: Dict[str, Any]
    compliance_standards: List[ComplianceStandard]
    enforcement_level: str  # strict, moderate, relaxed
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'rules': self.rules,
            'compliance_standards': [s.value for s in self.compliance_standards],
            'enforcement_level': self.enforcement_level,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


# Database Models
class SecurityEvent(Base):
    __tablename__ = 'security_events'
    
    id = Column(String, primary_key=True)
    event_type = Column(String, nullable=False)
    user_id = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    resource = Column(String)
    action = Column(String)
    result = Column(String)
    risk_score = Column(Float, default=0.0)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    audit_trail = relationship("AuditTrail", back_populates="security_event")


class AuditTrail(Base):
    __tablename__ = 'audit_trail'
    
    id = Column(String, primary_key=True)
    entity_type = Column(String, nullable=False)  # scraper, user, config, etc.
    entity_id = Column(String, nullable=False)
    action = Column(String, nullable=False)  # create, update, delete, access
    user_id = Column(String)
    changes = Column(JSON)  # before/after values
    ip_address = Column(String)
    session_id = Column(String)
    security_event_id = Column(String, ForeignKey('security_events.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    security_event = relationship("SecurityEvent", back_populates="audit_trail")


class DataClassification(Base):
    __tablename__ = 'data_classifications'
    
    id = Column(String, primary_key=True)
    entity_type = Column(String)
    entity_id = Column(String)
    field_name = Column(String)
    classification = Column(String)  # public, internal, confidential, restricted
    sensitivity_level = Column(Integer)  # 1-5
    pii_type = Column(String)  # email, phone, ssn, etc.
    retention_days = Column(Integer)
    encryption_required = Column(Boolean, default=False)
    compliance_tags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class AccessControl(Base):
    __tablename__ = 'access_controls'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    resource_type = Column(String)
    resource_id = Column(String)
    permissions = Column(JSON)  # read, write, delete, admin
    conditions = Column(JSON)  # time-based, ip-based, etc.
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String)


class EncryptionManager:
    """Manages encryption for sensitive data"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        self.field_keys = {}  # Field-specific encryption keys
    
    def generate_field_key(self, field_name: str) -> bytes:
        """Generate field-specific encryption key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=field_name.encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.field_keys[field_name] = key
        return key
    
    def encrypt_field(self, value: str, field_name: str) -> str:
        """Encrypt a field value"""
        if field_name not in self.field_keys:
            self.generate_field_key(field_name)
        
        fernet = Fernet(self.field_keys[field_name])
        encrypted = fernet.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_field(self, encrypted_value: str, field_name: str) -> str:
        """Decrypt a field value"""
        if field_name not in self.field_keys:
            self.generate_field_key(field_name)
        
        fernet = Fernet(self.field_keys[field_name])
        decoded = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted = fernet.decrypt(decoded)
        return decrypted.decode()
    
    def hash_sensitive_data(self, value: str, salt: Optional[str] = None) -> str:
        """Create secure hash of sensitive data"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{salt}:{value}".encode()
        hash_value = hashlib.pbkdf2_hmac('sha256', combined, salt.encode(), 100000)
        return f"{salt}:{hash_value.hex()}"
    
    def verify_hash(self, value: str, hashed: str) -> bool:
        """Verify a value against its hash"""
        try:
            salt, stored_hash = hashed.split(':', 1)
            test_hash = self.hash_sensitive_data(value, salt)
            return hmac.compare_digest(test_hash, hashed)
        except:
            return False


class AccessControlManager:
    """Manages access control and permissions"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.permission_cache = {}
        self.role_hierarchy = self._load_role_hierarchy()
    
    def _load_role_hierarchy(self) -> Dict[str, List[str]]:
        """Load role hierarchy"""
        return {
            'admin': ['manager', 'analyst', 'viewer'],
            'manager': ['analyst', 'viewer'],
            'analyst': ['viewer'],
            'viewer': []
        }
    
    async def check_permission(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if user has permission for action"""
        
        # Check cache first
        cache_key = f"{user_id}:{resource_type}:{resource_id}:{action}"
        if cache_key in self.permission_cache:
            cached = self.permission_cache[cache_key]
            if cached['expires'] > datetime.utcnow():
                return cached['allowed'], cached['reason']
        
        session = self.Session()
        try:
            # Get user's access controls
            controls = session.query(AccessControl).filter(
                AccessControl.user_id == user_id,
                (AccessControl.resource_type == resource_type) | (AccessControl.resource_type == '*'),
                (AccessControl.resource_id == resource_id) | (AccessControl.resource_id == '*')
            ).all()
            
            for control in controls:
                # Check expiration
                if control.expires_at and control.expires_at < datetime.utcnow():
                    continue
                
                # Check permissions
                if action in control.permissions or '*' in control.permissions:
                    # Check conditions
                    if self._check_conditions(control.conditions, context):
                        # Cache the result
                        self.permission_cache[cache_key] = {
                            'allowed': True,
                            'reason': 'Permission granted',
                            'expires': datetime.utcnow() + timedelta(minutes=5)
                        }
                        return True, "Permission granted"
            
            # No explicit permission found
            self.permission_cache[cache_key] = {
                'allowed': False,
                'reason': 'No permission found',
                'expires': datetime.utcnow() + timedelta(minutes=5)
            }
            return False, "Access denied: No permission found"
            
        finally:
            session.close()
    
    def _check_conditions(self, conditions: Optional[Dict[str, Any]], context: Optional[Dict[str, Any]]) -> bool:
        """Check access conditions"""
        if not conditions:
            return True
        
        if not context:
            return False
        
        # Time-based conditions
        if 'time_range' in conditions:
            current_hour = datetime.utcnow().hour
            start_hour = conditions['time_range'].get('start', 0)
            end_hour = conditions['time_range'].get('end', 24)
            if not (start_hour <= current_hour < end_hour):
                return False
        
        # IP-based conditions
        if 'allowed_ips' in conditions and context.get('ip_address'):
            ip = ipaddress.ip_address(context['ip_address'])
            allowed = False
            for allowed_range in conditions['allowed_ips']:
                if ip in ipaddress.ip_network(allowed_range):
                    allowed = True
                    break
            if not allowed:
                return False
        
        # Geo-based conditions
        if 'allowed_countries' in conditions and context.get('country'):
            if context['country'] not in conditions['allowed_countries']:
                return False
        
        return True
    
    async def grant_permission(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        permissions: List[str],
        granted_by: str,
        expires_in_days: Optional[int] = None,
        conditions: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Grant permissions to a user"""
        
        session = self.Session()
        try:
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            access_control = AccessControl(
                id=f"ac_{user_id}_{resource_type}_{resource_id}_{datetime.utcnow().timestamp()}",
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                permissions=permissions,
                conditions=conditions,
                expires_at=expires_at,
                created_by=granted_by
            )
            
            session.add(access_control)
            session.commit()
            
            # Clear cache
            self._clear_permission_cache(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error granting permission: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def _clear_permission_cache(self, user_id: str):
        """Clear permission cache for user"""
        keys_to_remove = [k for k in self.permission_cache if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.permission_cache[key]


class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.redis_client = None
        self.geo_reader = None
        
        # Try to load GeoIP database
        try:
            self.geo_reader = geoip2.database.Reader('GeoLite2-City.mmdb')
        except:
            logger.warning("GeoIP database not found")
    
    async def initialize(self):
        """Initialize audit logger"""
        self.redis_client = await redis.from_url('redis://localhost:6379')
    
    async def log_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = 'success',
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Log a security event"""
        
        event_id = f"evt_{datetime.utcnow().timestamp()}_{secrets.token_hex(8)}"
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(event_type, result, metadata)
        
        # Get geo information
        geo_info = {}
        if ip_address and self.geo_reader:
            try:
                response = self.geo_reader.city(ip_address)
                geo_info = {
                    'country': response.country.iso_code,
                    'city': response.city.name,
                    'lat': response.location.latitude,
                    'lon': response.location.longitude
                }
            except:
                pass
        
        # Prepare metadata
        event_metadata = metadata or {}
        event_metadata.update(geo_info)
        
        session = self.Session()
        try:
            event = SecurityEvent(
                id=event_id,
                event_type=event_type.value,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                action=action,
                result=result,
                risk_score=risk_score,
                metadata=event_metadata
            )
            
            session.add(event)
            session.commit()
            
            # Real-time alerting for high-risk events
            if risk_score > 0.7:
                await self._alert_high_risk_event(event)
            
            # Update user risk profile
            if user_id:
                await self._update_user_risk_profile(user_id, event_type, risk_score)
            
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
            session.rollback()
            return ""
        finally:
            session.close()
    
    async def log_data_access(
        self,
        user_id: str,
        entity_type: str,
        entity_id: str,
        fields_accessed: List[str],
        purpose: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """Log data access for compliance"""
        
        # Check data classification
        classifications = await self._get_data_classifications(entity_type, entity_id, fields_accessed)
        
        # Log access event
        metadata = {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'fields': fields_accessed,
            'purpose': purpose,
            'classifications': classifications
        }
        
        event_id = await self.log_event(
            SecurityEventType.DATA_ACCESS,
            user_id=user_id,
            resource=f"{entity_type}/{entity_id}",
            action='read',
            metadata=metadata,
            ip_address=ip_address
        )
        
        # Create audit trail
        session = self.Session()
        try:
            audit = AuditTrail(
                id=f"audit_{datetime.utcnow().timestamp()}_{secrets.token_hex(8)}",
                entity_type=entity_type,
                entity_id=entity_id,
                action='access',
                user_id=user_id,
                changes={'fields_accessed': fields_accessed},
                ip_address=ip_address,
                security_event_id=event_id
            )
            
            session.add(audit)
            session.commit()
            
        finally:
            session.close()
    
    async def log_change(
        self,
        user_id: str,
        entity_type: str,
        entity_id: str,
        action: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log entity changes for audit trail"""
        
        # Calculate what changed
        changes = self._calculate_changes(before, after)
        
        session = self.Session()
        try:
            audit = AuditTrail(
                id=f"audit_{datetime.utcnow().timestamp()}_{secrets.token_hex(8)}",
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                user_id=user_id,
                changes={
                    'before': before,
                    'after': after,
                    'diff': changes
                },
                ip_address=ip_address,
                session_id=session_id
            )
            
            session.add(audit)
            session.commit()
            
            # Check for suspicious changes
            if self._is_suspicious_change(entity_type, changes):
                await self.log_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    user_id=user_id,
                    resource=f"{entity_type}/{entity_id}",
                    action=action,
                    metadata={'changes': changes},
                    ip_address=ip_address
                )
            
        finally:
            session.close()
    
    def _calculate_risk_score(
        self,
        event_type: SecurityEventType,
        result: str,
        metadata: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate risk score for event"""
        
        base_scores = {
            SecurityEventType.LOGIN_FAILED: 0.3,
            SecurityEventType.ACCESS_DENIED: 0.4,
            SecurityEventType.PRIVILEGE_ESCALATION: 0.8,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 0.9,
            SecurityEventType.RATE_LIMIT_EXCEEDED: 0.5,
            SecurityEventType.CONFIG_CHANGE: 0.6
        }
        
        score = base_scores.get(event_type, 0.1)
        
        # Adjust based on result
        if result == 'failed':
            score += 0.2
        
        # Adjust based on metadata
        if metadata:
            if metadata.get('repeated_attempts', 0) > 5:
                score += 0.3
            if metadata.get('from_unknown_location', False):
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_changes(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between before and after"""
        changes = {}
        
        all_keys = set(before.keys()) | set(after.keys())
        
        for key in all_keys:
            before_val = before.get(key)
            after_val = after.get(key)
            
            if before_val != after_val:
                changes[key] = {
                    'before': before_val,
                    'after': after_val
                }
        
        return changes
    
    def _is_suspicious_change(self, entity_type: str, changes: Dict[str, Any]) -> bool:
        """Detect suspicious changes"""
        
        # Sensitive fields by entity type
        sensitive_fields = {
            'user': ['role', 'permissions', 'email', 'is_active'],
            'scraper': ['config', 'credentials', 'schedule'],
            'config': ['security_settings', 'api_keys', 'database_url']
        }
        
        entity_sensitive = sensitive_fields.get(entity_type, [])
        
        for field in changes:
            if field in entity_sensitive:
                return True
        
        # Large number of changes
        if len(changes) > 10:
            return True
        
        return False
    
    async def _alert_high_risk_event(self, event: SecurityEvent):
        """Alert on high-risk security events"""
        if self.redis_client:
            alert = {
                'event_id': event.id,
                'event_type': event.event_type,
                'risk_score': event.risk_score,
                'user_id': event.user_id,
                'timestamp': event.timestamp.isoformat()
            }
            
            await self.redis_client.publish('security:alerts', json.dumps(alert))
    
    async def _update_user_risk_profile(self, user_id: str, event_type: SecurityEventType, risk_score: float):
        """Update user's risk profile"""
        if self.redis_client:
            key = f"user:risk:{user_id}"
            
            # Increment event counter
            await self.redis_client.hincrby(key, event_type.value, 1)
            
            # Update risk score (moving average)
            current_score = await self.redis_client.hget(key, 'risk_score')
            if current_score:
                new_score = (float(current_score) * 0.9) + (risk_score * 0.1)
            else:
                new_score = risk_score
            
            await self.redis_client.hset(key, 'risk_score', str(new_score))
            await self.redis_client.hset(key, 'last_updated', datetime.utcnow().isoformat())
            
            # Set expiry
            await self.redis_client.expire(key, 86400 * 30)  # 30 days
    
    async def _get_data_classifications(
        self,
        entity_type: str,
        entity_id: str,
        fields: List[str]
    ) -> Dict[str, str]:
        """Get data classifications for fields"""
        
        session = self.Session()
        try:
            classifications = {}
            
            results = session.query(DataClassification).filter(
                DataClassification.entity_type == entity_type,
                DataClassification.entity_id.in_([entity_id, '*']),
                DataClassification.field_name.in_(fields)
            ).all()
            
            for result in results:
                classifications[result.field_name] = {
                    'classification': result.classification,
                    'sensitivity': result.sensitivity_level,
                    'pii_type': result.pii_type
                }
            
            return classifications
            
        finally:
            session.close()


class ComplianceManager:
    """Manages compliance with various standards"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.policies = self._load_compliance_policies()
        self.encryption_manager = EncryptionManager()
    
    def _load_compliance_policies(self) -> Dict[ComplianceStandard, SecurityPolicy]:
        """Load compliance policies"""
        policies = {}
        
        # GDPR Policy
        policies[ComplianceStandard.GDPR] = SecurityPolicy(
            id='policy_gdpr',
            name='GDPR Compliance Policy',
            description='EU General Data Protection Regulation compliance',
            rules={
                'data_retention': {
                    'personal_data': 365,  # days
                    'sensitive_data': 90,
                    'logs': 180
                },
                'consent_required': True,
                'right_to_erasure': True,
                'data_portability': True,
                'breach_notification': 72,  # hours
                'encryption_required': ['email', 'phone', 'address', 'ssn'],
                'anonymization_threshold': 30  # days
            },
            compliance_standards=[ComplianceStandard.GDPR],
            enforcement_level='strict'
        )
        
        # PIPEDA Policy
        policies[ComplianceStandard.PIPEDA] = SecurityPolicy(
            id='policy_pipeda',
            name='PIPEDA Compliance Policy',
            description='Canadian privacy law compliance',
            rules={
                'consent_required': True,
                'purpose_limitation': True,
                'data_minimization': True,
                'accuracy_requirement': True,
                'safeguards': ['encryption', 'access_control', 'audit_logging'],
                'openness': True,
                'individual_access': True,
                'retention_limits': True
            },
            compliance_standards=[ComplianceStandard.PIPEDA],
            enforcement_level='strict'
        )
        
        return policies
    
    async def classify_data(
        self,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, DataClassification]:
        """Classify data fields for compliance"""
        
        classifications = {}
        
        for field_name, value in data.items():
            classification = self._classify_field(field_name, value)
            
            if classification:
                session = self.Session()
                try:
                    db_classification = DataClassification(
                        id=f"cls_{entity_type}_{entity_id}_{field_name}",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        field_name=field_name,
                        classification=classification['classification'],
                        sensitivity_level=classification['sensitivity'],
                        pii_type=classification.get('pii_type'),
                        retention_days=classification['retention_days'],
                        encryption_required=classification['encryption_required'],
                        compliance_tags=classification.get('compliance_tags', [])
                    )
                    
                    session.merge(db_classification)
                    session.commit()
                    
                    classifications[field_name] = db_classification
                    
                finally:
                    session.close()
        
        return classifications
    
    def _classify_field(self, field_name: str, value: Any) -> Optional[Dict[str, Any]]:
        """Classify a single field"""
        
        # Email detection
        if 'email' in field_name.lower() or (isinstance(value, str) and '@' in value):
            return {
                'classification': 'confidential',
                'sensitivity': 3,
                'pii_type': 'email',
                'retention_days': 365,
                'encryption_required': True,
                'compliance_tags': ['gdpr', 'pipeda']
            }
        
        # Phone detection
        phone_pattern = re.compile(r'[\d\s\-\(\)\+]{10,}')
        if 'phone' in field_name.lower() or (isinstance(value, str) and phone_pattern.match(value)):
            return {
                'classification': 'confidential',
                'sensitivity': 3,
                'pii_type': 'phone',
                'retention_days': 365,
                'encryption_required': True,
                'compliance_tags': ['gdpr', 'pipeda']
            }
        
        # SSN/SIN detection
        ssn_pattern = re.compile(r'^\d{3}-?\d{2}-?\d{4}$')
        if 'ssn' in field_name.lower() or 'sin' in field_name.lower() or \
           (isinstance(value, str) and ssn_pattern.match(value)):
            return {
                'classification': 'restricted',
                'sensitivity': 5,
                'pii_type': 'ssn',
                'retention_days': 90,
                'encryption_required': True,
                'compliance_tags': ['gdpr', 'pipeda', 'pci_dss']
            }
        
        # Address detection
        if 'address' in field_name.lower() or 'street' in field_name.lower():
            return {
                'classification': 'confidential',
                'sensitivity': 3,
                'pii_type': 'address',
                'retention_days': 365,
                'encryption_required': True,
                'compliance_tags': ['gdpr', 'pipeda']
            }
        
        # Financial data
        if any(term in field_name.lower() for term in ['credit', 'card', 'bank', 'account']):
            return {
                'classification': 'restricted',
                'sensitivity': 5,
                'pii_type': 'financial',
                'retention_days': 180,
                'encryption_required': True,
                'compliance_tags': ['pci_dss', 'gdpr']
            }
        
        # Default classification
        return {
            'classification': 'internal',
            'sensitivity': 1,
            'retention_days': 730,
            'encryption_required': False,
            'compliance_tags': []
        }
    
    async def apply_retention_policy(self, dry_run: bool = True) -> Dict[str, int]:
        """Apply data retention policies"""
        
        results = {
            'records_to_delete': 0,
            'records_to_anonymize': 0,
            'records_processed': 0
        }
        
        session = self.Session()
        try:
            # Get all data classifications
            classifications = session.query(DataClassification).all()
            
            for classification in classifications:
                # Check retention period
                retention_date = classification.created_at + timedelta(days=classification.retention_days)
                
                if datetime.utcnow() > retention_date:
                    if classification.sensitivity_level >= 4:
                        # Delete sensitive data
                        results['records_to_delete'] += 1
                        if not dry_run:
                            await self._delete_data(
                                classification.entity_type,
                                classification.entity_id,
                                classification.field_name
                            )
                    else:
                        # Anonymize less sensitive data
                        results['records_to_anonymize'] += 1
                        if not dry_run:
                            await self._anonymize_data(
                                classification.entity_type,
                                classification.entity_id,
                                classification.field_name
                            )
                
                results['records_processed'] += 1
            
            return results
            
        finally:
            session.close()
    
    async def generate_compliance_report(
        self,
        standards: List[ComplianceStandard]
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'standards': [s.value for s in standards],
            'compliance_status': {},
            'findings': [],
            'recommendations': []
        }
        
        for standard in standards:
            if standard in self.policies:
                policy = self.policies[standard]
                status = await self._check_compliance(policy)
                report['compliance_status'][standard.value] = status
                
                # Add findings
                if status['score'] < 100:
                    for issue in status['issues']:
                        report['findings'].append({
                            'standard': standard.value,
                            'severity': issue['severity'],
                            'description': issue['description'],
                            'recommendation': issue['recommendation']
                        })
        
        # Overall compliance score
        scores = [s['score'] for s in report['compliance_status'].values()]
        report['overall_score'] = sum(scores) / len(scores) if scores else 0
        
        # Generate recommendations
        report['recommendations'] = self._generate_compliance_recommendations(report['findings'])
        
        return report
    
    async def _check_compliance(self, policy: SecurityPolicy) -> Dict[str, Any]:
        """Check compliance with a specific policy"""
        
        status = {
            'policy_id': policy.id,
            'score': 100,
            'issues': [],
            'last_checked': datetime.utcnow().isoformat()
        }
        
        # Check encryption requirements
        if 'encryption_required' in policy.rules:
            encryption_check = await self._check_encryption_compliance(policy.rules['encryption_required'])
            if not encryption_check['compliant']:
                status['score'] -= 20
                status['issues'].append({
                    'severity': 'high',
                    'description': 'Required fields not encrypted',
                    'fields': encryption_check['non_compliant_fields'],
                    'recommendation': 'Enable encryption for sensitive fields'
                })
        
        # Check retention compliance
        if 'data_retention' in policy.rules:
            retention_check = await self._check_retention_compliance(policy.rules['data_retention'])
            if not retention_check['compliant']:
                status['score'] -= 15
                status['issues'].append({
                    'severity': 'medium',
                    'description': 'Data retention policy violations',
                    'count': retention_check['violations'],
                    'recommendation': 'Run retention policy cleanup'
                })
        
        # Check access controls
        access_check = await self._check_access_control_compliance()
        if not access_check['compliant']:
            status['score'] -= 25
            status['issues'].append({
                'severity': 'high',
                'description': 'Insufficient access controls',
                'details': access_check['issues'],
                'recommendation': 'Review and update access control policies'
            })
        
        return status
    
    async def _check_encryption_compliance(self, required_fields: List[str]) -> Dict[str, Any]:
        """Check if required fields are encrypted"""
        
        # This would check actual data storage
        # For demo, returning sample result
        return {
            'compliant': True,
            'non_compliant_fields': []
        }
    
    async def _check_retention_compliance(self, retention_rules: Dict[str, int]) -> Dict[str, Any]:
        """Check data retention compliance"""
        
        session = self.Session()
        try:
            violations = 0
            
            for data_type, max_days in retention_rules.items():
                # Count records exceeding retention
                cutoff_date = datetime.utcnow() - timedelta(days=max_days)
                
                # This would check actual data
                # For demo, using audit trail as example
                old_records = session.query(AuditTrail).filter(
                    AuditTrail.timestamp < cutoff_date
                ).count()
                
                violations += old_records
            
            return {
                'compliant': violations == 0,
                'violations': violations
            }
            
        finally:
            session.close()
    
    async def _check_access_control_compliance(self) -> Dict[str, Any]:
        """Check access control compliance"""
        
        issues = []
        
        # Check for overly broad permissions
        session = self.Session()
        try:
            broad_permissions = session.query(AccessControl).filter(
                AccessControl.resource_id == '*'
            ).count()
            
            if broad_permissions > 0:
                issues.append(f"{broad_permissions} overly broad permissions found")
            
            # Check for expired permissions still active
            expired = session.query(AccessControl).filter(
                AccessControl.expires_at < datetime.utcnow()
            ).count()
            
            if expired > 0:
                issues.append(f"{expired} expired permissions still active")
            
            return {
                'compliant': len(issues) == 0,
                'issues': issues
            }
            
        finally:
            session.close()
    
    def _generate_compliance_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        # Group by severity
        high_severity = [f for f in findings if f['severity'] == 'high']
        medium_severity = [f for f in findings if f['severity'] == 'medium']
        
        if high_severity:
            recommendations.append(f"Address {len(high_severity)} high-severity compliance issues immediately")
        
        if medium_severity:
            recommendations.append(f"Schedule remediation for {len(medium_severity)} medium-severity issues")
        
        # Specific recommendations
        encryption_issues = [f for f in findings if 'encryption' in f['description'].lower()]
        if encryption_issues:
            recommendations.append("Implement field-level encryption for all PII data")
        
        retention_issues = [f for f in findings if 'retention' in f['description'].lower()]
        if retention_issues:
            recommendations.append("Configure automated data retention policies")
        
        return recommendations
    
    async def _delete_data(self, entity_type: str, entity_id: str, field_name: str):
        """Delete specific data field"""
        # Implementation would depend on data storage
        logger.info(f"Would delete {field_name} from {entity_type}/{entity_id}")
    
    async def _anonymize_data(self, entity_type: str, entity_id: str, field_name: str):
        """Anonymize specific data field"""
        # Implementation would depend on data storage
        logger.info(f"Would anonymize {field_name} from {entity_type}/{entity_id}")


class SecurityMonitor:
    """Real-time security monitoring and threat detection"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.threat_patterns = self._load_threat_patterns()
        self.user_baselines = {}
        self.redis_client = None
    
    def _load_threat_patterns(self) -> List[Dict[str, Any]]:
        """Load known threat patterns"""
        return [
            {
                'name': 'brute_force_attack',
                'pattern': {
                    'event_type': SecurityEventType.LOGIN_FAILED,
                    'threshold': 5,
                    'window': 300  # 5 minutes
                },
                'severity': 'high'
            },
            {
                'name': 'privilege_escalation_attempt',
                'pattern': {
                    'event_type': SecurityEventType.ACCESS_DENIED,
                    'followed_by': SecurityEventType.PRIVILEGE_ESCALATION,
                    'window': 600
                },
                'severity': 'critical'
            },
            {
                'name': 'data_exfiltration',
                'pattern': {
                    'event_type': SecurityEventType.DATA_EXPORT,
                    'threshold': 10,
                    'window': 3600,
                    'data_volume': 1000000  # bytes
                },
                'severity': 'critical'
            },
            {
                'name': 'api_abuse',
                'pattern': {
                    'event_type': SecurityEventType.RATE_LIMIT_EXCEEDED,
                    'threshold': 3,
                    'window': 3600
                },
                'severity': 'medium'
            }
        ]
    
    async def initialize(self):
        """Initialize security monitor"""
        self.redis_client = await redis.from_url('redis://localhost:6379')
        
        # Subscribe to security events
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe('security:events')
        
        # Start monitoring
        asyncio.create_task(self._monitor_events(pubsub))
    
    async def _monitor_events(self, pubsub):
        """Monitor security events in real-time"""
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    event = json.loads(message['data'])
                    await self._analyze_event(event)
                except Exception as e:
                    logger.error(f"Error processing security event: {e}")
    
    async def _analyze_event(self, event: Dict[str, Any]):
        """Analyze security event for threats"""
        
        # Check against threat patterns
        for pattern in self.threat_patterns:
            if await self._matches_pattern(event, pattern):
                await self._trigger_threat_alert(event, pattern)
        
        # Update user baseline
        if event.get('user_id'):
            await self._update_user_baseline(event['user_id'], event)
        
        # Anomaly detection
        if await self._is_anomalous(event):
            await self._trigger_anomaly_alert(event)
    
    async def _matches_pattern(self, event: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if event matches threat pattern"""
        
        pattern_def = pattern['pattern']
        
        # Simple event type match
        if 'event_type' in pattern_def:
            if event.get('event_type') != pattern_def['event_type'].value:
                return False
        
        # Threshold check
        if 'threshold' in pattern_def:
            key = f"threat:{pattern['name']}:{event.get('user_id', 'system')}"
            
            # Increment counter
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, pattern_def['window'])
            
            # Check threshold
            count = int(await self.redis_client.get(key) or 0)
            if count >= pattern_def['threshold']:
                return True
        
        return False
    
    async def _is_anomalous(self, event: Dict[str, Any]) -> bool:
        """Detect anomalous behavior"""
        
        user_id = event.get('user_id')
        if not user_id or user_id not in self.user_baselines:
            return False
        
        # Extract features
        features = self._extract_event_features(event)
        
        # Predict anomaly
        try:
            prediction = self.anomaly_detector.predict([features])
            return prediction[0] == -1  # -1 indicates anomaly
        except:
            return False
    
    def _extract_event_features(self, event: Dict[str, Any]) -> np.ndarray:
        """Extract features for anomaly detection"""
        
        # Simple feature extraction
        features = [
            hash(event.get('event_type', '')) % 1000,
            event.get('risk_score', 0),
            len(event.get('metadata', {})),
            datetime.fromisoformat(event.get('timestamp', datetime.utcnow().isoformat())).hour,
            datetime.fromisoformat(event.get('timestamp', datetime.utcnow().isoformat())).weekday()
        ]
        
        return np.array(features)
    
    async def _update_user_baseline(self, user_id: str, event: Dict[str, Any]):
        """Update user behavior baseline"""
        
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                'events': [],
                'last_updated': datetime.utcnow()
            }
        
        baseline = self.user_baselines[user_id]
        baseline['events'].append(self._extract_event_features(event))
        
        # Keep last 1000 events
        if len(baseline['events']) > 1000:
            baseline['events'] = baseline['events'][-1000:]
        
        # Retrain anomaly detector periodically
        if len(baseline['events']) > 100 and len(baseline['events']) % 100 == 0:
            self.anomaly_detector.fit(baseline['events'])
    
    async def _trigger_threat_alert(self, event: Dict[str, Any], pattern: Dict[str, Any]):
        """Trigger threat detection alert"""
        
        alert = {
            'type': 'threat_detected',
            'threat': pattern['name'],
            'severity': pattern['severity'],
            'event': event,
            'timestamp': datetime.utcnow().isoformat(),
            'recommended_action': self._get_recommended_action(pattern['name'])
        }
        
        await self.redis_client.publish('security:threats', json.dumps(alert))
        
        # Take automated action for critical threats
        if pattern['severity'] == 'critical':
            await self._take_automated_action(event, pattern)
    
    async def _trigger_anomaly_alert(self, event: Dict[str, Any]):
        """Trigger anomaly detection alert"""
        
        alert = {
            'type': 'anomaly_detected',
            'severity': 'medium',
            'event': event,
            'timestamp': datetime.utcnow().isoformat(),
            'description': 'Unusual behavior detected'
        }
        
        await self.redis_client.publish('security:anomalies', json.dumps(alert))
    
    def _get_recommended_action(self, threat_name: str) -> str:
        """Get recommended action for threat"""
        
        actions = {
            'brute_force_attack': 'Block IP address and enforce stronger authentication',
            'privilege_escalation_attempt': 'Revoke user permissions and investigate',
            'data_exfiltration': 'Suspend user account and audit all data access',
            'api_abuse': 'Apply stricter rate limits and review API usage'
        }
        
        return actions.get(threat_name, 'Investigate and take appropriate action')
    
    async def _take_automated_action(self, event: Dict[str, Any], pattern: Dict[str, Any]):
        """Take automated action for critical threats"""
        
        user_id = event.get('user_id')
        
        if pattern['name'] == 'data_exfiltration' and user_id:
            # Suspend user account
            await self.redis_client.setex(
                f"user:suspended:{user_id}",
                3600,  # 1 hour
                json.dumps({
                    'reason': 'Automated suspension due to data exfiltration threat',
                    'suspended_at': datetime.utcnow().isoformat()
                })
            )
            
            logger.warning(f"User {user_id} automatically suspended due to {pattern['name']}")


# Multi-factor authentication
class MFAManager:
    """Manages multi-factor authentication"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
    
    async def setup_totp(self, user_id: str, user_email: str) -> Dict[str, str]:
        """Setup TOTP for user"""
        
        # Generate secret
        secret = pyotp.random_base32()
        
        # Generate provisioning URI
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name='OpenPolicy Platform'
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Store encrypted secret
        # In production, this would be stored securely
        
        return {
            'secret': secret,
            'qr_code': f"data:image/png;base64,{qr_code_base64}",
            'provisioning_uri': provisioning_uri
        }
    
    async def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        
        # Get user's secret (would be retrieved from secure storage)
        secret = await self._get_user_secret(user_id)
        
        if not secret:
            return False
        
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    
    async def _get_user_secret(self, user_id: str) -> Optional[str]:
        """Get user's TOTP secret"""
        # In production, this would retrieve from secure storage
        return None


# API Key Management
class APIKeyManager:
    """Manages API keys with rotation and policies"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.encryption_manager = EncryptionManager()
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create new API key"""
        
        # Generate secure key
        key_id = f"key_{secrets.token_hex(8)}"
        key_secret = secrets.token_urlsafe(32)
        
        # Hash the secret for storage
        hashed_secret = self.encryption_manager.hash_sensitive_data(key_secret)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Store key metadata
        # In production, this would be saved to database
        
        return {
            'key_id': key_id,
            'key_secret': key_secret,  # Only returned once
            'name': name,
            'permissions': permissions,
            'expires_at': expires_at.isoformat() if expires_at else None,
            'rate_limit': rate_limit
        }
    
    async def validate_api_key(self, key_id: str, key_secret: str) -> Optional[Dict[str, Any]]:
        """Validate API key"""
        
        # Retrieve key metadata
        # In production, this would fetch from database
        
        # Verify secret
        # Check expiration
        # Check rate limits
        
        return None
    
    async def rotate_api_key(self, key_id: str, user_id: str) -> Dict[str, Any]:
        """Rotate API key"""
        
        # Get existing key metadata
        # Create new secret
        # Update database
        # Return new credentials
        
        return {}


# Example usage
async def security_demo():
    """Demo security features"""
    
    # Initialize components
    access_manager = AccessControlManager('postgresql://user:pass@localhost/security')
    audit_logger = AuditLogger('postgresql://user:pass@localhost/security')
    await audit_logger.initialize()
    
    compliance_manager = ComplianceManager('postgresql://user:pass@localhost/security')
    security_monitor = SecurityMonitor()
    await security_monitor.initialize()
    
    # Example: Log security event
    event_id = await audit_logger.log_event(
        SecurityEventType.LOGIN_SUCCESS,
        user_id='user123',
        ip_address='192.168.1.100',
        user_agent='Mozilla/5.0...'
    )
    
    print(f"Logged security event: {event_id}")
    
    # Example: Check permission
    allowed, reason = await access_manager.check_permission(
        user_id='user123',
        resource_type='scraper',
        resource_id='scraper_001',
        action='read'
    )
    
    print(f"Permission check: {allowed} - {reason}")
    
    # Example: Generate compliance report
    report = await compliance_manager.generate_compliance_report([
        ComplianceStandard.GDPR,
        ComplianceStandard.PIPEDA
    ])
    
    print(f"Compliance score: {report['overall_score']}%")
    
    # Example: Setup MFA
    mfa_manager = MFAManager('postgresql://user:pass@localhost/security')
    mfa_setup = await mfa_manager.setup_totp('user123', 'user@example.com')
    
    print(f"MFA Secret: {mfa_setup['secret']}")


if __name__ == "__main__":
    asyncio.run(security_demo())