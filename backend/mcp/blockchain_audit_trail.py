"""
Blockchain Audit Trail System - 40by6
Immutable audit logging with blockchain technology
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
import uuid
from collections import defaultdict
import asyncpg
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, JSON, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import web3
from web3 import Web3
from eth_account import Account
import ipfshttpclient
from merkletools import MerkleTools
import redis
from prometheus_client import Counter, Histogram, Gauge
import msgpack
import lz4.frame
from typing_extensions import Protocol

logger = logging.getLogger(__name__)

# Metrics
blocks_created = Counter('blockchain_blocks_created_total', 'Total blocks created')
blocks_validated = Counter('blockchain_blocks_validated_total', 'Total blocks validated')
audit_entries = Counter('blockchain_audit_entries_total', 'Total audit entries', ['event_type', 'severity'])
chain_height = Gauge('blockchain_chain_height', 'Current blockchain height')
validation_errors = Counter('blockchain_validation_errors_total', 'Total validation errors')
consensus_time = Histogram('blockchain_consensus_duration_seconds', 'Time to reach consensus')

Base = declarative_base()


class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication & Authorization
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_CHANGED = "role_changed"
    
    # Data Operations
    DATA_ACCESS = "data_access"
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # System Operations
    CONFIG_CHANGE = "config_change"
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    SERVICE_RESTART = "service_restart"
    DEPLOYMENT = "deployment"
    BACKUP = "backup"
    RESTORE = "restore"
    
    # Security Events
    SECURITY_SCAN = "security_scan"
    THREAT_DETECTED = "threat_detected"
    VULNERABILITY_FOUND = "vulnerability_found"
    ENCRYPTION_KEY_ROTATION = "encryption_key_rotation"
    CERTIFICATE_RENEWAL = "certificate_renewal"
    
    # Compliance
    COMPLIANCE_CHECK = "compliance_check"
    POLICY_VIOLATION = "policy_violation"
    AUDIT_REQUEST = "audit_request"
    EVIDENCE_COLLECTION = "evidence_collection"
    
    # Scraper Operations
    SCRAPER_RUN = "scraper_run"
    SCRAPER_SUCCESS = "scraper_success"
    SCRAPER_FAILURE = "scraper_failure"
    SCRAPER_CONFIG_CHANGE = "scraper_config_change"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ConsensusAlgorithm(Enum):
    """Blockchain consensus algorithms"""
    PROOF_OF_WORK = "pow"
    PROOF_OF_AUTHORITY = "poa"
    PRACTICAL_BYZANTINE_FAULT_TOLERANCE = "pbft"
    RAFT = "raft"


@dataclass
class AuditEntry:
    """Individual audit log entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType = AuditEventType.DATA_ACCESS
    severity: AuditSeverity = AuditSeverity.INFO
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: str = ""
    result: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    compliance_frameworks: Set[str] = field(default_factory=set)  # GDPR, HIPAA, SOC2, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "tags": list(self.tags),
            "compliance_frameworks": list(self.compliance_frameworks)
        }
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of entry"""
        # Create deterministic JSON representation
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class Block:
    """Blockchain block"""
    index: int
    timestamp: datetime
    entries: List[AuditEntry]
    previous_hash: str
    nonce: int = 0
    merkle_root: Optional[str] = None
    validator: Optional[str] = None
    signature: Optional[str] = None
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle tree root of entries"""
        if not self.entries:
            return hashlib.sha256(b"empty").hexdigest()
        
        mt = MerkleTools()
        for entry in self.entries:
            mt.add_leaf(entry.calculate_hash())
        
        mt.make_tree()
        return mt.get_merkle_root()
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "merkle_root": self.merkle_root or self.calculate_merkle_root(),
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "validator": self.validator
        }
        
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "entries": [entry.to_dict() for entry in self.entries],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "merkle_root": self.merkle_root,
            "hash": self.calculate_hash(),
            "validator": self.validator,
            "signature": self.signature,
            "entry_count": len(self.entries)
        }


class BlockchainStorage(Base):
    """Database storage for blockchain"""
    __tablename__ = 'blockchain_blocks'
    
    id = Column(Integer, primary_key=True)
    index = Column(Integer, unique=True, nullable=False)
    hash = Column(String(64), unique=True, nullable=False)
    previous_hash = Column(String(64), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    merkle_root = Column(String(64))
    nonce = Column(Integer, default=0)
    validator = Column(String(255))
    signature = Column(Text)
    entries = Column(JSON)
    entry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_blockchain_index', 'index'),
        Index('idx_blockchain_hash', 'hash'),
        Index('idx_blockchain_timestamp', 'timestamp'),
    )


class AuditEntryStorage(Base):
    """Database storage for audit entries"""
    __tablename__ = 'audit_entries'
    
    id = Column(String(36), primary_key=True)
    block_index = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    user_id = Column(String(255))
    user_email = Column(String(255))
    ip_address = Column(String(45))
    resource_type = Column(String(100))
    resource_id = Column(String(255))
    action = Column(String(255))
    result = Column(String(50))
    details = Column(JSON)
    tags = Column(JSON)
    compliance_frameworks = Column(JSON)
    entry_hash = Column(String(64), unique=True, nullable=False)
    
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_event_type', 'event_type'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_block', 'block_index'),
    )


class CryptoManager:
    """Cryptographic operations manager"""
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self._load_or_generate_keys()
    
    def _load_or_generate_keys(self):
        """Load or generate RSA key pair"""
        try:
            # Try to load existing keys
            with open('blockchain_private.pem', 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            
            with open('blockchain_public.pem', 'rb') as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )
        except FileNotFoundError:
            # Generate new keys
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.public_key = self.private_key.public_key()
            
            # Save keys
            self._save_keys()
    
    def _save_keys(self):
        """Save RSA keys to files"""
        # Save private key
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open('blockchain_private.pem', 'wb') as f:
            f.write(private_pem)
        
        # Save public key
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        with open('blockchain_public.pem', 'wb') as f:
            f.write(public_pem)
    
    def sign_data(self, data: bytes) -> str:
        """Sign data with private key"""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature.hex()
    
    def verify_signature(self, data: bytes, signature: str, public_key_pem: str) -> bool:
        """Verify signature with public key"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode(),
                backend=default_backend()
            )
            
            public_key.verify(
                bytes.fromhex(signature),
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False


class ConsensusEngine:
    """Blockchain consensus engine"""
    
    def __init__(self, algorithm: ConsensusAlgorithm = ConsensusAlgorithm.PROOF_OF_AUTHORITY):
        self.algorithm = algorithm
        self.validators = set()
        self.pending_blocks = []
        self.votes = defaultdict(lambda: defaultdict(int))
    
    async def add_validator(self, validator_id: str, public_key: str):
        """Add authorized validator"""
        self.validators.add(validator_id)
        logger.info(f"Added validator: {validator_id}")
    
    async def propose_block(self, block: Block, proposer_id: str) -> bool:
        """Propose new block for consensus"""
        if self.algorithm == ConsensusAlgorithm.PROOF_OF_AUTHORITY:
            return await self._poa_consensus(block, proposer_id)
        elif self.algorithm == ConsensusAlgorithm.PROOF_OF_WORK:
            return await self._pow_consensus(block)
        elif self.algorithm == ConsensusAlgorithm.PRACTICAL_BYZANTINE_FAULT_TOLERANCE:
            return await self._pbft_consensus(block, proposer_id)
        else:
            raise ValueError(f"Unsupported consensus algorithm: {self.algorithm}")
    
    async def _poa_consensus(self, block: Block, proposer_id: str) -> bool:
        """Proof of Authority consensus"""
        with consensus_time.time():
            if proposer_id not in self.validators:
                logger.warning(f"Unauthorized validator: {proposer_id}")
                return False
            
            # In PoA, authorized validators can directly add blocks
            block.validator = proposer_id
            return True
    
    async def _pow_consensus(self, block: Block, difficulty: int = 4) -> bool:
        """Proof of Work consensus"""
        with consensus_time.time():
            target = "0" * difficulty
            
            while True:
                block_hash = block.calculate_hash()
                if block_hash.startswith(target):
                    logger.info(f"Found valid nonce: {block.nonce}")
                    return True
                
                block.nonce += 1
                
                # Prevent infinite loop
                if block.nonce > 1000000:
                    return False
    
    async def _pbft_consensus(self, block: Block, proposer_id: str) -> bool:
        """Practical Byzantine Fault Tolerance consensus"""
        with consensus_time.time():
            block_hash = block.calculate_hash()
            
            # Collect votes from validators
            self.votes[block_hash][proposer_id] = 1
            
            # Wait for votes (simplified)
            await asyncio.sleep(1)
            
            # Check if we have enough votes (2/3 + 1)
            total_votes = sum(self.votes[block_hash].values())
            required_votes = (2 * len(self.validators) // 3) + 1
            
            if total_votes >= required_votes:
                logger.info(f"Consensus reached for block {block.index}")
                return True
            
            return False


class BlockchainAuditSystem:
    """Main blockchain audit system"""
    
    def __init__(self, database_url: str, ipfs_api: Optional[str] = None):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        
        self.crypto = CryptoManager()
        self.consensus = ConsensusEngine()
        self.redis = redis.Redis(decode_responses=True)
        
        # IPFS client for distributed storage
        self.ipfs = None
        if ipfs_api:
            try:
                self.ipfs = ipfshttpclient.connect(ipfs_api)
            except:
                logger.warning("IPFS connection failed, using local storage only")
        
        # Ethereum integration for anchoring
        self.web3 = None
        self.contract = None
        self._setup_ethereum()
        
        self.pending_entries: List[AuditEntry] = []
        self.current_chain: List[Block] = []
        self._load_chain()
        
        # Start background tasks
        asyncio.create_task(self._block_creation_task())
        asyncio.create_task(self._chain_validation_task())
    
    def _setup_ethereum(self):
        """Setup Ethereum integration"""
        eth_node = os.getenv('ETH_NODE_URL')
        if eth_node:
            try:
                self.web3 = Web3(Web3.HTTPProvider(eth_node))
                
                # Load smart contract
                contract_address = os.getenv('AUDIT_CONTRACT_ADDRESS')
                contract_abi = json.loads(os.getenv('AUDIT_CONTRACT_ABI', '[]'))
                
                if contract_address and contract_abi:
                    self.contract = self.web3.eth.contract(
                        address=contract_address,
                        abi=contract_abi
                    )
                    logger.info("Ethereum integration initialized")
            except Exception as e:
                logger.warning(f"Ethereum setup failed: {e}")
    
    def _load_chain(self):
        """Load blockchain from storage"""
        session = self.Session()
        try:
            blocks = session.query(BlockchainStorage).order_by(
                BlockchainStorage.index
            ).all()
            
            for block_data in blocks:
                entries = []
                for entry_dict in block_data.entries:
                    entry = AuditEntry(**{
                        k: v for k, v in entry_dict.items()
                        if k not in ['tags', 'compliance_frameworks']
                    })
                    entry.tags = set(entry_dict.get('tags', []))
                    entry.compliance_frameworks = set(
                        entry_dict.get('compliance_frameworks', [])
                    )
                    entries.append(entry)
                
                block = Block(
                    index=block_data.index,
                    timestamp=block_data.timestamp,
                    entries=entries,
                    previous_hash=block_data.previous_hash,
                    nonce=block_data.nonce,
                    merkle_root=block_data.merkle_root,
                    validator=block_data.validator,
                    signature=block_data.signature
                )
                
                self.current_chain.append(block)
            
            chain_height.set(len(self.current_chain))
            logger.info(f"Loaded blockchain with {len(self.current_chain)} blocks")
            
        finally:
            session.close()
    
    async def add_audit_entry(self, entry: AuditEntry) -> str:
        """Add audit entry to pending queue"""
        
        # Validate entry
        if not entry.user_id and not entry.resource_id:
            raise ValueError("Audit entry must have user_id or resource_id")
        
        # Add compliance framework tags based on event
        if entry.event_type in [
            AuditEventType.DATA_ACCESS,
            AuditEventType.DATA_DELETE,
            AuditEventType.DATA_EXPORT
        ]:
            entry.compliance_frameworks.add("GDPR")
        
        if entry.resource_type == "health_data":
            entry.compliance_frameworks.add("HIPAA")
        
        if entry.event_type in [
            AuditEventType.LOGIN,
            AuditEventType.PERMISSION_GRANTED,
            AuditEventType.CONFIG_CHANGE
        ]:
            entry.compliance_frameworks.add("SOC2")
        
        # Add to pending queue
        self.pending_entries.append(entry)
        
        # Update metrics
        audit_entries.labels(
            entry.event_type.value,
            entry.severity.value
        ).inc()
        
        # Store in Redis for quick access
        entry_data = msgpack.packb(entry.to_dict())
        self.redis.zadd(
            'pending_audit_entries',
            {entry.id: time.time()}
        )
        self.redis.hset('audit_entries', entry.id, entry_data)
        
        logger.info(f"Added audit entry: {entry.id}")
        return entry.id
    
    async def create_block(self) -> Optional[Block]:
        """Create new block from pending entries"""
        
        if not self.pending_entries:
            return None
        
        # Get entries for block (max 1000)
        entries = self.pending_entries[:1000]
        self.pending_entries = self.pending_entries[1000:]
        
        # Create block
        previous_hash = "0" * 64
        if self.current_chain:
            previous_hash = self.current_chain[-1].calculate_hash()
        
        block = Block(
            index=len(self.current_chain),
            timestamp=datetime.now(timezone.utc),
            entries=entries,
            previous_hash=previous_hash
        )
        
        # Calculate Merkle root
        block.merkle_root = block.calculate_merkle_root()
        
        # Get consensus
        validator_id = os.getenv('VALIDATOR_ID', 'default_validator')
        if await self.consensus.propose_block(block, validator_id):
            # Sign block
            block_data = json.dumps({
                "index": block.index,
                "merkle_root": block.merkle_root,
                "previous_hash": block.previous_hash
            })
            block.signature = self.crypto.sign_data(block_data.encode())
            
            # Add to chain
            self.current_chain.append(block)
            
            # Persist block
            await self._persist_block(block)
            
            # Anchor to Ethereum if available
            if self.contract:
                await self._anchor_to_ethereum(block)
            
            # Store in IPFS if available
            if self.ipfs:
                await self._store_in_ipfs(block)
            
            # Update metrics
            blocks_created.inc()
            chain_height.set(len(self.current_chain))
            
            logger.info(f"Created block {block.index} with {len(entries)} entries")
            return block
        
        else:
            # Return entries to pending queue
            self.pending_entries = entries + self.pending_entries
            return None
    
    async def _persist_block(self, block: Block):
        """Persist block to database"""
        session = self.Session()
        try:
            # Save block
            block_storage = BlockchainStorage(
                index=block.index,
                hash=block.calculate_hash(),
                previous_hash=block.previous_hash,
                timestamp=block.timestamp,
                merkle_root=block.merkle_root,
                nonce=block.nonce,
                validator=block.validator,
                signature=block.signature,
                entries=[entry.to_dict() for entry in block.entries],
                entry_count=len(block.entries)
            )
            session.add(block_storage)
            
            # Save individual entries
            for entry in block.entries:
                entry_storage = AuditEntryStorage(
                    id=entry.id,
                    block_index=block.index,
                    timestamp=entry.timestamp,
                    event_type=entry.event_type.value,
                    severity=entry.severity.value,
                    user_id=entry.user_id,
                    user_email=entry.user_email,
                    ip_address=entry.ip_address,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    action=entry.action,
                    result=entry.result,
                    details=entry.details,
                    tags=list(entry.tags),
                    compliance_frameworks=list(entry.compliance_frameworks),
                    entry_hash=entry.calculate_hash()
                )
                session.add(entry_storage)
                
                # Remove from Redis
                self.redis.zrem('pending_audit_entries', entry.id)
                self.redis.hdel('audit_entries', entry.id)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to persist block: {e}")
            raise
        finally:
            session.close()
    
    async def _anchor_to_ethereum(self, block: Block):
        """Anchor block hash to Ethereum blockchain"""
        try:
            if not self.web3 or not self.contract:
                return
            
            block_hash = block.calculate_hash()
            
            # Prepare transaction
            account = Account.from_key(os.getenv('ETH_PRIVATE_KEY'))
            nonce = self.web3.eth.get_transaction_count(account.address)
            
            # Call smart contract function
            transaction = self.contract.functions.anchorBlock(
                block.index,
                block_hash,
                block.merkle_root
            ).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': 100000,
                'gasPrice': self.web3.toWei('20', 'gwei')
            })
            
            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(
                transaction,
                private_key=os.getenv('ETH_PRIVATE_KEY')
            )
            
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            logger.info(f"Anchored block {block.index} to Ethereum: {tx_hash.hex()}")
            
        except Exception as e:
            logger.error(f"Failed to anchor to Ethereum: {e}")
    
    async def _store_in_ipfs(self, block: Block):
        """Store block in IPFS"""
        try:
            if not self.ipfs:
                return
            
            # Prepare block data
            block_data = json.dumps(block.to_dict(), indent=2)
            
            # Add to IPFS
            result = self.ipfs.add_json(block.to_dict())
            ipfs_hash = result['Hash']
            
            # Pin the content
            self.ipfs.pin.add(ipfs_hash)
            
            # Store IPFS hash reference
            self.redis.hset('block_ipfs_hashes', str(block.index), ipfs_hash)
            
            logger.info(f"Stored block {block.index} in IPFS: {ipfs_hash}")
            
        except Exception as e:
            logger.error(f"Failed to store in IPFS: {e}")
    
    async def validate_chain(self) -> bool:
        """Validate entire blockchain"""
        
        if not self.current_chain:
            return True
        
        # Check genesis block
        if self.current_chain[0].previous_hash != "0" * 64:
            logger.error("Invalid genesis block")
            validation_errors.inc()
            return False
        
        # Validate each block
        for i in range(1, len(self.current_chain)):
            current_block = self.current_chain[i]
            previous_block = self.current_chain[i - 1]
            
            # Check hash linkage
            if current_block.previous_hash != previous_block.calculate_hash():
                logger.error(f"Invalid hash link at block {i}")
                validation_errors.inc()
                return False
            
            # Verify Merkle root
            calculated_root = current_block.calculate_merkle_root()
            if current_block.merkle_root != calculated_root:
                logger.error(f"Invalid Merkle root at block {i}")
                validation_errors.inc()
                return False
            
            # Verify signature if present
            if current_block.signature and current_block.validator:
                # In production, fetch validator's public key from registry
                # Here we use our own key for demo
                public_key_pem = self.crypto.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode()
                
                block_data = json.dumps({
                    "index": current_block.index,
                    "merkle_root": current_block.merkle_root,
                    "previous_hash": current_block.previous_hash
                })
                
                if not self.crypto.verify_signature(
                    block_data.encode(),
                    current_block.signature,
                    public_key_pem
                ):
                    logger.error(f"Invalid signature at block {i}")
                    validation_errors.inc()
                    return False
        
        blocks_validated.inc(len(self.current_chain))
        logger.info("Blockchain validation successful")
        return True
    
    async def query_audit_trail(
        self,
        filters: Dict[str, Any],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Query audit trail with filters"""
        
        session = self.Session()
        try:
            query = session.query(AuditEntryStorage)
            
            # Apply filters
            if 'user_id' in filters:
                query = query.filter(AuditEntryStorage.user_id == filters['user_id'])
            
            if 'event_type' in filters:
                query = query.filter(AuditEntryStorage.event_type == filters['event_type'])
            
            if 'resource_type' in filters:
                query = query.filter(
                    AuditEntryStorage.resource_type == filters['resource_type']
                )
            
            if 'resource_id' in filters:
                query = query.filter(
                    AuditEntryStorage.resource_id == filters['resource_id']
                )
            
            if start_time:
                query = query.filter(AuditEntryStorage.timestamp >= start_time)
            
            if end_time:
                query = query.filter(AuditEntryStorage.timestamp <= end_time)
            
            # Order by timestamp descending
            query = query.order_by(AuditEntryStorage.timestamp.desc())
            
            # Apply limit
            query = query.limit(limit)
            
            # Execute query
            results = query.all()
            
            # Convert to AuditEntry objects
            entries = []
            for result in results:
                entry = AuditEntry(
                    id=result.id,
                    timestamp=result.timestamp,
                    event_type=AuditEventType(result.event_type),
                    severity=AuditSeverity(result.severity),
                    user_id=result.user_id,
                    user_email=result.user_email,
                    ip_address=result.ip_address,
                    resource_type=result.resource_type,
                    resource_id=result.resource_id,
                    action=result.action,
                    result=result.result,
                    details=result.details or {},
                    tags=set(result.tags or []),
                    compliance_frameworks=set(result.compliance_frameworks or [])
                )
                entries.append(entry)
            
            return entries
            
        finally:
            session.close()
    
    async def generate_compliance_report(
        self,
        framework: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        
        session = self.Session()
        try:
            # Query relevant entries
            query = session.query(AuditEntryStorage).filter(
                AuditEntryStorage.compliance_frameworks.contains([framework]),
                AuditEntryStorage.timestamp >= start_time,
                AuditEntryStorage.timestamp <= end_time
            )
            
            entries = query.all()
            
            # Generate report
            report = {
                "framework": framework,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "total_events": len(entries),
                "events_by_type": defaultdict(int),
                "events_by_severity": defaultdict(int),
                "users_involved": set(),
                "resources_accessed": defaultdict(set),
                "compliance_violations": [],
                "key_activities": []
            }
            
            for entry in entries:
                report["events_by_type"][entry.event_type] += 1
                report["events_by_severity"][entry.severity] += 1
                
                if entry.user_id:
                    report["users_involved"].add(entry.user_id)
                
                if entry.resource_type and entry.resource_id:
                    report["resources_accessed"][entry.resource_type].add(
                        entry.resource_id
                    )
                
                # Check for violations
                if entry.result != "success" or entry.severity in ["error", "critical"]:
                    report["compliance_violations"].append({
                        "timestamp": entry.timestamp.isoformat(),
                        "event": entry.event_type,
                        "user": entry.user_id,
                        "details": entry.details
                    })
            
            # Convert sets to lists for JSON serialization
            report["users_involved"] = list(report["users_involved"])
            for resource_type in report["resources_accessed"]:
                report["resources_accessed"][resource_type] = list(
                    report["resources_accessed"][resource_type]
                )
            
            # Add blockchain verification
            report["blockchain_verification"] = {
                "total_blocks": len(self.current_chain),
                "chain_valid": await self.validate_chain(),
                "last_block_hash": self.current_chain[-1].calculate_hash() if self.current_chain else None
            }
            
            return report
            
        finally:
            session.close()
    
    async def export_audit_data(
        self,
        format: str = "json",
        filters: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Export audit data in various formats"""
        
        # Query data
        entries = await self.query_audit_trail(filters or {}, limit=10000)
        
        if format == "json":
            data = {
                "export_date": datetime.utcnow().isoformat(),
                "total_entries": len(entries),
                "entries": [entry.to_dict() for entry in entries],
                "blockchain_info": {
                    "chain_length": len(self.current_chain),
                    "last_block_hash": self.current_chain[-1].calculate_hash() if self.current_chain else None
                }
            }
            return json.dumps(data, indent=2).encode()
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    'id', 'timestamp', 'event_type', 'severity',
                    'user_id', 'user_email', 'ip_address',
                    'resource_type', 'resource_id', 'action', 'result'
                ]
            )
            
            writer.writeheader()
            for entry in entries:
                writer.writerow({
                    'id': entry.id,
                    'timestamp': entry.timestamp.isoformat(),
                    'event_type': entry.event_type.value,
                    'severity': entry.severity.value,
                    'user_id': entry.user_id,
                    'user_email': entry.user_email,
                    'ip_address': entry.ip_address,
                    'resource_type': entry.resource_type,
                    'resource_id': entry.resource_id,
                    'action': entry.action,
                    'result': entry.result
                })
            
            return output.getvalue().encode()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _block_creation_task(self):
        """Background task for creating blocks"""
        while True:
            try:
                # Wait for minimum entries or timeout
                await asyncio.sleep(60)  # Create block every minute
                
                if len(self.pending_entries) >= 10:  # Minimum 10 entries
                    block = await self.create_block()
                    if block:
                        logger.info(f"Created block {block.index}")
                
            except Exception as e:
                logger.error(f"Block creation error: {e}")
    
    async def _chain_validation_task(self):
        """Background task for validating chain"""
        while True:
            try:
                await asyncio.sleep(300)  # Validate every 5 minutes
                
                is_valid = await self.validate_chain()
                if not is_valid:
                    logger.critical("Blockchain validation failed!")
                    # In production, trigger alerts
                
            except Exception as e:
                logger.error(f"Chain validation error: {e}")


# Example usage
async def audit_demo():
    """Demo audit functionality"""
    
    # Initialize audit system
    audit_system = BlockchainAuditSystem(
        'postgresql://user:pass@localhost/audit_db',
        ipfs_api='/ip4/127.0.0.1/tcp/5001'
    )
    
    # Add audit entries
    entry1 = AuditEntry(
        event_type=AuditEventType.LOGIN,
        severity=AuditSeverity.INFO,
        user_id="user123",
        user_email="user@example.com",
        ip_address="192.168.1.100",
        action="User login",
        result="success",
        details={"method": "password", "mfa": True}
    )
    
    await audit_system.add_audit_entry(entry1)
    
    entry2 = AuditEntry(
        event_type=AuditEventType.DATA_ACCESS,
        severity=AuditSeverity.WARNING,
        user_id="user456",
        resource_type="scraper_data",
        resource_id="scraper_789",
        action="Accessed sensitive data",
        result="success",
        details={"data_size": 1024, "export_format": "json"}
    )
    
    await audit_system.add_audit_entry(entry2)
    
    # Create block manually (normally done by background task)
    block = await audit_system.create_block()
    if block:
        print(f"Created block: {json.dumps(block.to_dict(), indent=2)}")
    
    # Query audit trail
    entries = await audit_system.query_audit_trail(
        filters={"user_id": "user123"},
        limit=10
    )
    
    print(f"Found {len(entries)} audit entries")
    
    # Generate compliance report
    report = await audit_system.generate_compliance_report(
        "GDPR",
        datetime.utcnow() - timedelta(days=30),
        datetime.utcnow()
    )
    
    print(f"Compliance report: {json.dumps(report, indent=2)}")
    
    # Validate blockchain
    is_valid = await audit_system.validate_chain()
    print(f"Blockchain valid: {is_valid}")


if __name__ == "__main__":
    import os
    os.environ['VALIDATOR_ID'] = 'demo_validator'
    asyncio.run(audit_demo())