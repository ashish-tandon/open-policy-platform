"""
MCP Scraper Management System - 40by6 Implementation
Comprehensive system for managing 1700+ scrapers
"""

import asyncio
import json
import logging
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import aiohttp
import aiofiles
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import importlib.util
import sys
import traceback

logger = logging.getLogger(__name__)


class ScraperCategory(Enum):
    """Scraper categories based on jurisdiction"""
    FEDERAL_PARLIAMENT = "federal_parliament"
    FEDERAL_ELECTIONS = "federal_elections"
    FEDERAL_COMMITTEES = "federal_committees"
    PROVINCIAL_LEGISLATURE = "provincial_legislature"
    PROVINCIAL_ELECTIONS = "provincial_elections"
    MUNICIPAL_COUNCIL = "municipal_council"
    MUNICIPAL_COMMITTEES = "municipal_committees"
    CIVIC_PLATFORM = "civic_platform"
    THIRD_PARTY_API = "third_party_api"
    CUSTOM_SCRAPER = "custom_scraper"


class ScraperPlatform(Enum):
    """Supported scraping platforms"""
    LEGISTAR = "legistar"
    CIVIC_PLUS = "civic_plus"
    CIVIC_CLERK = "civic_clerk"
    GRANICUS = "granicus"
    PRIMEGOV = "primegov"
    OPENPARLIAMENT = "openparliament"
    REPRESENT_API = "represent_api"
    CUSTOM = "custom"


class ScraperStatus(Enum):
    """Scraper status states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"
    TESTING = "testing"


class DataIngestionStatus(Enum):
    """Data ingestion status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ScraperMetadata:
    """Metadata for each scraper"""
    id: str
    name: str
    category: ScraperCategory
    platform: ScraperPlatform
    jurisdiction: Dict[str, str]  # {type: federal/provincial/municipal, code: ca/on/toronto}
    url: str
    module_path: str
    class_name: str
    schedule: str  # cron expression
    priority: int = 5  # 1-10, higher is more important
    timeout: int = 300  # seconds
    retry_count: int = 3
    rate_limit: Dict[str, Any] = field(default_factory=dict)  # requests per time unit
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    failure_count: int = 0
    status: ScraperStatus = ScraperStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScraperRun:
    """Information about a scraper execution"""
    run_id: str
    scraper_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    records_scraped: int = 0
    records_ingested: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    data_quality_score: float = 0.0


class ScraperRegistry:
    """Registry for all scrapers in the system"""
    
    def __init__(self, base_path: str = "scrapers"):
        self.base_path = Path(base_path)
        self.scrapers: Dict[str, ScraperMetadata] = {}
        self.category_index: Dict[ScraperCategory, Set[str]] = {cat: set() for cat in ScraperCategory}
        self.platform_index: Dict[ScraperPlatform, Set[str]] = {plat: set() for plat in ScraperPlatform}
        self.jurisdiction_index: Dict[str, Set[str]] = {}
        
    async def discover_scrapers(self) -> int:
        """Discover all scrapers in the filesystem"""
        logger.info(f"Discovering scrapers in {self.base_path}")
        discovered = 0
        
        # Define scraper discovery patterns
        patterns = {
            ScraperCategory.FEDERAL_PARLIAMENT: ["federal/parliament/**/*.py", "federal/openparliament/**/*.py"],
            ScraperCategory.FEDERAL_ELECTIONS: ["federal/elections/**/*.py"],
            ScraperCategory.FEDERAL_COMMITTEES: ["federal/committees/**/*.py"],
            ScraperCategory.PROVINCIAL_LEGISLATURE: ["provincial/*/legislature/**/*.py"],
            ScraperCategory.PROVINCIAL_ELECTIONS: ["provincial/*/elections/**/*.py"],
            ScraperCategory.MUNICIPAL_COUNCIL: ["municipal/*/*/council/**/*.py", "scrapers-ca/ca_*/*.py"],
            ScraperCategory.MUNICIPAL_COMMITTEES: ["municipal/*/*/committees/**/*.py"],
            ScraperCategory.CIVIC_PLATFORM: ["civic/**/*.py"],
        }
        
        for category, file_patterns in patterns.items():
            for pattern in file_patterns:
                for file_path in self.base_path.glob(pattern):
                    if file_path.name.startswith("_") or file_path.name == "setup.py":
                        continue
                    
                    try:
                        metadata = await self._extract_scraper_metadata(file_path, category)
                        if metadata:
                            self.register_scraper(metadata)
                            discovered += 1
                    except Exception as e:
                        logger.error(f"Error discovering scraper {file_path}: {e}")
        
        logger.info(f"Discovered {discovered} scrapers")
        return discovered
    
    async def _extract_scraper_metadata(self, file_path: Path, category: ScraperCategory) -> Optional[ScraperMetadata]:
        """Extract metadata from a scraper file"""
        try:
            # Read the file to look for scraper class
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            
            # Look for common scraper patterns
            if any(pattern in content for pattern in ["class.*Scraper", "PersonScraper", "BillScraper", "VoteScraper"]):
                # Extract jurisdiction from path
                parts = file_path.parts
                jurisdiction = self._extract_jurisdiction(parts, category)
                
                # Generate unique ID
                scraper_id = hashlib.md5(str(file_path).encode()).hexdigest()[:12]
                
                # Determine platform
                platform = self._detect_platform(content, file_path)
                
                # Extract class name
                import re
                class_match = re.search(r'class\s+(\w+Scraper)', content)
                class_name = class_match.group(1) if class_match else "UnknownScraper"
                
                # Create metadata
                metadata = ScraperMetadata(
                    id=scraper_id,
                    name=f"{jurisdiction.get('name', 'Unknown')} - {class_name}",
                    category=category,
                    platform=platform,
                    jurisdiction=jurisdiction,
                    url=self._extract_url(content),
                    module_path=str(file_path.relative_to(self.base_path)),
                    class_name=class_name,
                    schedule=self._determine_schedule(category, jurisdiction),
                    tags=self._extract_tags(content, file_path),
                )
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return None
    
    def _extract_jurisdiction(self, parts: Tuple[str, ...], category: ScraperCategory) -> Dict[str, str]:
        """Extract jurisdiction information from file path"""
        jurisdiction = {"type": "unknown", "code": "unknown", "name": "Unknown"}
        
        if category in [ScraperCategory.FEDERAL_PARLIAMENT, ScraperCategory.FEDERAL_ELECTIONS]:
            jurisdiction = {"type": "federal", "code": "ca", "name": "Canada"}
        elif category in [ScraperCategory.PROVINCIAL_LEGISLATURE, ScraperCategory.PROVINCIAL_ELECTIONS]:
            # Extract province from path like provincial/on/legislature
            for i, part in enumerate(parts):
                if part == "provincial" and i + 1 < len(parts):
                    province_code = parts[i + 1]
                    jurisdiction = {
                        "type": "provincial",
                        "code": province_code,
                        "name": self._get_province_name(province_code)
                    }
                    break
        elif category in [ScraperCategory.MUNICIPAL_COUNCIL, ScraperCategory.MUNICIPAL_COMMITTEES]:
            # Extract municipality from path
            for i, part in enumerate(parts):
                if part == "municipal" and i + 2 < len(parts):
                    province_code = parts[i + 1]
                    city = parts[i + 2]
                    jurisdiction = {
                        "type": "municipal",
                        "code": f"{province_code}/{city}",
                        "name": f"{city.title()}, {self._get_province_name(province_code)}"
                    }
                    break
                elif part.startswith("ca_") and "_" in part[3:]:
                    # Handle scrapers-ca format like ca_on_toronto
                    parts_split = part.split("_")
                    if len(parts_split) >= 3:
                        province_code = parts_split[1]
                        city = "_".join(parts_split[2:])
                        jurisdiction = {
                            "type": "municipal",
                            "code": f"{province_code}/{city}",
                            "name": f"{city.replace('_', ' ').title()}, {self._get_province_name(province_code)}"
                        }
        
        return jurisdiction
    
    def _get_province_name(self, code: str) -> str:
        """Get province name from code"""
        provinces = {
            "ab": "Alberta", "bc": "British Columbia", "mb": "Manitoba",
            "nb": "New Brunswick", "nl": "Newfoundland and Labrador",
            "ns": "Nova Scotia", "nt": "Northwest Territories", "nu": "Nunavut",
            "on": "Ontario", "pe": "Prince Edward Island", "qc": "Quebec",
            "sk": "Saskatchewan", "yt": "Yukon"
        }
        return provinces.get(code.lower(), code.upper())
    
    def _detect_platform(self, content: str, file_path: Path) -> ScraperPlatform:
        """Detect the scraping platform from code content"""
        platform_indicators = {
            ScraperPlatform.LEGISTAR: ["legistar", "LegistarScraper"],
            ScraperPlatform.CIVIC_PLUS: ["civicplus", "CivicPlusScraper"],
            ScraperPlatform.GRANICUS: ["granicus", "GranicusScraper"],
            ScraperPlatform.OPENPARLIAMENT: ["openparliament", "parl.gc.ca"],
            ScraperPlatform.REPRESENT_API: ["represent.opennorth.ca"],
        }
        
        for platform, indicators in platform_indicators.items():
            if any(indicator in content.lower() for indicator in indicators):
                return platform
        
        return ScraperPlatform.CUSTOM
    
    def _extract_url(self, content: str) -> str:
        """Extract base URL from scraper code"""
        import re
        url_patterns = [
            r'url\s*=\s*["\']([^"\']+)["\']',
            r'base_url\s*=\s*["\']([^"\']+)["\']',
            r'https?://[^\s\'"]+',
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        return ""
    
    def _determine_schedule(self, category: ScraperCategory, jurisdiction: Dict[str, str]) -> str:
        """Determine appropriate schedule for scraper"""
        # Daily schedules for active legislatures
        if category in [ScraperCategory.FEDERAL_PARLIAMENT, ScraperCategory.PROVINCIAL_LEGISLATURE]:
            return "0 2 * * *"  # 2 AM daily
        # Weekly for municipal councils
        elif category in [ScraperCategory.MUNICIPAL_COUNCIL, ScraperCategory.MUNICIPAL_COMMITTEES]:
            return "0 3 * * 1"  # 3 AM Monday
        # Monthly for elections
        elif category in [ScraperCategory.FEDERAL_ELECTIONS, ScraperCategory.PROVINCIAL_ELECTIONS]:
            return "0 4 1 * *"  # 4 AM first of month
        else:
            return "0 5 * * *"  # 5 AM daily default
    
    def _extract_tags(self, content: str, file_path: Path) -> List[str]:
        """Extract relevant tags from scraper"""
        tags = []
        
        # Check for data types
        if "PersonScraper" in content or "representatives" in content.lower():
            tags.append("representatives")
        if "BillScraper" in content or "bills" in content.lower():
            tags.append("bills")
        if "VoteScraper" in content or "votes" in content.lower():
            tags.append("votes")
        if "CommitteeScraper" in content or "committees" in content.lower():
            tags.append("committees")
        if "EventScraper" in content or "events" in content.lower():
            tags.append("events")
        
        return tags
    
    def register_scraper(self, metadata: ScraperMetadata):
        """Register a scraper in the registry"""
        self.scrapers[metadata.id] = metadata
        self.category_index[metadata.category].add(metadata.id)
        self.platform_index[metadata.platform].add(metadata.id)
        
        # Index by jurisdiction
        jur_key = f"{metadata.jurisdiction['type']}:{metadata.jurisdiction['code']}"
        if jur_key not in self.jurisdiction_index:
            self.jurisdiction_index[jur_key] = set()
        self.jurisdiction_index[jur_key].add(metadata.id)
    
    def get_scrapers_by_category(self, category: ScraperCategory) -> List[ScraperMetadata]:
        """Get all scrapers in a category"""
        return [self.scrapers[sid] for sid in self.category_index.get(category, set())]
    
    def get_scrapers_by_platform(self, platform: ScraperPlatform) -> List[ScraperMetadata]:
        """Get all scrapers for a platform"""
        return [self.scrapers[sid] for sid in self.platform_index.get(platform, set())]
    
    def get_scrapers_by_jurisdiction(self, jurisdiction_type: str, jurisdiction_code: str) -> List[ScraperMetadata]:
        """Get all scrapers for a jurisdiction"""
        jur_key = f"{jurisdiction_type}:{jurisdiction_code}"
        return [self.scrapers[sid] for sid in self.jurisdiction_index.get(jur_key, set())]
    
    def export_registry(self, format: str = "json") -> str:
        """Export the registry to a file"""
        data = {
            "metadata": {
                "total_scrapers": len(self.scrapers),
                "categories": {cat.value: len(ids) for cat, ids in self.category_index.items()},
                "platforms": {plat.value: len(ids) for plat, ids in self.platform_index.items()},
                "jurisdictions": {jur: len(ids) for jur, ids in self.jurisdiction_index.items()},
                "exported_at": datetime.utcnow().isoformat()
            },
            "scrapers": [scraper.__dict__ for scraper in self.scrapers.values()]
        }
        
        output_path = Path("reports/scraper_registry.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return str(output_path)


class ScraperOrchestrator:
    """Orchestrate execution of multiple scrapers"""
    
    def __init__(self, registry: ScraperRegistry, max_concurrent: int = 10):
        self.registry = registry
        self.max_concurrent = max_concurrent
        self.running_scrapers: Dict[str, ScraperRun] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.redis_client = None
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
    
    async def initialize(self):
        """Initialize the orchestrator"""
        self.redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
    
    async def schedule_scrapers(self, scraper_ids: Optional[List[str]] = None):
        """Schedule scrapers for execution"""
        if scraper_ids is None:
            # Schedule all active scrapers
            scraper_ids = [
                sid for sid, meta in self.registry.scrapers.items()
                if meta.status == ScraperStatus.ACTIVE
            ]
        
        # Sort by priority
        scrapers = sorted(
            [self.registry.scrapers[sid] for sid in scraper_ids],
            key=lambda x: x.priority,
            reverse=True
        )
        
        for scraper in scrapers:
            await self.queue.put(scraper)
        
        logger.info(f"Scheduled {len(scrapers)} scrapers for execution")
    
    async def run_orchestration(self):
        """Main orchestration loop"""
        workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_concurrent)
        ]
        
        await asyncio.gather(*workers)
    
    async def _worker(self, worker_id: str):
        """Worker to process scrapers from queue"""
        while True:
            try:
                scraper = await self.queue.get()
                await self._execute_scraper(scraper, worker_id)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def _execute_scraper(self, scraper: ScraperMetadata, worker_id: str):
        """Execute a single scraper"""
        run_id = f"{scraper.id}_{datetime.utcnow().timestamp()}"
        run = ScraperRun(
            run_id=run_id,
            scraper_id=scraper.id,
            start_time=datetime.utcnow()
        )
        
        self.running_scrapers[run_id] = run
        
        try:
            logger.info(f"Worker {worker_id} executing scraper {scraper.name}")
            
            # Load and execute scraper
            scraper_instance = await self._load_scraper(scraper)
            if scraper_instance:
                # Run with timeout
                result = await asyncio.wait_for(
                    self._run_scraper_instance(scraper_instance, scraper),
                    timeout=scraper.timeout
                )
                
                run.records_scraped = result.get("records_scraped", 0)
                run.status = "completed"
                
                # Update scraper metadata
                scraper.last_run = datetime.utcnow()
                scraper.last_success = datetime.utcnow()
                scraper.failure_count = 0
            else:
                raise Exception("Failed to load scraper")
                
        except asyncio.TimeoutError:
            logger.error(f"Scraper {scraper.name} timed out after {scraper.timeout}s")
            run.status = "timeout"
            run.errors.append({"type": "timeout", "message": f"Exceeded {scraper.timeout}s"})
            scraper.failure_count += 1
        except Exception as e:
            logger.error(f"Scraper {scraper.name} failed: {e}")
            run.status = "failed"
            run.errors.append({"type": "exception", "message": str(e), "traceback": traceback.format_exc()})
            scraper.failure_count += 1
        finally:
            run.end_time = datetime.utcnow()
            del self.running_scrapers[run_id]
            
            # Save run to database
            await self._save_run(run)
            
            # Update scraper status if too many failures
            if scraper.failure_count >= 5:
                scraper.status = ScraperStatus.FAILED
                logger.warning(f"Scraper {scraper.name} marked as FAILED after {scraper.failure_count} failures")
    
    async def _load_scraper(self, metadata: ScraperMetadata) -> Optional[Any]:
        """Dynamically load a scraper module"""
        try:
            module_path = self.registry.base_path / metadata.module_path
            spec = importlib.util.spec_from_file_location(metadata.id, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[metadata.id] = module
            spec.loader.exec_module(module)
            
            # Get the scraper class
            scraper_class = getattr(module, metadata.class_name, None)
            if scraper_class:
                return scraper_class(metadata.config)
            
            return None
        except Exception as e:
            logger.error(f"Failed to load scraper {metadata.module_path}: {e}")
            return None
    
    async def _run_scraper_instance(self, scraper_instance: Any, metadata: ScraperMetadata) -> Dict[str, Any]:
        """Run a scraper instance"""
        # This would be customized based on the scraper type
        # For now, a generic implementation
        result = {"records_scraped": 0}
        
        # Check if scraper has async run method
        if hasattr(scraper_instance, "run"):
            if asyncio.iscoroutinefunction(scraper_instance.run):
                data = await scraper_instance.run()
            else:
                # Run sync scrapers in thread pool
                data = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, scraper_instance.run
                )
            
            result["records_scraped"] = len(data) if isinstance(data, list) else 1
            
            # Send to ingestion pipeline
            await self._ingest_data(metadata, data)
        
        return result
    
    async def _ingest_data(self, metadata: ScraperMetadata, data: Any):
        """Send scraped data to ingestion pipeline"""
        ingestion_msg = {
            "scraper_id": metadata.id,
            "scraper_name": metadata.name,
            "jurisdiction": metadata.jurisdiction,
            "category": metadata.category.value,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Push to Redis queue for processing
        await self.redis_client.lpush("scraper:ingestion:queue", json.dumps(ingestion_msg, default=str))
    
    async def _save_run(self, run: ScraperRun):
        """Save scraper run to database"""
        # This would save to the database
        pass
    
    def get_running_scrapers(self) -> List[ScraperRun]:
        """Get currently running scrapers"""
        return list(self.running_scrapers.values())
    
    async def stop_scraper(self, run_id: str) -> bool:
        """Stop a running scraper"""
        # Implementation for stopping a scraper
        pass


class DataIngestionPipeline:
    """Pipeline for ingesting scraped data into the database"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.redis_client = None
        self.processors = {}
        self._setup_processors()
    
    def _setup_processors(self):
        """Setup data processors for different data types"""
        self.processors = {
            "representatives": self._process_representatives,
            "bills": self._process_bills,
            "votes": self._process_votes,
            "committees": self._process_committees,
            "events": self._process_events,
        }
    
    async def initialize(self):
        """Initialize the ingestion pipeline"""
        self.redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
    
    async def run_ingestion_loop(self):
        """Main ingestion processing loop"""
        while True:
            try:
                # Get message from queue
                msg = await self.redis_client.brpop("scraper:ingestion:queue", timeout=5)
                if msg:
                    _, data = msg
                    await self._process_ingestion(json.loads(data))
            except Exception as e:
                logger.error(f"Ingestion error: {e}")
                await asyncio.sleep(1)
    
    async def _process_ingestion(self, msg: Dict[str, Any]):
        """Process a single ingestion message"""
        scraper_id = msg["scraper_id"]
        data = msg["data"]
        
        logger.info(f"Processing ingestion from scraper {msg['scraper_name']}")
        
        # Determine data type and process
        data_type = self._detect_data_type(data)
        if data_type in self.processors:
            processor = self.processors[data_type]
            result = await processor(data, msg)
            
            # Log ingestion result
            await self._log_ingestion_result(scraper_id, result)
        else:
            logger.warning(f"Unknown data type for scraper {scraper_id}")
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect the type of data being ingested"""
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            if isinstance(sample, dict):
                # Check for common fields
                if "name" in sample and ("email" in sample or "role" in sample):
                    return "representatives"
                elif "title" in sample and "introduced_date" in sample:
                    return "bills"
                elif "vote_id" in sample or "yeas" in sample:
                    return "votes"
                elif "committee_name" in sample:
                    return "committees"
                elif "event_date" in sample or "agenda" in sample:
                    return "events"
        
        return "unknown"
    
    async def _process_representatives(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Process representative data"""
        processed = 0
        errors = []
        
        with self.Session() as session:
            for rep_data in data:
                try:
                    # Validate and clean data
                    cleaned = self._clean_representative_data(rep_data)
                    
                    # Check for existing record
                    existing = session.execute(
                        text("SELECT id FROM representatives WHERE email = :email"),
                        {"email": cleaned.get("email")}
                    ).first()
                    
                    if existing:
                        # Update existing
                        session.execute(
                            text("""
                                UPDATE representatives 
                                SET name = :name, role = :role, updated_at = NOW()
                                WHERE email = :email
                            """),
                            cleaned
                        )
                    else:
                        # Insert new
                        session.execute(
                            text("""
                                INSERT INTO representatives (name, email, role, jurisdiction_id)
                                VALUES (:name, :email, :role, :jurisdiction_id)
                            """),
                            cleaned
                        )
                    
                    processed += 1
                    
                except Exception as e:
                    errors.append({"data": rep_data, "error": str(e)})
            
            session.commit()
        
        return {
            "processed": processed,
            "errors": len(errors),
            "error_details": errors[:10]  # Limit error details
        }
    
    def _clean_representative_data(self, data: Dict) -> Dict:
        """Clean and validate representative data"""
        return {
            "name": data.get("name", "").strip(),
            "email": data.get("email", "").lower().strip(),
            "role": data.get("role", "").strip(),
            "jurisdiction_id": data.get("jurisdiction_id", "")
        }
    
    async def _process_bills(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Process bill data"""
        # Implementation similar to representatives
        pass
    
    async def _process_votes(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Process vote data"""
        # Implementation
        pass
    
    async def _process_committees(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Process committee data"""
        # Implementation
        pass
    
    async def _process_events(self, data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Process event data"""
        # Implementation
        pass
    
    async def _log_ingestion_result(self, scraper_id: str, result: Dict[str, Any]):
        """Log the result of data ingestion"""
        log_entry = {
            "scraper_id": scraper_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processed": result.get("processed", 0),
            "errors": result.get("errors", 0),
            "status": "success" if result.get("errors", 0) == 0 else "partial"
        }
        
        # Save to Redis for monitoring
        await self.redis_client.lpush("scraper:ingestion:results", json.dumps(log_entry))
        
        # Keep only last 1000 results
        await self.redis_client.ltrim("scraper:ingestion:results", 0, 999)


class ScraperMonitor:
    """Monitor scraper health and performance"""
    
    def __init__(self, registry: ScraperRegistry):
        self.registry = registry
        self.redis_client = None
        self.metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_records": 0,
            "average_duration": 0,
            "scrapers_by_status": {}
        }
    
    async def initialize(self):
        """Initialize the monitor"""
        self.redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
    
    async def update_metrics(self):
        """Update monitoring metrics"""
        # Count scrapers by status
        status_counts = {}
        for status in ScraperStatus:
            count = sum(1 for s in self.registry.scrapers.values() if s.status == status)
            status_counts[status.value] = count
        
        self.metrics["scrapers_by_status"] = status_counts
        self.metrics["total_scrapers"] = len(self.registry.scrapers)
        
        # Get recent run statistics from Redis
        recent_results = await self.redis_client.lrange("scraper:ingestion:results", 0, 99)
        if recent_results:
            results = [json.loads(r) for r in recent_results]
            successful = sum(1 for r in results if r["status"] == "success")
            self.metrics["recent_success_rate"] = successful / len(results)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get overall system health report"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics,
            "alerts": self._check_alerts(),
            "recommendations": self._generate_recommendations()
        }
    
    def _check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        # Check for high failure rate
        if self.metrics.get("recent_success_rate", 1) < 0.8:
            alerts.append({
                "level": "warning",
                "message": f"High failure rate: {(1 - self.metrics.get('recent_success_rate', 1)) * 100:.1f}%"
            })
        
        # Check for too many failed scrapers
        failed_count = self.metrics["scrapers_by_status"].get("failed", 0)
        if failed_count > 50:
            alerts.append({
                "level": "critical",
                "message": f"{failed_count} scrapers in failed state"
            })
        
        return alerts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if self.metrics["scrapers_by_status"].get("inactive", 0) > 100:
            recommendations.append("Review and remove inactive scrapers")
        
        if self.metrics.get("average_duration", 0) > 300:
            recommendations.append("Optimize slow-running scrapers")
        
        return recommendations


# Main MCP Scraper Management System
class MCPScraperManagementSystem:
    """Main system coordinating all scraper components"""
    
    def __init__(self):
        self.registry = ScraperRegistry()
        self.orchestrator = ScraperOrchestrator(self.registry)
        self.ingestion = DataIngestionPipeline()
        self.monitor = ScraperMonitor(self.registry)
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing MCP Scraper Management System...")
        
        # Discover all scrapers
        await self.registry.discover_scrapers()
        
        # Initialize components
        await self.orchestrator.initialize()
        await self.ingestion.initialize()
        await self.monitor.initialize()
        
        logger.info(f"System initialized with {len(self.registry.scrapers)} scrapers")
    
    async def run(self):
        """Run the complete system"""
        # Start all components
        tasks = [
            asyncio.create_task(self.orchestrator.run_orchestration()),
            asyncio.create_task(self.ingestion.run_ingestion_loop()),
            asyncio.create_task(self._monitor_loop()),
            asyncio.create_task(self._scheduler_loop()),
        ]
        
        await asyncio.gather(*tasks)
    
    async def _monitor_loop(self):
        """Monitoring loop"""
        while True:
            await self.monitor.update_metrics()
            await asyncio.sleep(60)  # Update every minute
    
    async def _scheduler_loop(self):
        """Scheduler loop for running scrapers on schedule"""
        while True:
            # Check for scrapers that need to run based on schedule
            now = datetime.utcnow()
            due_scrapers = []
            
            for scraper in self.registry.scrapers.values():
                if scraper.status == ScraperStatus.ACTIVE:
                    # Simple schedule check (would use croniter in production)
                    if scraper.last_run is None or (now - scraper.last_run).seconds > 3600:
                        due_scrapers.append(scraper.id)
            
            if due_scrapers:
                await self.orchestrator.schedule_scrapers(due_scrapers)
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "registry": {
                "total_scrapers": len(self.registry.scrapers),
                "by_category": {cat.value: len(ids) for cat, ids in self.registry.category_index.items()},
                "by_platform": {plat.value: len(ids) for plat, ids in self.registry.platform_index.items()},
            },
            "orchestrator": {
                "running": len(self.orchestrator.running_scrapers),
                "queued": self.orchestrator.queue.qsize()
            },
            "monitor": self.monitor.get_health_report()
        }


if __name__ == "__main__":
    # Run the scraper management system
    async def main():
        system = MCPScraperManagementSystem()
        await system.initialize()
        
        # Export registry
        registry_path = system.registry.export_registry()
        logger.info(f"Scraper registry exported to {registry_path}")
        
        # Get status
        status = system.get_status()
        logger.info(f"System status: {json.dumps(status, indent=2)}")
        
        # Run the system
        await system.run()
    
    asyncio.run(main())