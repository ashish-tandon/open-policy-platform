"""
Scraper Marketplace and Template System - 40by6
Share, discover, and deploy scraper templates
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
import hashlib
import uuid
from pathlib import Path
import git
import docker
import jinja2
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON, Boolean, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import aiohttp
import boto3
from minio import Minio
import redis.asyncio as redis

logger = logging.getLogger(__name__)

Base = declarative_base()


class TemplateStatus(Enum):
    """Template status in marketplace"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    FEATURED = "featured"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class TemplateCategory(Enum):
    """Categories for scraper templates"""
    GOVERNMENT = "government"
    MUNICIPAL = "municipal"
    LEGISLATIVE = "legislative"
    ELECTIONS = "elections"
    PUBLIC_RECORDS = "public_records"
    COMMITTEES = "committees"
    BUDGETS = "budgets"
    MEETINGS = "meetings"
    DOCUMENTS = "documents"
    CUSTOM = "custom"


@dataclass
class ScraperTemplate:
    """Represents a scraper template"""
    id: str
    name: str
    description: str
    category: TemplateCategory
    platform: str  # legistar, civic_plus, custom, etc.
    author: str
    version: str
    status: TemplateStatus
    config_schema: Dict[str, Any]
    code_template: str
    docker_template: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    rating: float = 0.0
    downloads: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'platform': self.platform,
            'author': self.author,
            'version': self.version,
            'status': self.status.value,
            'config_schema': self.config_schema,
            'tags': self.tags,
            'rating': self.rating,
            'downloads': self.downloads,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


# SQLAlchemy models
class TemplateModel(Base):
    __tablename__ = 'scraper_templates'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    category = Column(String)
    platform = Column(String)
    author = Column(String)
    version = Column(String)
    status = Column(String)
    config_schema = Column(JSON)
    code_template = Column(String)
    docker_template = Column(String)
    requirements = Column(JSON)
    tags = Column(JSON)
    rating = Column(Float, default=0.0)
    downloads = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    reviews = relationship("TemplateReview", back_populates="template")
    deployments = relationship("TemplateDeployment", back_populates="template")


class TemplateReview(Base):
    __tablename__ = 'template_reviews'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    template_id = Column(String, ForeignKey('scraper_templates.id'))
    user_id = Column(String)
    rating = Column(Integer)  # 1-5
    comment = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    template = relationship("TemplateModel", back_populates="reviews")


class TemplateDeployment(Base):
    __tablename__ = 'template_deployments'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    template_id = Column(String, ForeignKey('scraper_templates.id'))
    scraper_id = Column(String)
    deployed_by = Column(String)
    config = Column(JSON)
    status = Column(String)
    deployed_at = Column(DateTime, default=datetime.utcnow)
    
    template = relationship("TemplateModel", back_populates="deployments")


class TemplateBuilder:
    """Builds scraper templates from existing scrapers"""
    
    def __init__(self):
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates/'),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    async def create_template_from_scraper(
        self,
        scraper_path: str,
        metadata: Dict[str, Any]
    ) -> ScraperTemplate:
        """Create a template from an existing scraper"""
        
        # Read scraper code
        with open(scraper_path, 'r') as f:
            scraper_code = f.read()
        
        # Extract configuration points
        config_schema = self._extract_config_schema(scraper_code)
        
        # Templatize the code
        template_code = self._templatize_code(scraper_code, config_schema)
        
        # Extract requirements
        requirements = self._extract_requirements(scraper_path)
        
        # Create Dockerfile template if needed
        docker_template = self._create_docker_template(metadata.get('platform'))
        
        # Generate template ID
        template_id = f"tpl_{metadata['platform']}_{uuid.uuid4().hex[:8]}"
        
        template = ScraperTemplate(
            id=template_id,
            name=metadata['name'],
            description=metadata.get('description', ''),
            category=TemplateCategory(metadata.get('category', 'custom')),
            platform=metadata['platform'],
            author=metadata.get('author', 'system'),
            version='1.0.0',
            status=TemplateStatus.DRAFT,
            config_schema=config_schema,
            code_template=template_code,
            docker_template=docker_template,
            requirements=requirements,
            tags=metadata.get('tags', [])
        )
        
        return template
    
    def _extract_config_schema(self, code: str) -> Dict[str, Any]:
        """Extract configuration schema from code"""
        schema = {
            'type': 'object',
            'properties': {},
            'required': []
        }
        
        # Look for configuration variables
        import re
        
        # Find environment variables
        env_vars = re.findall(r'os\.getenv\(["\'](\w+)["\']\)', code)
        for var in env_vars:
            schema['properties'][var.lower()] = {
                'type': 'string',
                'description': f'{var} configuration',
                'env_var': var
            }
        
        # Find URL patterns
        urls = re.findall(r'["\']https?://[^"\']+["\']', code)
        if urls:
            schema['properties']['base_url'] = {
                'type': 'string',
                'description': 'Base URL for the scraper',
                'default': urls[0].strip('"\'')
            }
            schema['required'].append('base_url')
        
        # Find selectors
        selectors = re.findall(r'select\(["\']([^"\']+)["\']\)', code)
        if selectors:
            schema['properties']['selectors'] = {
                'type': 'object',
                'description': 'CSS selectors for data extraction',
                'properties': {
                    f'selector_{i}': {'type': 'string', 'default': sel}
                    for i, sel in enumerate(selectors[:5])  # Limit to 5
                }
            }
        
        return schema
    
    def _templatize_code(self, code: str, schema: Dict[str, Any]) -> str:
        """Convert code to template with placeholders"""
        template_code = code
        
        # Replace hardcoded values with template variables
        for prop, config in schema.get('properties', {}).items():
            if 'default' in config and config['default'] in template_code:
                template_code = template_code.replace(
                    config['default'],
                    f"{{{{ config.{prop} }}}}"
                )
        
        # Add template header
        header = """# Scraper Template - Auto-generated
# Configure using the provided schema
{% for key, value in config.items() %}
# {{ key }}: {{ value }}
{% endfor %}

"""
        
        return header + template_code
    
    def _extract_requirements(self, scraper_path: str) -> List[str]:
        """Extract Python requirements"""
        requirements = set()
        
        # Common scraping libraries
        with open(scraper_path, 'r') as f:
            code = f.read()
            
        if 'beautifulsoup' in code.lower():
            requirements.add('beautifulsoup4>=4.9.0')
        if 'requests' in code:
            requirements.add('requests>=2.25.0')
        if 'selenium' in code:
            requirements.add('selenium>=3.141.0')
        if 'scrapy' in code:
            requirements.add('scrapy>=2.5.0')
        if 'pandas' in code:
            requirements.add('pandas>=1.2.0')
        
        # Always include these
        requirements.update([
            'aiohttp>=3.8.0',
            'pydantic>=1.8.0',
            'python-dateutil>=2.8.0'
        ])
        
        return sorted(list(requirements))
    
    def _create_docker_template(self, platform: str) -> str:
        """Create Dockerfile template"""
        return f"""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scraper code
COPY . .

# Set environment variables
ENV PLATFORM={platform}
ENV PYTHONUNBUFFERED=1

# Run scraper
CMD ["python", "-m", "scraper"]
"""


class ScraperMarketplace:
    """Main marketplace for scraper templates"""
    
    def __init__(self, database_url: str, storage_config: Dict[str, Any]):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Storage backend (S3-compatible)
        self.storage = Minio(
            storage_config.get('endpoint', 'localhost:9000'),
            access_key=storage_config.get('access_key'),
            secret_key=storage_config.get('secret_key'),
            secure=storage_config.get('secure', False)
        )
        self.bucket_name = storage_config.get('bucket', 'scraper-templates')
        
        # Template builder
        self.builder = TemplateBuilder()
        
        # Cache
        self.redis_client = None
        
        # Initialize storage
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure storage bucket exists"""
        try:
            if not self.storage.bucket_exists(self.bucket_name):
                self.storage.make_bucket(self.bucket_name)
        except Exception as e:
            logger.error(f"Error creating bucket: {e}")
    
    async def initialize(self):
        """Initialize marketplace"""
        self.redis_client = await redis.from_url('redis://localhost:6379')
        logger.info("Scraper marketplace initialized")
    
    async def publish_template(self, template: ScraperTemplate) -> bool:
        """Publish a template to the marketplace"""
        session = self.Session()
        
        try:
            # Validate template
            if not self._validate_template(template):
                return False
            
            # Store template code in object storage
            code_key = f"templates/{template.id}/code.py"
            self.storage.put_object(
                self.bucket_name,
                code_key,
                template.code_template.encode(),
                len(template.code_template)
            )
            
            # Store Dockerfile if present
            if template.docker_template:
                docker_key = f"templates/{template.id}/Dockerfile"
                self.storage.put_object(
                    self.bucket_name,
                    docker_key,
                    template.docker_template.encode(),
                    len(template.docker_template)
                )
            
            # Save to database
            db_template = TemplateModel(
                id=template.id,
                name=template.name,
                description=template.description,
                category=template.category.value,
                platform=template.platform,
                author=template.author,
                version=template.version,
                status=template.status.value,
                config_schema=template.config_schema,
                code_template=code_key,
                docker_template=f"templates/{template.id}/Dockerfile" if template.docker_template else None,
                requirements=template.requirements,
                tags=template.tags
            )
            
            session.add(db_template)
            session.commit()
            
            # Clear cache
            await self._clear_cache_for_category(template.category)
            
            logger.info(f"Published template: {template.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing template: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    async def search_templates(
        self,
        query: Optional[str] = None,
        category: Optional[TemplateCategory] = None,
        platform: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_rating: float = 0.0,
        status: Optional[TemplateStatus] = TemplateStatus.APPROVED,
        limit: int = 50,
        offset: int = 0
    ) -> List[ScraperTemplate]:
        """Search for templates in the marketplace"""
        
        # Check cache first
        cache_key = f"templates:search:{query}:{category}:{platform}:{tags}:{min_rating}:{status}:{limit}:{offset}"
        cached = await self.redis_client.get(cache_key)
        if cached:
            return [ScraperTemplate(**t) for t in json.loads(cached)]
        
        session = self.Session()
        
        try:
            # Build query
            query_obj = session.query(TemplateModel)
            
            if status:
                query_obj = query_obj.filter(TemplateModel.status == status.value)
            
            if category:
                query_obj = query_obj.filter(TemplateModel.category == category.value)
            
            if platform:
                query_obj = query_obj.filter(TemplateModel.platform == platform)
            
            if min_rating > 0:
                query_obj = query_obj.filter(TemplateModel.rating >= min_rating)
            
            if query:
                search_term = f"%{query}%"
                query_obj = query_obj.filter(
                    (TemplateModel.name.ilike(search_term)) |
                    (TemplateModel.description.ilike(search_term))
                )
            
            # TODO: Add tag filtering with JSON
            
            # Order by rating and downloads
            query_obj = query_obj.order_by(
                TemplateModel.rating.desc(),
                TemplateModel.downloads.desc()
            )
            
            # Pagination
            results = query_obj.offset(offset).limit(limit).all()
            
            # Convert to template objects
            templates = []
            for result in results:
                template = ScraperTemplate(
                    id=result.id,
                    name=result.name,
                    description=result.description,
                    category=TemplateCategory(result.category),
                    platform=result.platform,
                    author=result.author,
                    version=result.version,
                    status=TemplateStatus(result.status),
                    config_schema=result.config_schema,
                    code_template='',  # Don't load code in search results
                    requirements=result.requirements,
                    tags=result.tags,
                    rating=result.rating,
                    downloads=result.downloads,
                    created_at=result.created_at,
                    updated_at=result.updated_at
                )
                templates.append(template)
            
            # Cache results
            await self.redis_client.setex(
                cache_key,
                300,  # 5 minutes
                json.dumps([t.to_dict() for t in templates])
            )
            
            return templates
            
        finally:
            session.close()
    
    async def get_template(self, template_id: str) -> Optional[ScraperTemplate]:
        """Get a specific template with full details"""
        session = self.Session()
        
        try:
            result = session.query(TemplateModel).filter_by(id=template_id).first()
            
            if not result:
                return None
            
            # Load code from storage
            code_template = ''
            try:
                response = self.storage.get_object(self.bucket_name, result.code_template)
                code_template = response.read().decode()
            except Exception as e:
                logger.error(f"Error loading template code: {e}")
            
            # Load Dockerfile if present
            docker_template = None
            if result.docker_template:
                try:
                    response = self.storage.get_object(self.bucket_name, result.docker_template)
                    docker_template = response.read().decode()
                except Exception as e:
                    logger.error(f"Error loading Dockerfile: {e}")
            
            template = ScraperTemplate(
                id=result.id,
                name=result.name,
                description=result.description,
                category=TemplateCategory(result.category),
                platform=result.platform,
                author=result.author,
                version=result.version,
                status=TemplateStatus(result.status),
                config_schema=result.config_schema,
                code_template=code_template,
                docker_template=docker_template,
                requirements=result.requirements,
                tags=result.tags,
                rating=result.rating,
                downloads=result.downloads,
                created_at=result.created_at,
                updated_at=result.updated_at
            )
            
            return template
            
        finally:
            session.close()
    
    async def deploy_template(
        self,
        template_id: str,
        config: Dict[str, Any],
        target_id: str,
        deployed_by: str
    ) -> Optional[str]:
        """Deploy a template as a new scraper"""
        
        # Get template
        template = await self.get_template(template_id)
        if not template:
            logger.error(f"Template not found: {template_id}")
            return None
        
        # Validate configuration against schema
        if not self._validate_config(config, template.config_schema):
            logger.error("Invalid configuration for template")
            return None
        
        # Generate scraper code from template
        scraper_code = self._generate_scraper_code(template, config)
        
        # Create deployment directory
        deployment_id = f"scraper_{target_id}_{uuid.uuid4().hex[:8]}"
        deployment_path = Path(f"deployments/{deployment_id}")
        deployment_path.mkdir(parents=True, exist_ok=True)
        
        # Write scraper code
        scraper_file = deployment_path / "scraper.py"
        with open(scraper_file, 'w') as f:
            f.write(scraper_code)
        
        # Write requirements
        requirements_file = deployment_path / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(template.requirements))
        
        # Write Dockerfile if present
        if template.docker_template:
            dockerfile = deployment_path / "Dockerfile"
            with open(dockerfile, 'w') as f:
                f.write(template.docker_template)
        
        # Write configuration
        config_file = deployment_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Deploy using Docker
        try:
            if await self._deploy_with_docker(deployment_path, deployment_id):
                # Record deployment
                session = self.Session()
                try:
                    deployment = TemplateDeployment(
                        template_id=template_id,
                        scraper_id=deployment_id,
                        deployed_by=deployed_by,
                        config=config,
                        status='deployed'
                    )
                    session.add(deployment)
                    
                    # Increment download count
                    template_model = session.query(TemplateModel).filter_by(id=template_id).first()
                    if template_model:
                        template_model.downloads += 1
                    
                    session.commit()
                    
                    logger.info(f"Successfully deployed template {template_id} as {deployment_id}")
                    return deployment_id
                    
                finally:
                    session.close()
            
        except Exception as e:
            logger.error(f"Error deploying template: {e}")
            
        return None
    
    async def rate_template(
        self,
        template_id: str,
        user_id: str,
        rating: int,
        comment: Optional[str] = None
    ) -> bool:
        """Rate a template"""
        
        if not 1 <= rating <= 5:
            return False
        
        session = self.Session()
        
        try:
            # Check if user already reviewed
            existing = session.query(TemplateReview).filter_by(
                template_id=template_id,
                user_id=user_id
            ).first()
            
            if existing:
                # Update existing review
                existing.rating = rating
                existing.comment = comment
                existing.created_at = datetime.utcnow()
            else:
                # Create new review
                review = TemplateReview(
                    template_id=template_id,
                    user_id=user_id,
                    rating=rating,
                    comment=comment
                )
                session.add(review)
            
            # Update template rating
            template = session.query(TemplateModel).filter_by(id=template_id).first()
            if template:
                # Calculate new average rating
                reviews = session.query(TemplateReview).filter_by(template_id=template_id).all()
                if reviews:
                    avg_rating = sum(r.rating for r in reviews) / len(reviews)
                    template.rating = avg_rating
            
            session.commit()
            
            # Clear cache
            await self._clear_cache_for_template(template_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error rating template: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    async def get_featured_templates(self, limit: int = 10) -> List[ScraperTemplate]:
        """Get featured templates"""
        return await self.search_templates(
            status=TemplateStatus.FEATURED,
            limit=limit
        )
    
    async def get_trending_templates(self, days: int = 7, limit: int = 10) -> List[ScraperTemplate]:
        """Get trending templates based on recent downloads"""
        session = self.Session()
        
        try:
            # Get recent deployments
            since = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT t.*, COUNT(d.id) as recent_downloads
                FROM scraper_templates t
                LEFT JOIN template_deployments d ON t.id = d.template_id
                WHERE d.deployed_at > :since
                GROUP BY t.id
                ORDER BY recent_downloads DESC
                LIMIT :limit
            """)
            
            results = session.execute(query, {'since': since, 'limit': limit})
            
            templates = []
            for row in results:
                template = ScraperTemplate(
                    id=row.id,
                    name=row.name,
                    description=row.description,
                    category=TemplateCategory(row.category),
                    platform=row.platform,
                    author=row.author,
                    version=row.version,
                    status=TemplateStatus(row.status),
                    config_schema=row.config_schema,
                    code_template='',
                    requirements=row.requirements,
                    tags=row.tags,
                    rating=row.rating,
                    downloads=row.downloads,
                    created_at=row.created_at,
                    updated_at=row.updated_at
                )
                templates.append(template)
            
            return templates
            
        finally:
            session.close()
    
    def _validate_template(self, template: ScraperTemplate) -> bool:
        """Validate template before publishing"""
        # Check required fields
        if not all([template.name, template.code_template, template.config_schema]):
            return False
        
        # Validate schema
        if not isinstance(template.config_schema, dict):
            return False
        
        # Check code template has placeholders
        if '{{' not in template.code_template:
            logger.warning("Template has no placeholders")
        
        return True
    
    def _validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        # Check required fields
        for field in schema.get('required', []):
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate types (simplified)
        for field, value in config.items():
            if field in schema.get('properties', {}):
                expected_type = schema['properties'][field].get('type')
                if expected_type == 'string' and not isinstance(value, str):
                    logger.error(f"Invalid type for {field}: expected string")
                    return False
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    logger.error(f"Invalid type for {field}: expected number")
                    return False
        
        return True
    
    def _generate_scraper_code(self, template: ScraperTemplate, config: Dict[str, Any]) -> str:
        """Generate scraper code from template and config"""
        # Use Jinja2 to render template
        jinja_template = jinja2.Template(template.code_template)
        
        # Add config and helper functions
        context = {
            'config': config,
            'datetime': datetime,
            'uuid': uuid
        }
        
        return jinja_template.render(**context)
    
    async def _deploy_with_docker(self, deployment_path: Path, deployment_id: str) -> bool:
        """Deploy scraper using Docker"""
        try:
            client = docker.from_env()
            
            # Build image
            image, logs = client.images.build(
                path=str(deployment_path),
                tag=f"scraper:{deployment_id}",
                rm=True
            )
            
            # Run container
            container = client.containers.run(
                image=image.id,
                name=f"scraper_{deployment_id}",
                environment={
                    'SCRAPER_ID': deployment_id,
                    'CONFIG_FILE': '/app/config.yaml'
                },
                detach=True,
                restart_policy={'Name': 'unless-stopped'}
            )
            
            logger.info(f"Deployed scraper container: {container.id}")
            return True
            
        except Exception as e:
            logger.error(f"Docker deployment error: {e}")
            return False
    
    async def _clear_cache_for_template(self, template_id: str):
        """Clear cache for a specific template"""
        if self.redis_client:
            pattern = f"templates:*{template_id}*"
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)
    
    async def _clear_cache_for_category(self, category: TemplateCategory):
        """Clear cache for a category"""
        if self.redis_client:
            pattern = f"templates:search:*{category.value}*"
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)


# Template library with pre-built templates
class TemplateLibrary:
    """Pre-built template library"""
    
    @staticmethod
    def get_municipal_meeting_template() -> ScraperTemplate:
        """Template for municipal meeting scrapers"""
        return ScraperTemplate(
            id="tpl_municipal_meetings_v1",
            name="Municipal Meeting Scraper",
            description="Scrapes meeting agendas, minutes, and videos from municipal websites",
            category=TemplateCategory.MEETINGS,
            platform="legistar",
            author="OpenPolicy",
            version="1.0.0",
            status=TemplateStatus.APPROVED,
            config_schema={
                "type": "object",
                "properties": {
                    "base_url": {
                        "type": "string",
                        "description": "Base URL of the municipal website"
                    },
                    "city_name": {
                        "type": "string",
                        "description": "Name of the city"
                    },
                    "meeting_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["City Council", "Committee"]
                    }
                },
                "required": ["base_url", "city_name"]
            },
            code_template="""
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json

class MunicipalMeetingScraper:
    def __init__(self, config):
        self.base_url = config['base_url']
        self.city_name = config['city_name']
        self.meeting_types = config.get('meeting_types', ['City Council'])
    
    def scrape_meetings(self):
        meetings = []
        
        for meeting_type in self.meeting_types:
            url = f"{self.base_url}/meetings/{meeting_type.replace(' ', '_')}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for meeting in soup.select('.meeting-row'):
                meeting_data = {
                    'city': self.city_name,
                    'type': meeting_type,
                    'date': meeting.select_one('.meeting-date').text,
                    'time': meeting.select_one('.meeting-time').text,
                    'location': meeting.select_one('.meeting-location').text,
                    'agenda_url': meeting.select_one('.agenda-link')['href'],
                    'scraped_at': datetime.utcnow().isoformat()
                }
                meetings.append(meeting_data)
        
        return meetings

# Usage
scraper = MunicipalMeetingScraper(config)
results = scraper.scrape_meetings()
print(json.dumps(results, indent=2))
""",
            requirements=[
                "requests>=2.25.0",
                "beautifulsoup4>=4.9.0",
                "python-dateutil>=2.8.0"
            ],
            tags=["municipal", "meetings", "agendas", "minutes"]
        )
    
    @staticmethod
    def get_legislative_bill_template() -> ScraperTemplate:
        """Template for legislative bill scrapers"""
        return ScraperTemplate(
            id="tpl_legislative_bills_v1",
            name="Legislative Bill Tracker",
            description="Tracks bills, sponsors, and status through legislative systems",
            category=TemplateCategory.LEGISLATIVE,
            platform="custom",
            author="OpenPolicy",
            version="1.0.0",
            status=TemplateStatus.APPROVED,
            config_schema={
                "type": "object",
                "properties": {
                    "legislature_url": {
                        "type": "string",
                        "description": "URL of the legislature website"
                    },
                    "session": {
                        "type": "string",
                        "description": "Legislative session (e.g., '2024')"
                    },
                    "bill_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["HB", "SB", "HR", "SR"]
                    }
                },
                "required": ["legislature_url", "session"]
            },
            code_template="""
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from datetime import datetime
import re

class LegislativeBillScraper:
    def __init__(self, config):
        self.base_url = config['legislature_url']
        self.session = config['session']
        self.bill_types = config.get('bill_types', ['HB', 'SB'])
    
    async def scrape_bills(self):
        bills = []
        
        async with aiohttp.ClientSession() as session:
            for bill_type in self.bill_types:
                url = f"{self.base_url}/bills/{self.session}/{bill_type}"
                
                async with session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    for bill_row in soup.select('.bill-item'):
                        bill_data = await self._extract_bill_data(bill_row, session)
                        bills.append(bill_data)
        
        return bills
    
    async def _extract_bill_data(self, bill_element, session):
        bill_number = bill_element.select_one('.bill-number').text
        bill_url = bill_element.select_one('.bill-link')['href']
        
        # Get detailed bill info
        async with session.get(f"{self.base_url}{bill_url}") as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            return {
                'number': bill_number,
                'title': soup.select_one('.bill-title').text,
                'sponsor': soup.select_one('.primary-sponsor').text,
                'status': soup.select_one('.bill-status').text,
                'introduced_date': soup.select_one('.intro-date').text,
                'summary': soup.select_one('.bill-summary').text,
                'url': f"{self.base_url}{bill_url}",
                'session': self.session,
                'scraped_at': datetime.utcnow().isoformat()
            }

# Usage
scraper = LegislativeBillScraper(config)
bills = asyncio.run(scraper.scrape_bills())
""",
            requirements=[
                "aiohttp>=3.8.0",
                "beautifulsoup4>=4.9.0",
                "python-dateutil>=2.8.0"
            ],
            tags=["legislative", "bills", "legislation", "government"]
        )


# Example usage
async def marketplace_demo():
    """Demo the marketplace functionality"""
    
    # Initialize marketplace
    storage_config = {
        'endpoint': 'localhost:9000',
        'access_key': 'minioadmin',
        'secret_key': 'minioadmin',
        'bucket': 'scraper-templates'
    }
    
    marketplace = ScraperMarketplace(
        'postgresql://user:pass@localhost/marketplace',
        storage_config
    )
    
    await marketplace.initialize()
    
    # Get pre-built templates
    municipal_template = TemplateLibrary.get_municipal_meeting_template()
    legislative_template = TemplateLibrary.get_legislative_bill_template()
    
    # Publish templates
    await marketplace.publish_template(municipal_template)
    await marketplace.publish_template(legislative_template)
    
    # Search templates
    templates = await marketplace.search_templates(
        category=TemplateCategory.MEETINGS,
        min_rating=4.0
    )
    
    print(f"Found {len(templates)} meeting templates")
    
    # Deploy a template
    if templates:
        template = templates[0]
        config = {
            'base_url': 'https://toronto.ca',
            'city_name': 'Toronto'
        }
        
        deployment_id = await marketplace.deploy_template(
            template.id,
            config,
            'toronto_meetings',
            'admin'
        )
        
        print(f"Deployed template as: {deployment_id}")
    
    # Get trending templates
    trending = await marketplace.get_trending_templates()
    print(f"Trending templates: {[t.name for t in trending]}")


if __name__ == "__main__":
    asyncio.run(marketplace_demo())