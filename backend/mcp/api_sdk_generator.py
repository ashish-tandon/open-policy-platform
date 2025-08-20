"""
API SDK Generator - 40by6
Generate client SDKs for multiple programming languages from OpenAPI spec
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
import re
import shutil
import subprocess
import tempfile
import zipfile
import tarfile
from jinja2 import Template, Environment, FileSystemLoader, select_autoescape
import black
import autopep8
import isort
from openapi_spec_validator import validate_spec
from openapi_schema_to_json_schema import to_json_schema
import requests
import aiohttp
import git
from semantic_version import Version
import toml
from typing_extensions import Literal

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    DART = "dart"
    CPP = "cpp"
    ELIXIR = "elixir"
    SCALA = "scala"


class SDKFeature(Enum):
    """SDK features to generate"""
    CLIENT = "client"
    MODELS = "models"
    AUTHENTICATION = "authentication"
    PAGINATION = "pagination"
    RETRY = "retry"
    RATE_LIMITING = "rate_limiting"
    CACHING = "caching"
    WEBSOCKET = "websocket"
    ASYNC = "async"
    VALIDATION = "validation"
    SERIALIZATION = "serialization"
    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


@dataclass
class SDKConfig:
    """SDK generation configuration"""
    language: Language
    package_name: str
    version: str
    features: Set[SDKFeature] = field(default_factory=lambda: {SDKFeature.CLIENT, SDKFeature.MODELS})
    author: Optional[str] = None
    license: str = "MIT"
    repository_url: Optional[str] = None
    description: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    custom_templates: Optional[Path] = None
    output_dir: Optional[Path] = None
    include_examples: bool = True
    include_tests: bool = True
    style_guide: Optional[str] = None  # e.g., "pep8", "google", "airbnb"


@dataclass
class APIEndpoint:
    """Represents an API endpoint"""
    path: str
    method: str
    operation_id: str
    summary: Optional[str] = None
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class APIModel:
    """Represents an API model/schema"""
    name: str
    properties: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    description: Optional[str] = None
    example: Optional[Dict[str, Any]] = None
    discriminator: Optional[str] = None
    inheritance: Optional[str] = None


class OpenAPIParser:
    """Parse OpenAPI specification"""
    
    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        self.spec = None
        self.endpoints: List[APIEndpoint] = []
        self.models: Dict[str, APIModel] = {}
        self.security_schemes: Dict[str, Any] = {}
    
    def parse(self) -> Tuple[List[APIEndpoint], Dict[str, APIModel]]:
        """Parse OpenAPI spec"""
        
        # Load spec
        with open(self.spec_path, 'r') as f:
            if self.spec_path.endswith('.yaml') or self.spec_path.endswith('.yml'):
                self.spec = yaml.safe_load(f)
            else:
                self.spec = json.load(f)
        
        # Validate spec
        validate_spec(self.spec)
        
        # Parse components
        self._parse_security_schemes()
        self._parse_models()
        self._parse_endpoints()
        
        return self.endpoints, self.models
    
    def _parse_security_schemes(self):
        """Parse security schemes"""
        components = self.spec.get('components', {})
        self.security_schemes = components.get('securitySchemes', {})
    
    def _parse_models(self):
        """Parse models/schemas"""
        components = self.spec.get('components', {})
        schemas = components.get('schemas', {})
        
        for name, schema in schemas.items():
            model = APIModel(
                name=name,
                properties=schema.get('properties', {}),
                required=schema.get('required', []),
                description=schema.get('description'),
                example=schema.get('example')
            )
            
            # Check for inheritance
            if 'allOf' in schema:
                for item in schema['allOf']:
                    if '$ref' in item:
                        ref_name = item['$ref'].split('/')[-1]
                        model.inheritance = ref_name
            
            self.models[name] = model
    
    def _parse_endpoints(self):
        """Parse API endpoints"""
        paths = self.spec.get('paths', {})
        
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method in ['get', 'post', 'put', 'patch', 'delete', 'head', 'options']:
                    endpoint = APIEndpoint(
                        path=path,
                        method=method.upper(),
                        operation_id=operation.get('operationId', f"{method}_{path.replace('/', '_')}"),
                        summary=operation.get('summary'),
                        description=operation.get('description'),
                        parameters=operation.get('parameters', []),
                        request_body=operation.get('requestBody'),
                        responses=operation.get('responses', {}),
                        security=operation.get('security', []),
                        tags=operation.get('tags', [])
                    )
                    self.endpoints.append(endpoint)


class BaseSDKGenerator:
    """Base class for SDK generators"""
    
    def __init__(self, config: SDKConfig, spec: Dict[str, Any]):
        self.config = config
        self.spec = spec
        self.env = self._setup_jinja_env()
    
    def _setup_jinja_env(self) -> Environment:
        """Setup Jinja2 environment"""
        
        # Use custom templates if provided
        if self.config.custom_templates:
            loader = FileSystemLoader(self.config.custom_templates)
        else:
            # Use built-in templates
            template_dir = Path(__file__).parent / 'templates' / self.config.language.value
            loader = FileSystemLoader(str(template_dir))
        
        env = Environment(
            loader=loader,
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        env.filters['snake_case'] = self._to_snake_case
        env.filters['camel_case'] = self._to_camel_case
        env.filters['pascal_case'] = self._to_pascal_case
        env.filters['kebab_case'] = self._to_kebab_case
        env.filters['constant_case'] = self._to_constant_case
        
        return env
    
    def generate(
        self,
        endpoints: List[APIEndpoint],
        models: Dict[str, APIModel]
    ) -> Dict[str, str]:
        """Generate SDK files"""
        raise NotImplementedError("Subclasses must implement generate()")
    
    def _to_snake_case(self, s: str) -> str:
        """Convert to snake_case"""
        s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
        s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
        return s.lower()
    
    def _to_camel_case(self, s: str) -> str:
        """Convert to camelCase"""
        parts = s.split('_')
        return parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])
    
    def _to_pascal_case(self, s: str) -> str:
        """Convert to PascalCase"""
        return ''.join(p.capitalize() for p in s.split('_'))
    
    def _to_kebab_case(self, s: str) -> str:
        """Convert to kebab-case"""
        return s.replace('_', '-').lower()
    
    def _to_constant_case(self, s: str) -> str:
        """Convert to CONSTANT_CASE"""
        return s.upper()


class PythonSDKGenerator(BaseSDKGenerator):
    """Generate Python SDK"""
    
    def generate(
        self,
        endpoints: List[APIEndpoint],
        models: Dict[str, APIModel]
    ) -> Dict[str, str]:
        """Generate Python SDK files"""
        
        files = {}
        
        # Generate package structure
        package_dir = self.config.package_name.replace('-', '_')
        
        # __init__.py
        files[f"{package_dir}/__init__.py"] = self._generate_init()
        
        # Client
        if SDKFeature.CLIENT in self.config.features:
            files[f"{package_dir}/client.py"] = self._generate_client(endpoints)
        
        # Models
        if SDKFeature.MODELS in self.config.features:
            files[f"{package_dir}/models.py"] = self._generate_models(models)
        
        # Authentication
        if SDKFeature.AUTHENTICATION in self.config.features:
            files[f"{package_dir}/auth.py"] = self._generate_auth()
        
        # Utilities
        files[f"{package_dir}/utils.py"] = self._generate_utils()
        
        # Exceptions
        files[f"{package_dir}/exceptions.py"] = self._generate_exceptions()
        
        # Configuration
        files[f"{package_dir}/config.py"] = self._generate_config()
        
        # Examples
        if self.config.include_examples:
            files["examples/basic_usage.py"] = self._generate_example()
            files["examples/advanced_usage.py"] = self._generate_advanced_example()
        
        # Tests
        if self.config.include_tests:
            files["tests/test_client.py"] = self._generate_client_tests()
            files["tests/test_models.py"] = self._generate_model_tests()
            files["tests/conftest.py"] = self._generate_test_config()
        
        # Setup files
        files["setup.py"] = self._generate_setup()
        files["requirements.txt"] = self._generate_requirements()
        files["README.md"] = self._generate_readme()
        files["LICENSE"] = self._generate_license()
        files[".gitignore"] = self._generate_gitignore()
        
        # CI/CD
        files[".github/workflows/test.yml"] = self._generate_github_actions()
        
        # Format code
        for path, content in files.items():
            if path.endswith('.py'):
                files[path] = self._format_python_code(content)
        
        return files
    
    def _generate_init(self) -> str:
        """Generate __init__.py"""
        template = '''"""
{{ config.package_name }} - {{ config.description or "API Client Library" }}
Version: {{ config.version }}
"""

__version__ = "{{ config.version }}"
__author__ = "{{ config.author or 'Generated' }}"

from .client import Client
from .exceptions import (
    APIException,
    AuthenticationError,
    RateLimitError,
    ValidationError,
)

{% if 'models' in features %}
from .models import *
{% endif %}

__all__ = [
    "Client",
    "APIException",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
]
'''
        return self.env.from_string(template).render(
            config=self.config,
            features=[f.value for f in self.config.features]
        )
    
    def _generate_client(self, endpoints: List[APIEndpoint]) -> str:
        """Generate client.py"""
        template = '''"""
API Client for {{ spec.info.title }}
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urljoin, urlencode
import json

import httpx
from httpx import Response, HTTPError

from .exceptions import APIException, AuthenticationError, RateLimitError
from .config import Config
from .utils import retry_with_backoff, rate_limit
{% if 'authentication' in features %}
from .auth import AuthProvider
{% endif %}
{% if 'models' in features %}
from . import models
{% endif %}

logger = logging.getLogger(__name__)


class Client:
    """Main API client"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        {% if 'authentication' in features %}
        auth_provider: Optional[AuthProvider] = None,
        {% endif %}
        **kwargs
    ):
        self.base_url = base_url or Config.BASE_URL
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        {% if 'authentication' in features %}
        self.auth_provider = auth_provider
        {% endif %}
        
        # HTTP client
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._default_headers(),
            **kwargs
        )
        
        # Async HTTP client
        self.async_client = None
        
        # Initialize API groups
        {% for tag in tags %}
        self.{{ tag|snake_case }} = {{ tag|pascal_case }}API(self)
        {% endfor %}
    
    def _default_headers(self) -> Dict[str, str]:
        """Get default headers"""
        headers = {
            "User-Agent": f"{{ config.package_name }}/{{ config.version }}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    {% if 'async' in features %}
    async def __aenter__(self):
        """Async context manager entry"""
        self.async_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._default_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.async_client:
            await self.async_client.aclose()
    {% endif %}
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def close(self):
        """Close the client"""
        self.client.close()
        if self.async_client:
            asyncio.create_task(self.async_client.aclose())
    
    @retry_with_backoff(max_retries=3)
    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Response:
        """Make HTTP request"""
        
        # Merge headers
        request_headers = self._default_headers()
        if headers:
            request_headers.update(headers)
        
        {% if 'authentication' in features %}
        # Apply authentication
        if self.auth_provider:
            request_headers.update(self.auth_provider.get_headers())
        {% endif %}
        
        # Make request
        try:
            response = self.client.request(
                method=method,
                url=path,
                params=params,
                json=json_data,
                headers=request_headers,
                **kwargs
            )
            response.raise_for_status()
            return response
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Authentication failed")
            elif e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            else:
                raise APIException(f"API error: {e.response.text}")
        except HTTPError as e:
            raise APIException(f"HTTP error: {str(e)}")
    
    {% if 'async' in features %}
    async def arequest(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Response:
        """Make async HTTP request"""
        
        if not self.async_client:
            raise RuntimeError("Async client not initialized. Use 'async with' context manager.")
        
        # Similar to sync request but async
        request_headers = self._default_headers()
        if headers:
            request_headers.update(headers)
        
        try:
            response = await self.async_client.request(
                method=method,
                url=path,
                params=params,
                json=json_data,
                headers=request_headers,
                **kwargs
            )
            response.raise_for_status()
            return response
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Authentication failed")
            elif e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            else:
                raise APIException(f"API error: {e.response.text}")
    {% endif %}


{% for tag in tags %}
class {{ tag|pascal_case }}API:
    """{{ tag }} API operations"""
    
    def __init__(self, client: Client):
        self.client = client
    
    {% for endpoint in endpoints if tag in endpoint.tags %}
    {% if 'rate_limiting' in features %}
    @rate_limit(calls=100, period=60)
    {% endif %}
    def {{ endpoint.operation_id|snake_case }}(
        self,
        {% for param in endpoint.parameters %}
        {{ param.name }}: {{ python_type(param.schema) }}{% if not param.required %} = None{% endif %},
        {% endfor %}
        {% if endpoint.request_body %}
        body: {% if 'models' in features %}models.{{ get_model_name(endpoint.request_body) }}{% else %}Dict[str, Any]{% endif %},
        {% endif %}
    ) -> {% if 'models' in features and endpoint.responses.get('200') %}models.{{ get_response_model(endpoint.responses['200']) }}{% else %}Dict[str, Any]{% endif %}:
        """
        {{ endpoint.summary or endpoint.description or endpoint.operation_id }}
        
        {% if endpoint.description %}
        {{ endpoint.description }}
        {% endif %}
        
        {% for param in endpoint.parameters %}
        :param {{ param.name }}: {{ param.description or 'No description' }}
        {% endfor %}
        {% if endpoint.request_body %}
        :param body: Request body
        {% endif %}
        :return: Response data
        """
        
        # Build path
        path = "{{ endpoint.path }}"
        {% for param in endpoint.parameters if param.in == 'path' %}
        path = path.replace("{" + "{{ param.name }}" + "}", str({{ param.name }}))
        {% endfor %}
        
        # Build query parameters
        params = {}
        {% for param in endpoint.parameters if param.in == 'query' %}
        if {{ param.name }} is not None:
            params["{{ param.name }}"] = {{ param.name }}
        {% endfor %}
        
        # Make request
        response = self.client.request(
            method="{{ endpoint.method }}",
            path=path,
            {% if params %}
            params=params,
            {% endif %}
            {% if endpoint.request_body %}
            json_data=body{% if 'models' in features %}.dict(){% endif %},
            {% endif %}
        )
        
        # Parse response
        data = response.json()
        {% if 'models' in features and endpoint.responses.get('200') %}
        return models.{{ get_response_model(endpoint.responses['200']) }}(**data)
        {% else %}
        return data
        {% endif %}
    
    {% if 'async' in features %}
    async def {{ endpoint.operation_id|snake_case }}_async(
        self,
        {% for param in endpoint.parameters %}
        {{ param.name }}: {{ python_type(param.schema) }}{% if not param.required %} = None{% endif %},
        {% endfor %}
        {% if endpoint.request_body %}
        body: {% if 'models' in features %}models.{{ get_model_name(endpoint.request_body) }}{% else %}Dict[str, Any]{% endif %},
        {% endif %}
    ) -> {% if 'models' in features and endpoint.responses.get('200') %}models.{{ get_response_model(endpoint.responses['200']) }}{% else %}Dict[str, Any]{% endif %}:
        """Async version of {{ endpoint.operation_id|snake_case }}"""
        
        # Similar implementation but using arequest
        path = "{{ endpoint.path }}"
        {% for param in endpoint.parameters if param.in == 'path' %}
        path = path.replace("{" + "{{ param.name }}" + "}", str({{ param.name }}))
        {% endfor %}
        
        params = {}
        {% for param in endpoint.parameters if param.in == 'query' %}
        if {{ param.name }} is not None:
            params["{{ param.name }}"] = {{ param.name }}
        {% endfor %}
        
        response = await self.client.arequest(
            method="{{ endpoint.method }}",
            path=path,
            {% if params %}
            params=params,
            {% endif %}
            {% if endpoint.request_body %}
            json_data=body{% if 'models' in features %}.dict(){% endif %},
            {% endif %}
        )
        
        data = response.json()
        {% if 'models' in features and endpoint.responses.get('200') %}
        return models.{{ get_response_model(endpoint.responses['200']) }}(**data)
        {% else %}
        return data
        {% endif %}
    {% endif %}
    
    {% endfor %}
{% endfor %}
'''
        # Extract unique tags
        tags = set()
        for endpoint in endpoints:
            tags.update(endpoint.tags)
        
        return self.env.from_string(template).render(
            spec=self.spec,
            config=self.config,
            endpoints=endpoints,
            tags=sorted(tags),
            features=[f.value for f in self.config.features],
            python_type=self._python_type,
            get_model_name=self._get_model_name,
            get_response_model=self._get_response_model
        )
    
    def _generate_models(self, models: Dict[str, APIModel]) -> str:
        """Generate models.py"""
        template = '''"""
Data models for {{ spec.info.title }}
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

{% if 'pydantic' in dependencies %}
from pydantic import BaseModel, Field, validator
{% else %}
from dataclasses import dataclass, field
from typing import ClassVar
{% endif %}


{% for name, model in models.items() %}
{% if 'pydantic' in dependencies %}
class {{ name }}(BaseModel):
    """{{ model.description or name }}"""
    
    {% for prop_name, prop_schema in model.properties.items() %}
    {{ prop_name|snake_case }}: {{ python_type(prop_schema) }}{% if prop_name not in model.required %} = None{% endif %}{% if prop_schema.description %} # {{ prop_schema.description }}{% endif %}
    {% endfor %}
    
    {% if model.example %}
    class Config:
        schema_extra = {
            "example": {{ model.example }}
        }
    {% endif %}
    
    {% if 'validation' in features %}
    {% for prop_name, prop_schema in model.properties.items() if 'pattern' in prop_schema or 'minimum' in prop_schema or 'maximum' in prop_schema %}
    @validator('{{ prop_name|snake_case }}')
    def validate_{{ prop_name|snake_case }}(cls, v):
        {% if 'pattern' in prop_schema %}
        import re
        if v and not re.match(r"{{ prop_schema.pattern }}", v):
            raise ValueError(f"Invalid {{ prop_name }}")
        {% endif %}
        {% if 'minimum' in prop_schema %}
        if v < {{ prop_schema.minimum }}:
            raise ValueError(f"{{ prop_name }} must be >= {{ prop_schema.minimum }}")
        {% endif %}
        {% if 'maximum' in prop_schema %}
        if v > {{ prop_schema.maximum }}:
            raise ValueError(f"{{ prop_name }} must be <= {{ prop_schema.maximum }}")
        {% endif %}
        return v
    {% endfor %}
    {% endif %}

{% else %}
@dataclass
class {{ name }}:
    """{{ model.description or name }}"""
    
    {% for prop_name, prop_schema in model.properties.items() %}
    {{ prop_name|snake_case }}: {{ python_type(prop_schema) }}{% if prop_name not in model.required %} = None{% endif %}
    {% endfor %}
    
    {% if 'serialization' in features %}
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            {% for prop_name in model.properties %}
            "{{ prop_name }}": self.{{ prop_name|snake_case }},
            {% endfor %}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "{{ name }}":
        """Create from dictionary"""
        return cls(
            {% for prop_name in model.properties %}
            {{ prop_name|snake_case }}=data.get("{{ prop_name }}"),
            {% endfor %}
        )
    {% endif %}
{% endif %}


{% endfor %}

# Enums
{% for name, schema in schemas.items() if schema.get('enum') %}
class {{ name }}(str, Enum):
    """{{ schema.description or name }}"""
    {% for value in schema.enum %}
    {{ value|constant_case }} = "{{ value }}"
    {% endfor %}
{% endfor %}
'''
        return self.env.from_string(template).render(
            spec=self.spec,
            models=models,
            schemas=self.spec.get('components', {}).get('schemas', {}),
            features=[f.value for f in self.config.features],
            dependencies=self.config.dependencies,
            python_type=self._python_type
        )
    
    def _generate_auth(self) -> str:
        """Generate auth.py"""
        template = '''"""
Authentication providers for {{ config.package_name }}
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import base64
import hashlib
import hmac
import time
import jwt
import httpx


class AuthProvider(ABC):
    """Base authentication provider"""
    
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        pass
    
    @abstractmethod
    def refresh(self) -> None:
        """Refresh authentication if needed"""
        pass


class APIKeyAuth(AuthProvider):
    """API Key authentication"""
    
    def __init__(self, api_key: str, header_name: str = "X-API-Key"):
        self.api_key = api_key
        self.header_name = header_name
    
    def get_headers(self) -> Dict[str, str]:
        return {self.header_name: self.api_key}
    
    def refresh(self) -> None:
        # API keys don't need refresh
        pass


class BearerTokenAuth(AuthProvider):
    """Bearer token authentication"""
    
    def __init__(self, token: str):
        self.token = token
    
    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}
    
    def refresh(self) -> None:
        # Static tokens don't need refresh
        pass


class OAuth2Auth(AuthProvider):
    """OAuth2 authentication with auto-refresh"""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: Optional[str] = None,
        initial_token: Optional[str] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self.access_token = initial_token
        self.refresh_token = None
        self.expires_at = None
        
        if not self.access_token:
            self.refresh()
    
    def get_headers(self) -> Dict[str, str]:
        # Check if token needs refresh
        if self.expires_at and datetime.utcnow() >= self.expires_at:
            self.refresh()
        
        return {"Authorization": f"Bearer {self.access_token}"}
    
    def refresh(self) -> None:
        """Get new access token"""
        
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        
        if self.scope:
            data["scope"] = self.scope
        
        if self.refresh_token:
            data["grant_type"] = "refresh_token"
            data["refresh_token"] = self.refresh_token
        
        # Make token request
        response = httpx.post(self.token_url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token")
        
        # Calculate expiration
        expires_in = token_data.get("expires_in", 3600)
        self.expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)


class JWTAuth(AuthProvider):
    """JWT authentication"""
    
    def __init__(
        self,
        private_key: str,
        key_id: str,
        issuer: str,
        audience: str,
        algorithm: str = "RS256",
        ttl: int = 3600
    ):
        self.private_key = private_key
        self.key_id = key_id
        self.issuer = issuer
        self.audience = audience
        self.algorithm = algorithm
        self.ttl = ttl
        self._token = None
        self._expires_at = None
    
    def get_headers(self) -> Dict[str, str]:
        # Generate new token if needed
        if not self._token or datetime.utcnow() >= self._expires_at:
            self._generate_token()
        
        return {"Authorization": f"Bearer {self._token}"}
    
    def refresh(self) -> None:
        self._generate_token()
    
    def _generate_token(self) -> None:
        """Generate new JWT token"""
        
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self.ttl)
        
        payload = {
            "iss": self.issuer,
            "aud": self.audience,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": hashlib.sha256(f"{now.timestamp()}".encode()).hexdigest()
        }
        
        headers = {"kid": self.key_id}
        
        self._token = jwt.encode(
            payload,
            self.private_key,
            algorithm=self.algorithm,
            headers=headers
        )
        
        self._expires_at = expires_at - timedelta(seconds=60)  # Refresh 1 min early


class HMACAuth(AuthProvider):
    """HMAC signature authentication"""
    
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key
    
    def get_headers(self) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{self.access_key}:{timestamp}"
        
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-Access-Key": self.access_key,
            "X-Timestamp": timestamp,
            "X-Signature": signature
        }
    
    def refresh(self) -> None:
        # HMAC doesn't need refresh
        pass
'''
        return self.env.from_string(template).render(config=self.config)
    
    def _generate_utils(self) -> str:
        """Generate utils.py"""
        template = '''"""
Utility functions for {{ config.package_name }}
"""

import time
import functools
import logging
from typing import Any, Callable, TypeVar, Optional
import random
import threading
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """Decorator for retrying functions with exponential backoff"""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        break
                    
                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    
    return decorator


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.clock = time.monotonic
        self.last_reset = self.clock()
        self.num_calls = 0
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with self.lock:
                now = self.clock()
                elapsed = now - self.last_reset
                
                if elapsed > self.period:
                    self.num_calls = 0
                    self.last_reset = now
                
                if self.num_calls >= self.calls:
                    sleep_time = self.period - elapsed
                    logger.debug(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    self.num_calls = 0
                    self.last_reset = self.clock()
                
                self.num_calls += 1
            
            return func(*args, **kwargs)
        
        return wrapper


def rate_limit(calls: int = 15, period: float = 900):
    """Rate limiting decorator"""
    return RateLimiter(calls, period)


class Cache:
    """Simple TTL cache"""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


def cached(ttl: int = 300):
    """Caching decorator"""
    cache = Cache(ttl)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit for {key}")
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        return wrapper
    
    return decorator


def validate_params(**validators):
    """Parameter validation decorator"""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid {param_name}: {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def chunked(iterable, size):
    """Split iterable into chunks"""
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            break
        yield chunk


def flatten(nested_list):
    """Flatten nested list"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
'''
        return self.env.from_string(template).render(config=self.config)
    
    def _generate_exceptions(self) -> str:
        """Generate exceptions.py"""
        return '''"""
Custom exceptions for API client
"""


class APIException(Exception):
    """Base API exception"""
    pass


class AuthenticationError(APIException):
    """Authentication failed"""
    pass


class AuthorizationError(APIException):
    """Authorization failed"""
    pass


class RateLimitError(APIException):
    """Rate limit exceeded"""
    pass


class ValidationError(APIException):
    """Validation error"""
    pass


class NotFoundError(APIException):
    """Resource not found"""
    pass


class ServerError(APIException):
    """Server error"""
    pass


class NetworkError(APIException):
    """Network error"""
    pass


class TimeoutError(APIException):
    """Request timeout"""
    pass
'''
    
    def _generate_config(self) -> str:
        """Generate config.py"""
        template = '''"""
Configuration for {{ config.package_name }}
"""

import os
from typing import Optional


class Config:
    """Configuration settings"""
    
    # API settings
    BASE_URL: str = os.getenv("{{ config.package_name|upper }}_BASE_URL", "{{ spec.servers[0].url if spec.servers else 'http://localhost:8000' }}")
    API_KEY: Optional[str] = os.getenv("{{ config.package_name|upper }}_API_KEY")
    
    # Client settings
    TIMEOUT: float = float(os.getenv("{{ config.package_name|upper }}_TIMEOUT", "30.0"))
    MAX_RETRIES: int = int(os.getenv("{{ config.package_name|upper }}_MAX_RETRIES", "3"))
    
    # Rate limiting
    RATE_LIMIT_CALLS: int = int(os.getenv("{{ config.package_name|upper }}_RATE_LIMIT_CALLS", "100"))
    RATE_LIMIT_PERIOD: float = float(os.getenv("{{ config.package_name|upper }}_RATE_LIMIT_PERIOD", "60.0"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("{{ config.package_name|upper }}_LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cache settings
    CACHE_ENABLED: bool = os.getenv("{{ config.package_name|upper }}_CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("{{ config.package_name|upper }}_CACHE_TTL", "300"))
    
    # SSL/TLS
    VERIFY_SSL: bool = os.getenv("{{ config.package_name|upper }}_VERIFY_SSL", "true").lower() == "true"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls()
'''
        return self.env.from_string(template).render(
            config=self.config,
            spec=self.spec
        )
    
    def _generate_example(self) -> str:
        """Generate basic example"""
        template = '''"""
Basic usage example for {{ config.package_name }}
"""

from {{ config.package_name }} import Client, APIException


def main():
    # Initialize client
    client = Client(
        api_key="your-api-key-here"
    )
    
    try:
        # Example API calls based on available endpoints
        {% for endpoint in endpoints[:3] %}
        {% if endpoint.method == 'GET' and not endpoint.parameters %}
        # {{ endpoint.summary or endpoint.operation_id }}
        result = client.{{ (endpoint.tags[0] if endpoint.tags else 'api')|snake_case }}.{{ endpoint.operation_id|snake_case }}()
        print(f"{{ endpoint.operation_id }}: {result}")
        {% endif %}
        {% endfor %}
        
    except APIException as e:
        print(f"API error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
'''
        return self.env.from_string(template).render(
            config=self.config,
            endpoints=self.endpoints[:10]  # Limit to first 10 endpoints
        )
    
    def _generate_advanced_example(self) -> str:
        """Generate advanced example"""
        template = '''"""
Advanced usage example for {{ config.package_name }}
"""

import asyncio
from {{ config.package_name }} import Client, OAuth2Auth, APIException
{% if 'models' in features %}
from {{ config.package_name }} import models
{% endif %}


async def async_example():
    """Async example with OAuth2 authentication"""
    
    # Setup OAuth2 authentication
    auth = OAuth2Auth(
        client_id="your-client-id",
        client_secret="your-client-secret",
        token_url="https://auth.example.com/token"
    )
    
    # Initialize async client
    async with Client(auth_provider=auth) as client:
        try:
            # Parallel requests
            tasks = []
            {% for endpoint in endpoints[:3] %}
            {% if endpoint.method == 'GET' and not endpoint.parameters and 'async' in features %}
            tasks.append(
                client.{{ (endpoint.tags[0] if endpoint.tags else 'api')|snake_case }}.{{ endpoint.operation_id|snake_case }}_async()
            )
            {% endif %}
            {% endfor %}
            
            if tasks:
                results = await asyncio.gather(*tasks)
                for i, result in enumerate(results):
                    print(f"Result {i + 1}: {result}")
            
        except APIException as e:
            print(f"API error: {e}")


def pagination_example():
    """Example with pagination"""
    
    client = Client()
    
    # Paginate through results
    page = 1
    while True:
        try:
            result = client.api.list_items(page=page, per_page=50)
            
            if not result.items:
                break
            
            for item in result.items:
                print(f"Processing: {item.id}")
            
            page += 1
            
        except APIException as e:
            print(f"Error: {e}")
            break


def error_handling_example():
    """Comprehensive error handling"""
    
    from {{ config.package_name }} import (
        Client,
        AuthenticationError,
        RateLimitError,
        ValidationError,
        NotFoundError
    )
    
    client = Client()
    
    try:
        result = client.api.get_resource("123")
    except AuthenticationError:
        print("Please check your API credentials")
    except RateLimitError:
        print("Rate limit exceeded. Please wait before retrying.")
    except ValidationError as e:
        print(f"Invalid request: {e}")
    except NotFoundError:
        print("Resource not found")
    except APIException as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Run async example
    asyncio.run(async_example())
    
    # Run other examples
    pagination_example()
    error_handling_example()
'''
        return self.env.from_string(template).render(
            config=self.config,
            endpoints=self.endpoints,
            features=[f.value for f in self.config.features]
        )
    
    def _generate_setup(self) -> str:
        """Generate setup.py"""
        template = '''"""
Setup configuration for {{ config.package_name }}
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="{{ config.package_name }}",
    version="{{ config.version }}",
    author="{{ config.author or 'Generated' }}",
    author_email="{{ config.author_email or '' }}",
    description="{{ config.description or 'API Client Library' }}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="{{ config.repository_url or '' }}",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: {{ config.license }} License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "{{ config.package_name }}-cli={{ config.package_name }}.cli:main",
        ],
    },
)
'''
        return self.env.from_string(template).render(config=self.config)
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt"""
        
        deps = [
            "httpx>=0.23.0",
            "python-dateutil>=2.8.0",
        ]
        
        if 'pydantic' in self.config.dependencies:
            deps.append("pydantic>=1.10.0")
        
        if SDKFeature.AUTHENTICATION in self.config.features:
            deps.extend([
                "PyJWT>=2.4.0",
                "cryptography>=37.0.0",
            ])
        
        if SDKFeature.ASYNC in self.config.features:
            deps.append("asyncio>=3.4.3")
        
        # Add custom dependencies
        for dep, version in self.config.dependencies.items():
            deps.append(f"{dep}>={version}")
        
        return '\n'.join(sorted(set(deps)))
    
    def _generate_readme(self) -> str:
        """Generate README.md"""
        template = '''# {{ config.package_name }}

{{ config.description or 'API Client Library' }}

## Installation

```bash
pip install {{ config.package_name }}
```

## Quick Start

```python
from {{ config.package_name }} import Client

# Initialize the client
client = Client(api_key="your-api-key")

# Make API calls
response = client.api.get_data()
print(response)
```

## Features

{% for feature in features %}
- {{ feature|title|replace('_', ' ') }}
{% endfor %}

## Authentication

The client supports multiple authentication methods:

```python
# API Key
client = Client(api_key="your-api-key")

# Bearer Token
from {{ config.package_name }}.auth import BearerTokenAuth
auth = BearerTokenAuth("your-token")
client = Client(auth_provider=auth)

# OAuth2
from {{ config.package_name }}.auth import OAuth2Auth
auth = OAuth2Auth(
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_url="https://auth.example.com/token"
)
client = Client(auth_provider=auth)
```

{% if 'async' in features %}
## Async Support

The client supports async operations:

```python
import asyncio
from {{ config.package_name }} import Client

async def main():
    async with Client(api_key="your-api-key") as client:
        response = await client.api.get_data_async()
        print(response)

asyncio.run(main())
```
{% endif %}

## Error Handling

```python
from {{ config.package_name }} import Client, APIException, RateLimitError

client = Client(api_key="your-api-key")

try:
    response = client.api.get_data()
except RateLimitError:
    print("Rate limit exceeded. Please try again later.")
except APIException as e:
    print(f"API error: {e}")
```

## Configuration

Configure the client using environment variables:

- `{{ config.package_name|upper }}_BASE_URL`: API base URL
- `{{ config.package_name|upper }}_API_KEY`: API key
- `{{ config.package_name|upper }}_TIMEOUT`: Request timeout (default: 30)
- `{{ config.package_name|upper }}_MAX_RETRIES`: Max retry attempts (default: 3)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the {{ config.license }} License.
'''
        return self.env.from_string(template).render(
            config=self.config,
            features=[f.value for f in self.config.features]
        )
    
    def _generate_license(self) -> str:
        """Generate LICENSE file"""
        
        if self.config.license.upper() == "MIT":
            return f'''MIT License

Copyright (c) {datetime.now().year} {self.config.author or 'The Authors'}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
        else:
            return f"Copyright {datetime.now().year} {self.config.author or 'The Authors'}"
    
    def _generate_gitignore(self) -> str:
        """Generate .gitignore"""
        return '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
'''
    
    def _generate_github_actions(self) -> str:
        """Generate GitHub Actions workflow"""
        return '''name: Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.7', '3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type check with mypy
      run: |
        mypy --ignore-missing-imports .
    
    - name: Test with pytest
      run: |
        pytest --cov=./ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
'''
    
    def _generate_client_tests(self) -> str:
        """Generate test_client.py"""
        template = '''"""
Tests for API client
"""

import pytest
from unittest.mock import Mock, patch
import httpx

from {{ config.package_name }} import Client, APIException, RateLimitError


@pytest.fixture
def client():
    """Create test client"""
    return Client(api_key="test-key")


@pytest.fixture
def mock_response():
    """Create mock response"""
    response = Mock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {"result": "success"}
    response.headers = {}
    return response


class TestClient:
    """Test client functionality"""
    
    def test_client_initialization(self):
        """Test client initialization"""
        client = Client(api_key="test-key", base_url="https://api.example.com")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.example.com"
    
    def test_default_headers(self, client):
        """Test default headers"""
        headers = client._default_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert "User-Agent" in headers
    
    @patch('httpx.Client.request')
    def test_successful_request(self, mock_request, client, mock_response):
        """Test successful API request"""
        mock_request.return_value = mock_response
        
        response = client.request("GET", "/test")
        
        assert response.json() == {"result": "success"}
        mock_request.assert_called_once()
    
    @patch('httpx.Client.request')
    def test_rate_limit_error(self, mock_request, client):
        """Test rate limit error handling"""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=Mock(),
            response=mock_response
        )
        mock_request.return_value = mock_response
        
        with pytest.raises(RateLimitError):
            client.request("GET", "/test")
    
    @patch('httpx.Client.request')
    def test_retry_on_failure(self, mock_request, client, mock_response):
        """Test retry mechanism"""
        # First two calls fail, third succeeds
        mock_request.side_effect = [
            httpx.NetworkError("Connection failed"),
            httpx.NetworkError("Connection failed"),
            mock_response
        ]
        
        response = client.request("GET", "/test")
        
        assert response.json() == {"result": "success"}
        assert mock_request.call_count == 3


{% if 'async' in features %}
@pytest.mark.asyncio
class TestAsyncClient:
    """Test async client functionality"""
    
    async def test_async_context_manager(self):
        """Test async context manager"""
        async with Client(api_key="test-key") as client:
            assert client.async_client is not None
    
    @patch('httpx.AsyncClient.request')
    async def test_async_request(self, mock_request, mock_response):
        """Test async request"""
        mock_request.return_value = mock_response
        
        async with Client(api_key="test-key") as client:
            response = await client.arequest("GET", "/test")
            assert response.json() == {"result": "success"}
{% endif %}
'''
        return self.env.from_string(template).render(
            config=self.config,
            features=[f.value for f in self.config.features]
        )
    
    def _generate_model_tests(self) -> str:
        """Generate test_models.py"""
        template = '''"""
Tests for data models
"""

import pytest
from datetime import datetime

{% if 'models' in features %}
from {{ config.package_name }} import models


class TestModels:
    """Test data models"""
    
    {% for name, model in models.items()[:3] %}
    def test_{{ name|snake_case }}_creation(self):
        """Test {{ name }} model creation"""
        data = {
            {% for prop_name in model.properties.keys()[:3] %}
            "{{ prop_name }}": "test_value",
            {% endfor %}
        }
        
        obj = models.{{ name }}(**data)
        {% for prop_name in model.properties.keys()[:3] %}
        assert obj.{{ prop_name|snake_case }} == "test_value"
        {% endfor %}
    
    {% if 'validation' in features and 'pydantic' in dependencies %}
    def test_{{ name|snake_case }}_validation(self):
        """Test {{ name }} model validation"""
        with pytest.raises(ValueError):
            models.{{ name }}()  # Missing required fields
    {% endif %}
    
    {% if 'serialization' in features %}
    def test_{{ name|snake_case }}_serialization(self):
        """Test {{ name }} model serialization"""
        data = {
            {% for prop_name in model.properties.keys()[:3] %}
            "{{ prop_name }}": "test_value",
            {% endfor %}
        }
        
        obj = models.{{ name }}(**data)
        {% if 'pydantic' in dependencies %}
        serialized = obj.dict()
        {% else %}
        serialized = obj.to_dict()
        {% endif %}
        
        assert serialized == data
    {% endif %}
    
    {% endfor %}
{% endif %}
'''
        return self.env.from_string(template).render(
            config=self.config,
            models=self.models,
            features=[f.value for f in self.config.features],
            dependencies=self.config.dependencies
        )
    
    def _generate_test_config(self) -> str:
        """Generate conftest.py"""
        return '''"""
Pytest configuration and fixtures
"""

import pytest
import os
from unittest.mock import patch


@pytest.fixture(autouse=True)
def reset_env():
    """Reset environment variables for each test"""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_env():
    """Mock environment variables"""
    with patch.dict(os.environ, {
        "API_BASE_URL": "https://test.example.com",
        "API_KEY": "test-api-key"
    }):
        yield
'''
    
    def _format_python_code(self, code: str) -> str:
        """Format Python code"""
        try:
            # Format with black
            code = black.format_str(code, mode=black.Mode())
        except:
            try:
                # Fallback to autopep8
                code = autopep8.fix_code(code)
            except:
                pass
        
        try:
            # Sort imports
            code = isort.code(code)
        except:
            pass
        
        return code
    
    def _python_type(self, schema: Dict[str, Any]) -> str:
        """Convert OpenAPI type to Python type"""
        
        if not schema:
            return "Any"
        
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "List",
            "object": "Dict[str, Any]"
        }
        
        schema_type = schema.get("type", "object")
        
        if schema_type == "array":
            items_type = self._python_type(schema.get("items", {}))
            return f"List[{items_type}]"
        
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            return ref_name
        
        python_type = type_mapping.get(schema_type, "Any")
        
        # Handle nullable
        if schema.get("nullable", False):
            python_type = f"Optional[{python_type}]"
        
        return python_type
    
    def _get_model_name(self, request_body: Dict[str, Any]) -> str:
        """Extract model name from request body"""
        
        content = request_body.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})
        
        if "$ref" in schema:
            return schema["$ref"].split("/")[-1]
        
        return "Dict[str, Any]"
    
    def _get_response_model(self, response: Dict[str, Any]) -> str:
        """Extract model name from response"""
        
        content = response.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})
        
        if "$ref" in schema:
            return schema["$ref"].split("/")[-1]
        
        return "Dict[str, Any]"


class SDKPackager:
    """Package SDK for distribution"""
    
    def __init__(self, files: Dict[str, str], config: SDKConfig):
        self.files = files
        self.config = config
    
    def package(self, output_format: str = "zip") -> bytes:
        """Package SDK files"""
        
        if output_format == "zip":
            return self._create_zip()
        elif output_format == "tar":
            return self._create_tar()
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def _create_zip(self) -> bytes:
        """Create ZIP archive"""
        
        import io
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path, content in self.files.items():
                zf.writestr(path, content)
        
        return buffer.getvalue()
    
    def _create_tar(self) -> bytes:
        """Create TAR archive"""
        
        import io
        buffer = io.BytesIO()
        
        with tarfile.open(fileobj=buffer, mode='w:gz') as tf:
            for path, content in self.files.items():
                info = tarfile.TarInfo(name=path)
                info.size = len(content.encode())
                tf.addfile(tarinfo=info, fileobj=io.BytesIO(content.encode()))
        
        return buffer.getvalue()


class SDKGenerator:
    """Main SDK generator"""
    
    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        self.parser = OpenAPIParser(spec_path)
        self.generators = {
            Language.PYTHON: PythonSDKGenerator,
            # Add more language generators here
        }
    
    def generate(self, config: SDKConfig) -> Dict[str, str]:
        """Generate SDK for specified language"""
        
        # Parse OpenAPI spec
        endpoints, models = self.parser.parse()
        
        # Get appropriate generator
        generator_class = self.generators.get(config.language)
        if not generator_class:
            raise ValueError(f"Unsupported language: {config.language}")
        
        # Generate SDK
        generator = generator_class(config, self.parser.spec)
        files = generator.generate(endpoints, models)
        
        # Save to output directory if specified
        if config.output_dir:
            self._save_files(files, config.output_dir)
        
        return files
    
    def _save_files(self, files: Dict[str, str], output_dir: Path):
        """Save generated files to directory"""
        
        output_dir = Path(output_dir)
        
        for path, content in files.items():
            file_path = output_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"SDK generated at: {output_dir}")


# Example usage
async def generate_sdks():
    """Generate SDKs for multiple languages"""
    
    # Generate Python SDK
    python_config = SDKConfig(
        language=Language.PYTHON,
        package_name="openpolicy-client",
        version="1.0.0",
        features={
            SDKFeature.CLIENT,
            SDKFeature.MODELS,
            SDKFeature.AUTHENTICATION,
            SDKFeature.ASYNC,
            SDKFeature.RETRY,
            SDKFeature.RATE_LIMITING,
            SDKFeature.VALIDATION,
            SDKFeature.TESTING,
        },
        author="Open Policy Platform",
        license="MIT",
        repository_url="https://github.com/openpolicy/python-client",
        description="Python client for Open Policy Platform API",
        dependencies={"pydantic": "1.10.0"},
        include_examples=True,
        include_tests=True,
        output_dir=Path("generated/python")
    )
    
    generator = SDKGenerator("openapi.yaml")
    python_files = generator.generate(python_config)
    
    # Package SDK
    packager = SDKPackager(python_files, python_config)
    zip_data = packager.package("zip")
    
    with open("openpolicy-python-sdk.zip", "wb") as f:
        f.write(zip_data)
    
    print(f"Generated Python SDK with {len(python_files)} files")
    
    # Generate TypeScript SDK
    # typescript_config = SDKConfig(
    #     language=Language.TYPESCRIPT,
    #     package_name="@openpolicy/client",
    #     version="1.0.0",
    #     ...
    # )
    # typescript_files = generator.generate(typescript_config)


if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_sdks())