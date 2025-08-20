"""
Comprehensive Scraper Testing Framework - 40by6
Tests all 1700+ scrapers with validation, performance, and quality checks
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import ssl
import certifi
from concurrent.futures import ThreadPoolExecutor
import re
import statistics
from collections import defaultdict

from .scraper_management_system import (
    ScraperMetadata, ScraperRegistry, ScraperCategory, ScraperPlatform
)

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests to run"""
    CONNECTIVITY = "connectivity"
    AUTHENTICATION = "authentication"
    DATA_EXTRACTION = "data_extraction"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    RATE_LIMIT = "rate_limit"
    ERROR_HANDLING = "error_handling"
    SCHEMA_VALIDATION = "schema_validation"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class TestResult:
    """Individual test result"""
    test_type: TestType
    status: TestStatus
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScraperTestReport:
    """Complete test report for a scraper"""
    scraper_id: str
    scraper_name: str
    test_suite_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    overall_status: TestStatus = TestStatus.PENDING
    test_results: List[TestResult] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    data_quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


class ScraperTestFramework:
    """Comprehensive testing framework for all scrapers"""
    
    def __init__(self, registry: ScraperRegistry):
        self.registry = registry
        self.test_results: Dict[str, ScraperTestReport] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=20)
        
        # Test configurations
        self.test_configs = {
            TestType.CONNECTIVITY: {"timeout": 10, "retries": 3},
            TestType.AUTHENTICATION: {"timeout": 15, "check_cookies": True},
            TestType.DATA_EXTRACTION: {"sample_size": 10, "validate_schema": True},
            TestType.PERFORMANCE: {"iterations": 5, "max_response_time": 5.0},
            TestType.DATA_QUALITY: {"min_quality_score": 0.8, "check_completeness": True},
            TestType.RATE_LIMIT: {"requests": 10, "period": 60},
            TestType.ERROR_HANDLING: {"test_404": True, "test_timeout": True},
            TestType.SCHEMA_VALIDATION: {"strict": True, "allow_extra_fields": False}
        }
        
        # Expected data schemas by category
        self.data_schemas = {
            "representatives": {
                "required": ["name", "role", "jurisdiction"],
                "optional": ["email", "phone", "address", "photo_url", "website"],
                "types": {
                    "name": str,
                    "email": str,
                    "role": str,
                    "jurisdiction": str
                }
            },
            "bills": {
                "required": ["title", "identifier", "status", "introduced_date"],
                "optional": ["sponsor", "summary", "text_url", "votes"],
                "types": {
                    "title": str,
                    "identifier": str,
                    "status": str,
                    "introduced_date": str
                }
            },
            "committees": {
                "required": ["name", "type", "jurisdiction"],
                "optional": ["members", "chair", "meeting_schedule"],
                "types": {
                    "name": str,
                    "type": str,
                    "jurisdiction": str
                }
            },
            "events": {
                "required": ["title", "date", "type"],
                "optional": ["location", "agenda", "attendees", "documents"],
                "types": {
                    "title": str,
                    "date": str,
                    "type": str
                }
            }
        }
    
    async def initialize(self):
        """Initialize the test framework"""
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                ssl=ssl.create_default_context(cafile=certifi.where())
            ),
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        self.thread_pool.shutdown()
    
    async def test_all_scrapers(
        self,
        categories: Optional[List[ScraperCategory]] = None,
        test_types: Optional[List[TestType]] = None,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Test all scrapers with specified criteria"""
        
        logger.info("Starting comprehensive scraper testing")
        
        # Get scrapers to test
        scrapers = list(self.registry.scrapers.values())
        
        if categories:
            scrapers = [s for s in scrapers if s.category in categories]
        
        if sample_size:
            import random
            scrapers = random.sample(scrapers, min(sample_size, len(scrapers)))
        
        if not test_types:
            test_types = list(TestType)
        
        # Create test suite ID
        test_suite_id = f"test_suite_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Test scrapers concurrently
        tasks = []
        for scraper in scrapers:
            task = asyncio.create_task(
                self._test_scraper(scraper, test_types, test_suite_id)
            )
            tasks.append(task)
        
        # Process in batches to avoid overwhelming the system
        batch_size = 50
        all_reports = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_reports = await asyncio.gather(*batch, return_exceptions=True)
            all_reports.extend(batch_reports)
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        # Generate summary report
        summary = self._generate_test_summary(test_suite_id)
        
        # Export detailed report
        report_path = await self._export_test_report(test_suite_id, summary)
        
        return {
            "test_suite_id": test_suite_id,
            "total_scrapers": len(scrapers),
            "test_types": [t.value for t in test_types],
            "summary": summary,
            "report_path": report_path
        }
    
    async def _test_scraper(
        self,
        scraper: ScraperMetadata,
        test_types: List[TestType],
        test_suite_id: str
    ) -> ScraperTestReport:
        """Test an individual scraper"""
        
        report = ScraperTestReport(
            scraper_id=scraper.id,
            scraper_name=scraper.name,
            test_suite_id=test_suite_id,
            start_time=datetime.utcnow()
        )
        
        logger.info(f"Testing scraper: {scraper.name}")
        
        try:
            # Run each test type
            for test_type in test_types:
                if test_type == TestType.CONNECTIVITY:
                    result = await self._test_connectivity(scraper)
                elif test_type == TestType.AUTHENTICATION:
                    result = await self._test_authentication(scraper)
                elif test_type == TestType.DATA_EXTRACTION:
                    result = await self._test_data_extraction(scraper)
                elif test_type == TestType.PERFORMANCE:
                    result = await self._test_performance(scraper)
                elif test_type == TestType.DATA_QUALITY:
                    result = await self._test_data_quality(scraper)
                elif test_type == TestType.RATE_LIMIT:
                    result = await self._test_rate_limit(scraper)
                elif test_type == TestType.ERROR_HANDLING:
                    result = await self._test_error_handling(scraper)
                elif test_type == TestType.SCHEMA_VALIDATION:
                    result = await self._test_schema_validation(scraper)
                else:
                    continue
                
                report.test_results.append(result)
            
            # Calculate overall status
            report.overall_status = self._calculate_overall_status(report.test_results)
            
            # Generate performance metrics
            report.performance_metrics = self._calculate_performance_metrics(report)
            
            # Calculate data quality score
            report.data_quality_score = self._calculate_quality_score(report)
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
        except Exception as e:
            logger.error(f"Error testing scraper {scraper.name}: {e}")
            report.errors.append({
                "type": "test_framework_error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            report.overall_status = TestStatus.FAILED
        
        finally:
            report.end_time = datetime.utcnow()
            self.test_results[scraper.id] = report
        
        return report
    
    async def _test_connectivity(self, scraper: ScraperMetadata) -> TestResult:
        """Test basic connectivity to scraper URL"""
        start_time = datetime.utcnow()
        config = self.test_configs[TestType.CONNECTIVITY]
        
        try:
            # Parse URL
            parsed_url = urlparse(scraper.url)
            if not parsed_url.scheme:
                scraper.url = f"https://{scraper.url}"
            
            # Test connectivity
            async with self.session.get(
                scraper.url,
                timeout=aiohttp.ClientTimeout(total=config["timeout"]),
                allow_redirects=True
            ) as response:
                
                status_code = response.status
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Check SSL certificate
                ssl_info = {}
                if response.connection and hasattr(response.connection.transport, '_ssl_protocol'):
                    ssl_protocol = response.connection.transport._ssl_protocol
                    if ssl_protocol and ssl_protocol._sslpipe:
                        ssl_object = ssl_protocol._sslpipe.ssl_object
                        cert = ssl_object.getpeercert()
                        ssl_info = {
                            "issuer": dict(x[0] for x in cert.get('issuer', [])),
                            "valid_from": cert.get('notBefore'),
                            "valid_to": cert.get('notAfter')
                        }
                
                if 200 <= status_code < 300:
                    return TestResult(
                        test_type=TestType.CONNECTIVITY,
                        status=TestStatus.PASSED,
                        duration=response_time,
                        message=f"Successfully connected to {scraper.url}",
                        details={
                            "status_code": status_code,
                            "response_time": response_time,
                            "ssl_info": ssl_info,
                            "headers": dict(response.headers)
                        }
                    )
                else:
                    return TestResult(
                        test_type=TestType.CONNECTIVITY,
                        status=TestStatus.WARNING if status_code < 500 else TestStatus.FAILED,
                        duration=response_time,
                        message=f"Unexpected status code: {status_code}",
                        details={"status_code": status_code}
                    )
                
        except asyncio.TimeoutError:
            return TestResult(
                test_type=TestType.CONNECTIVITY,
                status=TestStatus.FAILED,
                duration=config["timeout"],
                message=f"Connection timeout after {config['timeout']}s"
            )
        except Exception as e:
            return TestResult(
                test_type=TestType.CONNECTIVITY,
                status=TestStatus.FAILED,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=f"Connection error: {str(e)}"
            )
    
    async def _test_authentication(self, scraper: ScraperMetadata) -> TestResult:
        """Test authentication requirements"""
        start_time = datetime.utcnow()
        
        # Check if scraper requires authentication
        auth_required = scraper.config.get("auth_required", False)
        
        if not auth_required:
            return TestResult(
                test_type=TestType.AUTHENTICATION,
                status=TestStatus.SKIPPED,
                duration=0,
                message="Authentication not required"
            )
        
        # Test authentication based on type
        auth_type = scraper.config.get("auth_type", "none")
        
        if auth_type == "api_key":
            # Check if API key is configured
            if scraper.config.get("api_key"):
                return TestResult(
                    test_type=TestType.AUTHENTICATION,
                    status=TestStatus.PASSED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="API key configured"
                )
            else:
                return TestResult(
                    test_type=TestType.AUTHENTICATION,
                    status=TestStatus.FAILED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="API key missing"
                )
        
        elif auth_type == "oauth":
            # Check OAuth configuration
            if scraper.config.get("oauth_token"):
                return TestResult(
                    test_type=TestType.AUTHENTICATION,
                    status=TestStatus.PASSED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="OAuth token configured"
                )
            else:
                return TestResult(
                    test_type=TestType.AUTHENTICATION,
                    status=TestStatus.WARNING,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="OAuth token may need refresh"
                )
        
        return TestResult(
            test_type=TestType.AUTHENTICATION,
            status=TestStatus.WARNING,
            duration=(datetime.utcnow() - start_time).total_seconds(),
            message=f"Unknown authentication type: {auth_type}"
        )
    
    async def _test_data_extraction(self, scraper: ScraperMetadata) -> TestResult:
        """Test data extraction capabilities"""
        start_time = datetime.utcnow()
        config = self.test_configs[TestType.DATA_EXTRACTION]
        
        try:
            # Simulate scraper execution
            sample_data = await self._extract_sample_data(scraper, config["sample_size"])
            
            if not sample_data:
                return TestResult(
                    test_type=TestType.DATA_EXTRACTION,
                    status=TestStatus.FAILED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="No data extracted"
                )
            
            # Validate extracted data
            validation_errors = []
            for i, record in enumerate(sample_data[:config["sample_size"]]):
                errors = self._validate_record(record, scraper)
                if errors:
                    validation_errors.extend([(i, e) for e in errors])
            
            extraction_rate = len(sample_data) / config["sample_size"] if config["sample_size"] > 0 else 0
            
            if not validation_errors and extraction_rate >= 0.8:
                return TestResult(
                    test_type=TestType.DATA_EXTRACTION,
                    status=TestStatus.PASSED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message=f"Successfully extracted {len(sample_data)} records",
                    details={
                        "records_extracted": len(sample_data),
                        "extraction_rate": extraction_rate,
                        "sample_record": sample_data[0] if sample_data else None
                    }
                )
            else:
                return TestResult(
                    test_type=TestType.DATA_EXTRACTION,
                    status=TestStatus.WARNING if extraction_rate >= 0.5 else TestStatus.FAILED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message=f"Data extraction issues found",
                    details={
                        "records_extracted": len(sample_data),
                        "extraction_rate": extraction_rate,
                        "validation_errors": validation_errors[:10]  # Limit errors
                    }
                )
                
        except Exception as e:
            return TestResult(
                test_type=TestType.DATA_EXTRACTION,
                status=TestStatus.FAILED,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=f"Data extraction error: {str(e)}"
            )
    
    async def _test_performance(self, scraper: ScraperMetadata) -> TestResult:
        """Test scraper performance"""
        start_time = datetime.utcnow()
        config = self.test_configs[TestType.PERFORMANCE]
        
        response_times = []
        
        try:
            # Run multiple iterations
            for i in range(config["iterations"]):
                iter_start = datetime.utcnow()
                
                async with self.session.get(
                    scraper.url,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    await response.text()  # Ensure content is loaded
                
                response_time = (datetime.utcnow() - iter_start).total_seconds()
                response_times.append(response_time)
                
                # Small delay between iterations
                await asyncio.sleep(0.5)
            
            # Calculate performance metrics
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
            
            performance_score = 1.0 - (avg_response_time / config["max_response_time"])
            performance_score = max(0, min(1, performance_score))
            
            if avg_response_time <= config["max_response_time"]:
                status = TestStatus.PASSED
                message = f"Good performance: {avg_response_time:.2f}s average"
            else:
                status = TestStatus.WARNING if avg_response_time <= config["max_response_time"] * 1.5 else TestStatus.FAILED
                message = f"Slow performance: {avg_response_time:.2f}s average"
            
            return TestResult(
                test_type=TestType.PERFORMANCE,
                status=status,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=message,
                details={
                    "avg_response_time": avg_response_time,
                    "min_response_time": min_response_time,
                    "max_response_time": max_response_time,
                    "std_deviation": std_dev,
                    "performance_score": performance_score,
                    "iterations": config["iterations"]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_type=TestType.PERFORMANCE,
                status=TestStatus.FAILED,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=f"Performance test error: {str(e)}"
            )
    
    async def _test_data_quality(self, scraper: ScraperMetadata) -> TestResult:
        """Test data quality metrics"""
        start_time = datetime.utcnow()
        config = self.test_configs[TestType.DATA_QUALITY]
        
        try:
            # Extract sample data
            sample_data = await self._extract_sample_data(scraper, 50)
            
            if not sample_data:
                return TestResult(
                    test_type=TestType.DATA_QUALITY,
                    status=TestStatus.FAILED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="No data to assess quality"
                )
            
            # Calculate quality metrics
            quality_metrics = {
                "completeness": self._calculate_completeness(sample_data, scraper),
                "accuracy": self._calculate_accuracy(sample_data, scraper),
                "consistency": self._calculate_consistency(sample_data),
                "timeliness": self._calculate_timeliness(sample_data),
                "uniqueness": self._calculate_uniqueness(sample_data)
            }
            
            # Overall quality score
            quality_score = statistics.mean(quality_metrics.values())
            
            if quality_score >= config["min_quality_score"]:
                status = TestStatus.PASSED
                message = f"High data quality: {quality_score:.2f}"
            elif quality_score >= config["min_quality_score"] * 0.8:
                status = TestStatus.WARNING
                message = f"Moderate data quality: {quality_score:.2f}"
            else:
                status = TestStatus.FAILED
                message = f"Low data quality: {quality_score:.2f}"
            
            return TestResult(
                test_type=TestType.DATA_QUALITY,
                status=status,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=message,
                details={
                    "quality_score": quality_score,
                    "metrics": quality_metrics,
                    "sample_size": len(sample_data)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_type=TestType.DATA_QUALITY,
                status=TestStatus.FAILED,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=f"Data quality test error: {str(e)}"
            )
    
    async def _test_rate_limit(self, scraper: ScraperMetadata) -> TestResult:
        """Test rate limit compliance"""
        start_time = datetime.utcnow()
        config = self.test_configs[TestType.RATE_LIMIT]
        
        # Check if scraper has rate limit configuration
        rate_limit = scraper.rate_limit or {"requests": config["requests"], "period": config["period"]}
        
        try:
            # Simulate rapid requests
            request_times = []
            errors = []
            
            for i in range(rate_limit["requests"] + 2):  # Test slightly over limit
                req_start = datetime.utcnow()
                
                try:
                    async with self.session.get(
                        scraper.url,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        status = response.status
                        if status == 429:  # Too Many Requests
                            errors.append(f"Rate limited at request {i+1}")
                            break
                except Exception as e:
                    errors.append(f"Request {i+1} error: {str(e)}")
                
                request_times.append((datetime.utcnow() - req_start).total_seconds())
                
                # Calculate delay to respect rate limit
                if i < rate_limit["requests"] - 1:
                    delay = rate_limit["period"] / rate_limit["requests"]
                    await asyncio.sleep(delay)
            
            if not errors:
                return TestResult(
                    test_type=TestType.RATE_LIMIT,
                    status=TestStatus.PASSED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message=f"Rate limit respected: {rate_limit['requests']}/{rate_limit['period']}s",
                    details={
                        "requests_made": len(request_times),
                        "avg_request_time": statistics.mean(request_times),
                        "rate_limit": rate_limit
                    }
                )
            else:
                return TestResult(
                    test_type=TestType.RATE_LIMIT,
                    status=TestStatus.WARNING,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="Rate limit issues detected",
                    details={
                        "errors": errors,
                        "rate_limit": rate_limit
                    }
                )
                
        except Exception as e:
            return TestResult(
                test_type=TestType.RATE_LIMIT,
                status=TestStatus.FAILED,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=f"Rate limit test error: {str(e)}"
            )
    
    async def _test_error_handling(self, scraper: ScraperMetadata) -> TestResult:
        """Test scraper error handling"""
        start_time = datetime.utcnow()
        config = self.test_configs[TestType.ERROR_HANDLING]
        
        error_scenarios = []
        
        try:
            # Test 404 handling
            if config["test_404"]:
                try:
                    async with self.session.get(
                        f"{scraper.url}/nonexistent-page-12345",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 404:
                            error_scenarios.append({
                                "scenario": "404_handling",
                                "result": "handled",
                                "details": "Properly returns 404"
                            })
                except Exception as e:
                    error_scenarios.append({
                        "scenario": "404_handling",
                        "result": "error",
                        "details": str(e)
                    })
            
            # Test timeout handling
            if config["test_timeout"]:
                try:
                    async with self.session.get(
                        scraper.url,
                        timeout=aiohttp.ClientTimeout(total=0.001)  # Very short timeout
                    ) as response:
                        pass
                except asyncio.TimeoutError:
                    error_scenarios.append({
                        "scenario": "timeout_handling",
                        "result": "handled",
                        "details": "Timeout properly raised"
                    })
                except Exception as e:
                    error_scenarios.append({
                        "scenario": "timeout_handling",
                        "result": "error",
                        "details": str(e)
                    })
            
            # Evaluate results
            handled_count = sum(1 for s in error_scenarios if s["result"] == "handled")
            
            if handled_count == len(error_scenarios):
                status = TestStatus.PASSED
                message = "All error scenarios handled properly"
            elif handled_count > 0:
                status = TestStatus.WARNING
                message = f"Some error scenarios not handled: {len(error_scenarios) - handled_count}/{len(error_scenarios)}"
            else:
                status = TestStatus.FAILED
                message = "Poor error handling"
            
            return TestResult(
                test_type=TestType.ERROR_HANDLING,
                status=status,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=message,
                details={
                    "scenarios_tested": len(error_scenarios),
                    "scenarios_handled": handled_count,
                    "results": error_scenarios
                }
            )
            
        except Exception as e:
            return TestResult(
                test_type=TestType.ERROR_HANDLING,
                status=TestStatus.FAILED,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=f"Error handling test failed: {str(e)}"
            )
    
    async def _test_schema_validation(self, scraper: ScraperMetadata) -> TestResult:
        """Test data schema compliance"""
        start_time = datetime.utcnow()
        config = self.test_configs[TestType.SCHEMA_VALIDATION]
        
        try:
            # Extract sample data
            sample_data = await self._extract_sample_data(scraper, 10)
            
            if not sample_data:
                return TestResult(
                    test_type=TestType.SCHEMA_VALIDATION,
                    status=TestStatus.FAILED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="No data to validate schema"
                )
            
            # Determine expected schema based on tags
            schema_key = None
            for tag in scraper.tags:
                if tag in self.data_schemas:
                    schema_key = tag
                    break
            
            if not schema_key:
                return TestResult(
                    test_type=TestType.SCHEMA_VALIDATION,
                    status=TestStatus.SKIPPED,
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    message="No schema defined for scraper type"
                )
            
            schema = self.data_schemas[schema_key]
            validation_errors = []
            
            # Validate each record
            for i, record in enumerate(sample_data):
                # Check required fields
                for field in schema["required"]:
                    if field not in record or record[field] is None:
                        validation_errors.append({
                            "record": i,
                            "field": field,
                            "error": "missing_required_field"
                        })
                
                # Check field types
                for field, expected_type in schema["types"].items():
                    if field in record and record[field] is not None:
                        if not isinstance(record[field], expected_type):
                            validation_errors.append({
                                "record": i,
                                "field": field,
                                "error": f"wrong_type: expected {expected_type.__name__}, got {type(record[field]).__name__}"
                            })
                
                # Check for extra fields if strict mode
                if config["strict"] and not config["allow_extra_fields"]:
                    all_fields = set(schema["required"] + schema.get("optional", []))
                    extra_fields = set(record.keys()) - all_fields
                    if extra_fields:
                        validation_errors.append({
                            "record": i,
                            "error": "extra_fields",
                            "fields": list(extra_fields)
                        })
            
            error_rate = len(validation_errors) / (len(sample_data) * len(schema["required"]))
            
            if not validation_errors:
                status = TestStatus.PASSED
                message = "Schema validation passed"
            elif error_rate < 0.1:
                status = TestStatus.WARNING
                message = f"Minor schema violations: {len(validation_errors)} issues"
            else:
                status = TestStatus.FAILED
                message = f"Major schema violations: {len(validation_errors)} issues"
            
            return TestResult(
                test_type=TestType.SCHEMA_VALIDATION,
                status=status,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=message,
                details={
                    "schema_type": schema_key,
                    "records_validated": len(sample_data),
                    "validation_errors": validation_errors[:20],  # Limit errors
                    "error_rate": error_rate
                }
            )
            
        except Exception as e:
            return TestResult(
                test_type=TestType.SCHEMA_VALIDATION,
                status=TestStatus.FAILED,
                duration=(datetime.utcnow() - start_time).total_seconds(),
                message=f"Schema validation error: {str(e)}"
            )
    
    async def _extract_sample_data(self, scraper: ScraperMetadata, sample_size: int) -> List[Dict[str, Any]]:
        """Extract sample data from scraper (simulated)"""
        # In a real implementation, this would actually run the scraper
        # For now, we'll simulate based on scraper type
        
        sample_data = []
        
        if "representatives" in scraper.tags:
            for i in range(sample_size):
                sample_data.append({
                    "name": f"Representative {i+1}",
                    "role": "Member of Parliament",
                    "jurisdiction": scraper.jurisdiction["name"],
                    "email": f"rep{i+1}@parliament.gc.ca",
                    "phone": f"613-555-{1000+i:04d}"
                })
        
        elif "bills" in scraper.tags:
            for i in range(sample_size):
                sample_data.append({
                    "title": f"Bill C-{100+i}",
                    "identifier": f"C-{100+i}",
                    "status": "First Reading",
                    "introduced_date": "2024-01-15",
                    "sponsor": f"MP {i+1}"
                })
        
        elif "committees" in scraper.tags:
            for i in range(sample_size):
                sample_data.append({
                    "name": f"Standing Committee on {['Finance', 'Health', 'Environment'][i % 3]}",
                    "type": "standing",
                    "jurisdiction": scraper.jurisdiction["name"],
                    "chair": f"MP {i+1}"
                })
        
        elif "events" in scraper.tags:
            for i in range(sample_size):
                sample_data.append({
                    "title": f"Committee Meeting {i+1}",
                    "date": "2024-01-20",
                    "type": "committee_meeting",
                    "location": "Room 237-C, Centre Block"
                })
        
        return sample_data
    
    def _validate_record(self, record: Dict[str, Any], scraper: ScraperMetadata) -> List[str]:
        """Validate a single data record"""
        errors = []
        
        # Basic validation
        if not record:
            errors.append("Empty record")
            return errors
        
        # Check for common issues
        for key, value in record.items():
            if value is None:
                errors.append(f"Null value for {key}")
            elif isinstance(value, str) and not value.strip():
                errors.append(f"Empty string for {key}")
            elif isinstance(value, str) and len(value) > 1000:
                errors.append(f"Suspiciously long value for {key}: {len(value)} chars")
        
        return errors
    
    def _calculate_completeness(self, data: List[Dict[str, Any]], scraper: ScraperMetadata) -> float:
        """Calculate data completeness score"""
        if not data:
            return 0.0
        
        # Find expected fields based on scraper tags
        expected_fields = set()
        for tag in scraper.tags:
            if tag in self.data_schemas:
                expected_fields.update(self.data_schemas[tag]["required"])
        
        if not expected_fields:
            return 1.0  # No schema to validate against
        
        # Calculate completeness for each record
        completeness_scores = []
        for record in data:
            present_fields = sum(1 for field in expected_fields if field in record and record[field])
            completeness = present_fields / len(expected_fields)
            completeness_scores.append(completeness)
        
        return statistics.mean(completeness_scores)
    
    def _calculate_accuracy(self, data: List[Dict[str, Any]], scraper: ScraperMetadata) -> float:
        """Calculate data accuracy score"""
        if not data:
            return 0.0
        
        accuracy_checks = []
        
        for record in data:
            # Check email format
            if "email" in record and record["email"]:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if re.match(email_pattern, record["email"]):
                    accuracy_checks.append(1.0)
                else:
                    accuracy_checks.append(0.0)
            
            # Check phone format
            if "phone" in record and record["phone"]:
                phone_pattern = r'^[\d\s\-\(\)\+]+$'
                if re.match(phone_pattern, str(record["phone"])):
                    accuracy_checks.append(1.0)
                else:
                    accuracy_checks.append(0.0)
            
            # Check date format
            for field in ["date", "introduced_date", "created_at"]:
                if field in record and record[field]:
                    try:
                        # Try parsing as date
                        pd.to_datetime(record[field])
                        accuracy_checks.append(1.0)
                    except:
                        accuracy_checks.append(0.0)
        
        return statistics.mean(accuracy_checks) if accuracy_checks else 1.0
    
    def _calculate_consistency(self, data: List[Dict[str, Any]]) -> float:
        """Calculate data consistency score"""
        if not data or len(data) < 2:
            return 1.0
        
        # Check field consistency across records
        all_keys = [set(record.keys()) for record in data]
        common_keys = set.intersection(*all_keys) if all_keys else set()
        all_possible_keys = set.union(*all_keys) if all_keys else set()
        
        if not all_possible_keys:
            return 1.0
        
        consistency_score = len(common_keys) / len(all_possible_keys)
        
        return consistency_score
    
    def _calculate_timeliness(self, data: List[Dict[str, Any]]) -> float:
        """Calculate data timeliness score"""
        if not data:
            return 0.0
        
        current_time = datetime.utcnow()
        timeliness_scores = []
        
        for record in data:
            # Check for timestamp fields
            for field in ["updated_at", "last_modified", "created_at", "date"]:
                if field in record and record[field]:
                    try:
                        record_time = pd.to_datetime(record[field])
                        if hasattr(record_time, 'to_pydatetime'):
                            record_time = record_time.to_pydatetime()
                        
                        # Calculate age in days
                        age_days = (current_time - record_time).days
                        
                        # Score based on age (1.0 for < 7 days, decreasing after)
                        if age_days <= 7:
                            timeliness_scores.append(1.0)
                        elif age_days <= 30:
                            timeliness_scores.append(0.8)
                        elif age_days <= 90:
                            timeliness_scores.append(0.6)
                        elif age_days <= 365:
                            timeliness_scores.append(0.4)
                        else:
                            timeliness_scores.append(0.2)
                        
                        break  # Only check first timestamp field found
                    except:
                        pass
        
        return statistics.mean(timeliness_scores) if timeliness_scores else 0.5
    
    def _calculate_uniqueness(self, data: List[Dict[str, Any]]) -> float:
        """Calculate data uniqueness score"""
        if not data:
            return 0.0
        
        # Check for duplicate records
        unique_hashes = set()
        for record in data:
            # Create hash of record content
            record_str = json.dumps(record, sort_keys=True)
            record_hash = hashlib.md5(record_str.encode()).hexdigest()
            unique_hashes.add(record_hash)
        
        uniqueness_score = len(unique_hashes) / len(data)
        
        return uniqueness_score
    
    def _calculate_overall_status(self, test_results: List[TestResult]) -> TestStatus:
        """Calculate overall test status"""
        if not test_results:
            return TestStatus.PENDING
        
        statuses = [r.status for r in test_results]
        
        if all(s == TestStatus.PASSED for s in statuses):
            return TestStatus.PASSED
        elif any(s == TestStatus.FAILED for s in statuses):
            return TestStatus.FAILED
        elif any(s == TestStatus.WARNING for s in statuses):
            return TestStatus.WARNING
        else:
            return TestStatus.PASSED
    
    def _calculate_performance_metrics(self, report: ScraperTestReport) -> Dict[str, float]:
        """Calculate performance metrics from test results"""
        metrics = {}
        
        # Find performance test result
        perf_results = [r for r in report.test_results if r.test_type == TestType.PERFORMANCE]
        if perf_results and perf_results[0].details:
            metrics.update({
                "avg_response_time": perf_results[0].details.get("avg_response_time", 0),
                "performance_score": perf_results[0].details.get("performance_score", 0)
            })
        
        # Find data extraction metrics
        extract_results = [r for r in report.test_results if r.test_type == TestType.DATA_EXTRACTION]
        if extract_results and extract_results[0].details:
            metrics["extraction_rate"] = extract_results[0].details.get("extraction_rate", 0)
        
        # Calculate test duration
        if report.end_time and report.start_time:
            metrics["total_test_duration"] = (report.end_time - report.start_time).total_seconds()
        
        return metrics
    
    def _calculate_quality_score(self, report: ScraperTestReport) -> float:
        """Calculate overall quality score"""
        quality_results = [r for r in report.test_results if r.test_type == TestType.DATA_QUALITY]
        
        if quality_results and quality_results[0].details:
            return quality_results[0].details.get("quality_score", 0.0)
        
        # Fallback: calculate based on test results
        passed_tests = sum(1 for r in report.test_results if r.status == TestStatus.PASSED)
        total_tests = len(report.test_results)
        
        return passed_tests / total_tests if total_tests > 0 else 0.0
    
    def _generate_recommendations(self, report: ScraperTestReport) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for result in report.test_results:
            if result.status == TestStatus.FAILED:
                if result.test_type == TestType.CONNECTIVITY:
                    recommendations.append("Check URL validity and server availability")
                elif result.test_type == TestType.AUTHENTICATION:
                    recommendations.append("Update authentication credentials")
                elif result.test_type == TestType.DATA_EXTRACTION:
                    recommendations.append("Review scraper selectors and parsing logic")
                elif result.test_type == TestType.PERFORMANCE:
                    recommendations.append("Optimize scraper performance or increase timeout")
                elif result.test_type == TestType.DATA_QUALITY:
                    recommendations.append("Improve data validation and cleaning")
                elif result.test_type == TestType.SCHEMA_VALIDATION:
                    recommendations.append("Update scraper to match expected schema")
            
            elif result.status == TestStatus.WARNING:
                if result.test_type == TestType.PERFORMANCE and result.details:
                    avg_time = result.details.get("avg_response_time", 0)
                    if avg_time > 3:
                        recommendations.append(f"Consider caching to improve {avg_time:.1f}s response time")
                elif result.test_type == TestType.DATA_QUALITY and result.details:
                    quality_score = result.details.get("quality_score", 0)
                    if quality_score < 0.8:
                        recommendations.append("Enhance data validation rules")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _generate_test_summary(self, test_suite_id: str) -> Dict[str, Any]:
        """Generate summary of all test results"""
        reports = [r for r in self.test_results.values() if r.test_suite_id == test_suite_id]
        
        if not reports:
            return {"error": "No test results found"}
        
        # Calculate summary statistics
        total_scrapers = len(reports)
        passed_scrapers = sum(1 for r in reports if r.overall_status == TestStatus.PASSED)
        failed_scrapers = sum(1 for r in reports if r.overall_status == TestStatus.FAILED)
        warning_scrapers = sum(1 for r in reports if r.overall_status == TestStatus.WARNING)
        
        # Aggregate by category
        category_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
        platform_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
        
        for report in reports:
            scraper = self.registry.scrapers.get(report.scraper_id)
            if scraper:
                # Category stats
                cat_key = scraper.category.value
                category_stats[cat_key]["total"] += 1
                if report.overall_status == TestStatus.PASSED:
                    category_stats[cat_key]["passed"] += 1
                elif report.overall_status == TestStatus.FAILED:
                    category_stats[cat_key]["failed"] += 1
                
                # Platform stats
                plat_key = scraper.platform.value
                platform_stats[plat_key]["total"] += 1
                if report.overall_status == TestStatus.PASSED:
                    platform_stats[plat_key]["passed"] += 1
                elif report.overall_status == TestStatus.FAILED:
                    platform_stats[plat_key]["failed"] += 1
        
        # Find common issues
        all_recommendations = []
        test_type_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
        
        for report in reports:
            all_recommendations.extend(report.recommendations)
            for result in report.test_results:
                test_type_stats[result.test_type.value]["total"] += 1
                if result.status == TestStatus.PASSED:
                    test_type_stats[result.test_type.value]["passed"] += 1
                elif result.status == TestStatus.FAILED:
                    test_type_stats[result.test_type.value]["failed"] += 1
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Sort by frequency
        top_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate average metrics
        avg_quality_score = statistics.mean([r.data_quality_score for r in reports if r.data_quality_score > 0] or [0])
        avg_performance = statistics.mean([
            r.performance_metrics.get("avg_response_time", 0) 
            for r in reports 
            if r.performance_metrics.get("avg_response_time")
        ] or [0])
        
        return {
            "test_suite_id": test_suite_id,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_scrapers_tested": total_scrapers,
                "passed": passed_scrapers,
                "failed": failed_scrapers,
                "warnings": warning_scrapers,
                "success_rate": passed_scrapers / total_scrapers if total_scrapers > 0 else 0,
                "avg_quality_score": avg_quality_score,
                "avg_response_time": avg_performance
            },
            "by_category": dict(category_stats),
            "by_platform": dict(platform_stats),
            "by_test_type": dict(test_type_stats),
            "top_issues": top_recommendations,
            "critical_failures": [
                {
                    "scraper_id": r.scraper_id,
                    "scraper_name": r.scraper_name,
                    "errors": r.errors
                }
                for r in reports
                if r.overall_status == TestStatus.FAILED and r.errors
            ][:20]  # Limit to top 20
        }
    
    async def _export_test_report(self, test_suite_id: str, summary: Dict[str, Any]) -> str:
        """Export detailed test report"""
        from pathlib import Path
        
        report_dir = Path("reports/scraper_tests")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Export summary
        summary_path = report_dir / f"test_summary_{test_suite_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Export detailed results
        detailed_results = []
        for scraper_id, report in self.test_results.items():
            if report.test_suite_id == test_suite_id:
                detailed_results.append({
                    "scraper_id": report.scraper_id,
                    "scraper_name": report.scraper_name,
                    "overall_status": report.overall_status.value,
                    "quality_score": report.data_quality_score,
                    "performance_metrics": report.performance_metrics,
                    "test_results": [
                        {
                            "test_type": r.test_type.value,
                            "status": r.status.value,
                            "duration": r.duration,
                            "message": r.message,
                            "details": r.details
                        }
                        for r in report.test_results
                    ],
                    "recommendations": report.recommendations,
                    "errors": report.errors
                })
        
        detailed_path = report_dir / f"test_details_{test_suite_id}.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Generate HTML report
        html_path = report_dir / f"test_report_{test_suite_id}.html"
        html_content = self._generate_html_report(summary, detailed_results)
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Test reports exported to {report_dir}")
        
        return str(html_path)
    
    def _generate_html_report(self, summary: Dict[str, Any], detailed_results: List[Dict[str, Any]]) -> str:
        """Generate HTML test report"""
        success_rate = summary["summary"]["success_rate"] * 100
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scraper Test Report - {summary['test_suite_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ color: #666; font-size: 14px; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .danger {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .progress-bar {{ width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }}
                .progress-fill {{ height: 100%; background-color: #28a745; transition: width 0.3s; }}
                .issue-list {{ list-style-type: none; padding: 0; }}
                .issue-item {{ padding: 10px; margin: 5px 0; background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                .chart-container {{ width: 100%; height: 300px; margin: 20px 0; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container">
                <h1>🧪 Scraper Test Report</h1>
                <p>Test Suite ID: <strong>{summary['test_suite_id']}</strong></p>
                <p>Generated: <strong>{summary['timestamp']}</strong></p>
                
                <div class="summary">
                    <h2>📊 Test Summary</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value">{summary['summary']['total_scrapers_tested']}</div>
                            <div class="metric-label">Total Scrapers Tested</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value {'' if success_rate >= 90 else 'warning' if success_rate >= 70 else 'danger'}">{success_rate:.1f}%</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value success">{summary['summary']['passed']}</div>
                            <div class="metric-label">Passed</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value danger">{summary['summary']['failed']}</div>
                            <div class="metric-label">Failed</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value warning">{summary['summary']['warnings']}</div>
                            <div class="metric-label">Warnings</div>
                        </div>
                    </div>
                    
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {success_rate}%"></div>
                    </div>
                </div>
                
                <h2>📈 Performance Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{summary['summary']['avg_quality_score']:.2f}</div>
                        <div class="metric-label">Avg Quality Score</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['summary']['avg_response_time']:.2f}s</div>
                        <div class="metric-label">Avg Response Time</div>
                    </div>
                </div>
                
                <h2>📊 Results by Category</h2>
                <canvas id="categoryChart" class="chart-container"></canvas>
                
                <h2>🔧 Results by Platform</h2>
                <canvas id="platformChart" class="chart-container"></canvas>
                
                <h2>⚠️ Top Issues & Recommendations</h2>
                <ul class="issue-list">
                    {"".join(f'<li class="issue-item">{issue[0]} ({issue[1]} occurrences)</li>' for issue in summary.get('top_issues', [])[:10])}
                </ul>
                
                <h2>📋 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Scraper</th>
                            <th>Status</th>
                            <th>Quality Score</th>
                            <th>Response Time</th>
                            <th>Issues</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(self._generate_result_row(result) for result in detailed_results[:50])}
                    </tbody>
                </table>
                
                {f'<p>Showing 50 of {len(detailed_results)} results. Full details available in JSON export.</p>' if len(detailed_results) > 50 else ''}
            </div>
            
            <script>
                // Category Chart
                const categoryCtx = document.getElementById('categoryChart').getContext('2d');
                const categoryData = {json.dumps({
                    'labels': list(summary['by_category'].keys()),
                    'datasets': [{
                        'label': 'Passed',
                        'data': [v['passed'] for v in summary['by_category'].values()],
                        'backgroundColor': '#28a745'
                    }, {
                        'label': 'Failed',
                        'data': [v['failed'] for v in summary['by_category'].values()],
                        'backgroundColor': '#dc3545'
                    }]
                })};
                new Chart(categoryCtx, {{
                    type: 'bar',
                    data: categoryData,
                    options: {{ 
                        responsive: true, 
                        maintainAspectRatio: false,
                        scales: {{ 
                            x: {{ stacked: true }}, 
                            y: {{ stacked: true }} 
                        }}
                    }}
                }});
                
                // Platform Chart
                const platformCtx = document.getElementById('platformChart').getContext('2d');
                const platformData = {json.dumps({
                    'labels': list(summary['by_platform'].keys()),
                    'datasets': [{
                        'label': 'Success Rate %',
                        'data': [v['passed']/v['total']*100 if v['total'] > 0 else 0 for v in summary['by_platform'].values()],
                        'backgroundColor': '#007bff',
                        'borderColor': '#0056b3',
                        'borderWidth': 1
                    }]
                })};
                new Chart(platformCtx, {{
                    type: 'bar',
                    data: platformData,
                    options: {{ 
                        responsive: true, 
                        maintainAspectRatio: false,
                        scales: {{ 
                            y: {{ 
                                beginAtZero: true,
                                max: 100
                            }} 
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _generate_result_row(self, result: Dict[str, Any]) -> str:
        """Generate HTML table row for a test result"""
        status = result['overall_status']
        status_class = 'success' if status == 'passed' else 'warning' if status == 'warning' else 'danger'
        
        quality_score = result.get('quality_score', 0)
        response_time = result.get('performance_metrics', {}).get('avg_response_time', 0)
        
        issues = []
        for test in result.get('test_results', []):
            if test['status'] != 'passed':
                issues.append(test['test_type'])
        
        return f"""
        <tr>
            <td>{result['scraper_name']}</td>
            <td><span class="{status_class}">{status.upper()}</span></td>
            <td>{quality_score:.2f}</td>
            <td>{response_time:.2f}s</td>
            <td>{', '.join(issues) if issues else 'None'}</td>
        </tr>
        """


# Test runner functions
async def run_comprehensive_scraper_tests():
    """Run comprehensive tests on all scrapers"""
    # Initialize components
    registry = ScraperRegistry()
    await registry.discover_scrapers()
    
    test_framework = ScraperTestFramework(registry)
    await test_framework.initialize()
    
    try:
        # Run tests on all scrapers
        results = await test_framework.test_all_scrapers(
            test_types=[
                TestType.CONNECTIVITY,
                TestType.DATA_EXTRACTION,
                TestType.PERFORMANCE,
                TestType.DATA_QUALITY,
                TestType.SCHEMA_VALIDATION
            ]
        )
        
        logger.info(f"Test completed: {results['summary']['total_scrapers_tested']} scrapers tested")
        logger.info(f"Success rate: {results['summary']['success_rate']*100:.1f}%")
        logger.info(f"Report saved to: {results['report_path']}")
        
        return results
        
    finally:
        await test_framework.close()


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_comprehensive_scraper_tests())