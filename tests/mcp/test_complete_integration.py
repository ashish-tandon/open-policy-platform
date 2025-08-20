"""
Complete Integration Tests for MCP Stack - 40by6
Comprehensive testing of all components working together
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import aiohttp
import redis.asyncio as redis
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import os

# Import all MCP components
from backend.mcp.data_quality_agent import MCPDataQualityAgent
from backend.mcp.scraper_management_system import (
    MCPScraperManagementSystem, ScraperRegistry, ScraperOrchestrator,
    DataIngestionPipeline, ScraperMonitor, ScraperCategory, ScraperPlatform
)
from backend.mcp.scraper_testing_framework import ScraperTestFramework, TestType
from backend.mcp.ml_optimization_engine import MLOptimizationEngine, AutomatedOptimizer
from backend.mcp.real_time_analytics_engine import RealTimeAnalyticsEngine
from backend.mcp.automated_health_remediation import (
    AutomatedHealthRemediation, HealthIssueType, RemediationStatus
)
from backend.mcp.comprehensive_alerting_system import (
    ComprehensiveAlertingSystem, AlertSeverity, AlertChannel
)

# Test configuration
TEST_DATABASE_URL = os.getenv('TEST_DATABASE_URL', 'postgresql://test:test@localhost:5432/test_openpolicy')
TEST_REDIS_URL = os.getenv('TEST_REDIS_URL', 'redis://localhost:6379/1')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8001')


@pytest.fixture(scope='session')
async def test_database():
    """Setup test database"""
    engine = create_engine(TEST_DATABASE_URL)
    
    # Create tables
    with engine.connect() as conn:
        # Create necessary tables for testing
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scrapers (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                category VARCHAR,
                platform VARCHAR,
                status VARCHAR,
                failure_count INTEGER DEFAULT 0,
                last_run TIMESTAMP,
                last_success TIMESTAMP
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scraper_runs (
                id SERIAL PRIMARY KEY,
                scraper_id VARCHAR,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status VARCHAR,
                records_scraped INTEGER,
                data_quality_score FLOAT,
                errors JSONB
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS data_quality_issues (
                id SERIAL PRIMARY KEY,
                scraper_id VARCHAR,
                issue_type VARCHAR,
                severity VARCHAR,
                details JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        conn.commit()
    
    yield engine
    
    # Cleanup
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scrapers CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS scraper_runs CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS data_quality_issues CASCADE"))
        conn.commit()


@pytest.fixture
async def redis_client():
    """Redis client fixture"""
    client = await redis.from_url(TEST_REDIS_URL)
    yield client
    await client.flushdb()
    await client.close()


@pytest.fixture
async def mcp_system():
    """Complete MCP system fixture"""
    system = MCPScraperManagementSystem()
    await system.initialize()
    return system


class TestCompleteIntegration:
    """Test complete MCP stack integration"""
    
    @pytest.mark.asyncio
    async def test_full_scraper_lifecycle(self, mcp_system, test_database, redis_client):
        """Test complete scraper lifecycle from discovery to data quality"""
        # 1. Discover scrapers
        discovered = await mcp_system.registry.discover_scrapers()
        assert discovered > 0, "Should discover scrapers"
        
        # 2. Get a sample scraper
        scrapers = list(mcp_system.registry.scrapers.values())[:5]
        assert len(scrapers) > 0, "Should have scrapers"
        
        # 3. Schedule scrapers
        await mcp_system.orchestrator.schedule_scrapers([s.id for s in scrapers])
        
        # 4. Check queue
        queue_size = await redis_client.llen('scraper:queue')
        assert queue_size > 0, "Scrapers should be queued"
        
        # 5. Simulate execution
        for scraper in scrapers:
            # Create mock run
            with create_engine(TEST_DATABASE_URL).connect() as conn:
                conn.execute(text("""
                    INSERT INTO scraper_runs 
                    (scraper_id, start_time, end_time, status, records_scraped, data_quality_score)
                    VALUES (:id, :start, :end, :status, :records, :quality)
                """), {
                    'id': scraper.id,
                    'start': datetime.utcnow() - timedelta(minutes=5),
                    'end': datetime.utcnow(),
                    'status': 'completed',
                    'records': 100,
                    'quality': 0.95
                })
                conn.commit()
        
        # 6. Check monitoring
        monitor = mcp_system.monitor
        await monitor.update_metrics()
        health_report = monitor.get_health_report()
        
        assert 'metrics' in health_report
        assert health_report['metrics']['total_scrapers'] == len(mcp_system.registry.scrapers)
    
    @pytest.mark.asyncio
    async def test_data_quality_pipeline(self, test_database):
        """Test data quality agent integration"""
        agent = MCPDataQualityAgent(database_url=TEST_DATABASE_URL)
        
        # Run quality check
        report = await agent.run_comprehensive_check()
        
        assert report is not None
        assert hasattr(report, 'overall_quality_score')
        assert hasattr(report, 'table_reports')
        assert hasattr(report, 'recommendations')
        
        # Check if issues are logged
        with create_engine(TEST_DATABASE_URL).connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM data_quality_issues")).scalar()
            # May or may not have issues, just check query works
            assert result >= 0
    
    @pytest.mark.asyncio
    async def test_ml_optimization_integration(self, mcp_system):
        """Test ML optimization engine integration"""
        # Create ML engine
        ml_engine = MLOptimizationEngine()
        
        # Generate sample historical data
        historical_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'scraper_id': ['scraper_1'] * 100,
            'response_time': np.random.gamma(2, 2, 100),
            'status': np.random.choice(['success', 'failed'], 100, p=[0.9, 0.1]),
            'platform': 'legistar',
            'category': 'municipal_council'
        })
        
        # Train models
        await ml_engine.train_models(historical_data)
        assert ml_engine.models_trained
        
        # Test predictions
        sample_scraper = list(mcp_system.registry.scrapers.values())[0]
        
        # Predict response time
        response_prediction = await ml_engine.predict_response_time(
            sample_scraper, datetime.utcnow()
        )
        assert 'predicted_time' in response_prediction
        assert response_prediction['predicted_time'] > 0
        
        # Predict failure probability
        failure_prediction = await ml_engine.predict_failure_probability(
            sample_scraper, datetime.utcnow()
        )
        assert 'failure_probability' in failure_prediction
        assert 0 <= failure_prediction['failure_probability'] <= 1
        
        # Get optimization insights
        insights = ml_engine.get_optimization_insights()
        assert 'feature_importance' in insights
        assert 'recommendations' in insights
    
    @pytest.mark.asyncio
    async def test_analytics_engine_integration(self, test_database, redis_client):
        """Test real-time analytics engine"""
        analytics = RealTimeAnalyticsEngine(
            database_url=TEST_DATABASE_URL,
            redis_url=TEST_REDIS_URL
        )
        await analytics.initialize()
        
        # Generate executive dashboard
        dashboard = await analytics.get_executive_dashboard()
        
        assert dashboard is not None
        assert 'system_health' in dashboard.data
        assert 'scraper_performance' in dashboard.data
        assert 'data_quality' in dashboard.data
        assert 'insights' in dashboard.data
        assert len(dashboard.visualizations) > 0
        
        # Test custom report generation
        custom_config = {
            'name': 'Test Report',
            'start_time': datetime.utcnow() - timedelta(days=7),
            'end_time': datetime.utcnow(),
            'metrics': ['error_rate', 'response_time'],
            'grouping': 'day'
        }
        
        custom_report = await analytics.generate_custom_report(custom_config)
        assert custom_report is not None
        assert custom_report.query_id.startswith('custom_')
    
    @pytest.mark.asyncio
    async def test_health_remediation_integration(self, test_database, redis_client):
        """Test automated health remediation"""
        remediation = AutomatedHealthRemediation()
        
        # Simulate health issues
        with create_engine(TEST_DATABASE_URL).connect() as conn:
            # Create failing scraper
            conn.execute(text("""
                INSERT INTO scrapers (id, name, status, failure_count, last_run)
                VALUES ('failing_scraper', 'Test Failing Scraper', 'active', 10, NOW())
            """))
            conn.commit()
        
        # Check for issues
        issues = await remediation._check_scraper_failures({
            'threshold': 5,
            'severity_mapping': {5: 'medium', 10: 'high'}
        })
        
        assert len(issues) > 0
        assert issues[0].type == HealthIssueType.SCRAPER_FAILURE
        assert issues[0].severity == 'high'
        
        # Test remediation handling
        await remediation._handle_health_issue(issues[0])
        
        # Check if remediation was attempted
        assert len(remediation.active_remediations) > 0 or len(remediation.remediation_history) > 0
    
    @pytest.mark.asyncio
    async def test_alerting_system_integration(self, redis_client):
        """Test comprehensive alerting system"""
        alerting = ComprehensiveAlertingSystem()
        await alerting.initialize()
        
        # Test metric checking
        test_metrics = {
            'error_rate': 0.25,  # Above warning threshold
            'disk_usage_percent': 85,  # Warning level
            'failed_scrapers': 15,  # Above error threshold
            'response_time': 12.5
        }
        
        await alerting.check_metrics(test_metrics)
        
        # Check if alerts were generated
        alert_stats = await alerting.get_alert_stats()
        assert alert_stats['total_alerts'] > 0
        
        # Test anomaly detection
        for i in range(100):
            await alerting.anomaly_detector.detect_anomalies(
                'response_time',
                np.random.normal(5, 1),
                {'iteration': i}
            )
        
        # Inject anomaly
        anomaly = await alerting.anomaly_detector.detect_anomalies(
            'response_time',
            50,  # Extreme value
            {'test': 'anomaly'}
        )
        
        assert anomaly is not None
        assert anomaly['type'] is not None
        assert anomaly['score'] > 0
    
    @pytest.mark.asyncio
    async def test_scraper_testing_framework(self, mcp_system):
        """Test scraper testing framework"""
        test_framework = ScraperTestFramework(mcp_system.registry)
        await test_framework.initialize()
        
        # Run tests on sample scrapers
        scrapers = list(mcp_system.registry.scrapers.values())[:3]
        
        for scraper in scrapers:
            # Mock scraper URL for testing
            scraper.url = 'https://httpbin.org/status/200'
            
            # Test connectivity
            connectivity_result = await test_framework._test_connectivity(scraper)
            assert connectivity_result.test_type == TestType.CONNECTIVITY
            
            # Test performance
            performance_result = await test_framework._test_performance(scraper)
            assert performance_result.test_type == TestType.PERFORMANCE
            assert 'avg_response_time' in performance_result.details
        
        await test_framework.close()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, mcp_system, test_database, redis_client):
        """Test complete end-to-end workflow"""
        # 1. Initialize all components
        ml_engine = MLOptimizationEngine()
        analytics = RealTimeAnalyticsEngine(TEST_DATABASE_URL, TEST_REDIS_URL)
        await analytics.initialize()
        remediation = AutomatedHealthRemediation()
        alerting = ComprehensiveAlertingSystem()
        await alerting.initialize()
        
        # 2. Discover and schedule scrapers
        await mcp_system.registry.discover_scrapers()
        sample_scrapers = list(mcp_system.registry.scrapers.values())[:10]
        
        # 3. Optimize scheduling with ML
        if sample_scrapers:
            allocations = await ml_engine.optimize_resource_allocation(
                sample_scrapers, max_concurrent=5
            )
            assert len(allocations) <= 5
        
        # 4. Execute scrapers (simulated)
        for scraper in sample_scrapers[:3]:
            # Simulate successful run
            run_data = {
                'scraper_id': scraper.id,
                'scraper_name': scraper.name,
                'jurisdiction': scraper.jurisdiction,
                'category': scraper.category.value,
                'timestamp': datetime.utcnow().isoformat(),
                'data': [{'test': 'data'}]
            }
            
            await redis_client.lpush(
                'scraper:ingestion:queue',
                json.dumps(run_data)
            )
        
        # 5. Check metrics and alerts
        current_metrics = {
            'error_rate': 0.05,
            'active_scrapers': len(sample_scrapers),
            'queue_size': await redis_client.llen('scraper:queue'),
            'avg_response_time': 5.2
        }
        
        await alerting.check_metrics(current_metrics)
        
        # 6. Generate executive report
        executive_dashboard = await analytics.get_executive_dashboard()
        assert executive_dashboard is not None
        assert 'insights' in executive_dashboard.data
        
        # 7. Verify system health
        health_report = mcp_system.monitor.get_health_report()
        assert health_report['metrics']['total_scrapers'] > 0
        
        print("\n✅ End-to-end workflow completed successfully!")
        print(f"   - Discovered scrapers: {len(mcp_system.registry.scrapers)}")
        print(f"   - Processed scrapers: {len(sample_scrapers)}")
        print(f"   - System health: {health_report}")
        print(f"   - Alerts generated: {await alerting.get_alert_stats()}")


class TestAPIIntegration:
    """Test API endpoints integration"""
    
    @pytest.mark.asyncio
    async def test_mcp_health_endpoint(self):
        """Test MCP health endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE_URL}/api/v1/mcp/health") as response:
                assert response.status == 200
                data = await response.json()
                assert data['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_scraper_management_endpoints(self):
        """Test scraper management API endpoints"""
        async with aiohttp.ClientSession() as session:
            # List scrapers
            async with session.get(f"{API_BASE_URL}/api/v1/scrapers") as response:
                assert response.status == 200
                data = await response.json()
                assert 'total' in data
                assert 'scrapers' in data
            
            # Get statistics
            async with session.get(f"{API_BASE_URL}/api/v1/scrapers/stats") as response:
                assert response.status == 200
                stats = await response.json()
                assert 'total_scrapers' in stats
                assert 'by_category' in stats
                assert 'by_platform' in stats
            
            # Get monitoring dashboard
            async with session.get(f"{API_BASE_URL}/api/v1/scrapers/monitoring/dashboard") as response:
                assert response.status == 200
                dashboard = await response.json()
                assert 'overview' in dashboard
    
    @pytest.mark.asyncio
    async def test_quality_check_endpoint(self):
        """Test data quality check endpoint"""
        async with aiohttp.ClientSession() as session:
            # Trigger quality check
            async with session.post(f"{API_BASE_URL}/api/v1/mcp/quality/check") as response:
                assert response.status == 200
                data = await response.json()
                assert 'job_id' in data
                
                job_id = data['job_id']
                
                # Wait a bit
                await asyncio.sleep(2)
                
                # Check job status
                async with session.get(f"{API_BASE_URL}/api/v1/mcp/jobs/{job_id}") as response:
                    assert response.status == 200
                    job_data = await response.json()
                    assert 'status' in job_data
                    assert job_data['status'] in ['running', 'completed', 'failed']


class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    async def test_high_volume_ingestion(self, redis_client):
        """Test high volume data ingestion"""
        pipeline = DataIngestionPipeline(TEST_DATABASE_URL)
        await pipeline.initialize()
        
        # Generate large batch of data
        start_time = time.time()
        batch_size = 1000
        
        for i in range(batch_size):
            msg = {
                'scraper_id': f'perf_test_{i % 10}',
                'scraper_name': f'Performance Test {i % 10}',
                'jurisdiction': {'type': 'municipal', 'code': 'test', 'name': 'Test City'},
                'category': 'municipal_council',
                'timestamp': datetime.utcnow().isoformat(),
                'data': [
                    {
                        'name': f'Representative {j}',
                        'role': 'Councillor',
                        'email': f'rep{j}@test.com',
                        'jurisdiction': 'Test City'
                    }
                    for j in range(10)
                ]
            }
            
            await redis_client.lpush('scraper:ingestion:queue', json.dumps(msg))
        
        # Process queue
        processed = 0
        timeout = 30  # 30 seconds timeout
        
        while processed < batch_size and (time.time() - start_time) < timeout:
            msg = await redis_client.brpop('scraper:ingestion:queue', timeout=1)
            if msg:
                await pipeline._process_ingestion(json.loads(msg[1]))
                processed += 1
        
        elapsed = time.time() - start_time
        rate = processed / elapsed
        
        print(f"\n📊 Performance Test Results:")
        print(f"   - Processed: {processed} messages")
        print(f"   - Time: {elapsed:.2f} seconds")
        print(f"   - Rate: {rate:.2f} messages/second")
        
        assert processed == batch_size, f"Should process all {batch_size} messages"
        assert rate > 10, "Should process at least 10 messages per second"
    
    @pytest.mark.asyncio
    async def test_concurrent_scraper_execution(self, mcp_system):
        """Test concurrent scraper execution limits"""
        # Schedule many scrapers
        all_scrapers = list(mcp_system.registry.scrapers.values())
        await mcp_system.orchestrator.schedule_scrapers([s.id for s in all_scrapers[:50]])
        
        # Check that concurrent limit is respected
        running = mcp_system.orchestrator.get_running_scrapers()
        assert len(running) <= mcp_system.orchestrator.max_concurrent
        
        print(f"\n🚀 Concurrency Test:")
        print(f"   - Scheduled: 50 scrapers")
        print(f"   - Running: {len(running)}")
        print(f"   - Max concurrent: {mcp_system.orchestrator.max_concurrent}")


# Utility functions for testing
async def wait_for_condition(condition_func, timeout=10, interval=0.1):
    """Wait for a condition to be true"""
    start = time.time()
    while time.time() - start < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    return False


def generate_test_data(num_records: int) -> List[Dict[str, Any]]:
    """Generate test data for scrapers"""
    return [
        {
            'id': f'test_{i}',
            'name': f'Test Record {i}',
            'created_at': datetime.utcnow().isoformat(),
            'value': np.random.random()
        }
        for i in range(num_records)
    ]


if __name__ == "__main__":
    # Run all integration tests
    pytest.main([__file__, '-v', '--tb=short'])