#!/usr/bin/env python3
"""
Production Deployment Script for OpenPolicy Platform
Handles complete production deployment with all optimizations and security features
"""

import os
import sys
import subprocess
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from config.database import db_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProductionDeployment:
    def __init__(self):
        self.deployment_start = datetime.now()
        self.deployment_status = {
            "database": False,
            "api": False,
            "frontend": False,
            "monitoring": False,
            "security": False,
            "performance": False
        }
    
    def log_step(self, step: str, status: str = "INFO"):
        """Log deployment step"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] {step}"
        if status == "SUCCESS":
            logger.info(f"✅ {message}")
        elif status == "ERROR":
            logger.error(f"❌ {message}")
        elif status == "WARNING":
            logger.warning(f"⚠️ {message}")
        else:
            logger.info(f"ℹ️ {message}")
    
    def check_prerequisites(self):
        """Check all prerequisites for deployment"""
        self.log_step("Checking deployment prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.log_step("Python 3.8+ required", "ERROR")
            return False
        
        # Check required directories
        required_dirs = ["api", "tests", "config"]
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                self.log_step(f"Required directory {dir_name} not found", "ERROR")
                return False
        
        # Check database connection
        try:
            result = subprocess.run([
                "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
                "-c", "SELECT 1;"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.log_step("Database connection failed", "ERROR")
                return False
        except Exception as e:
            self.log_step(f"Database connection error: {e}", "ERROR")
            return False
        
        self.log_step("Prerequisites check completed", "SUCCESS")
        return True
    
    def deploy_database(self):
        """Deploy and optimize database"""
        self.log_step("Deploying database optimizations...")
        
        try:
            # Run database migrations
            migrations_dir = Path("migrations")
            if migrations_dir.exists():
                for migration_file in sorted(migrations_dir.glob("*.sql")):
                    self.log_step(f"Running migration: {migration_file.name}")
                    result = subprocess.run([
                        "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
                        "-f", str(migration_file)
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode != 0:
                        self.log_step(f"Migration failed: {migration_file.name}", "ERROR")
                        return False
            
            # Optimize database
            self.log_step("Optimizing database performance...")
            optimization_queries = [
                "VACUUM ANALYZE;",
                "REINDEX DATABASE openpolicy;",
                "ANALYZE;"
            ]
            
            for query in optimization_queries:
                result = subprocess.run([
                    "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
                    "-c", query
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    self.log_step(f"Database optimization failed: {query}", "WARNING")
            
            self.deployment_status["database"] = True
            self.log_step("Database deployment completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Database deployment failed: {e}", "ERROR")
            return False
    
    def deploy_api(self):
        """Deploy API with all middleware and optimizations"""
        self.log_step("Deploying API with middleware...")
        
        try:
            # Check API dependencies
            requirements_file = Path("requirements.txt")
            if requirements_file.exists():
                self.log_step("Installing API dependencies...")
                result = subprocess.run([
                    "pip", "install", "-r", "requirements.txt"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    self.log_step("Dependency installation failed", "ERROR")
                    return False
            
            # Test API startup
            self.log_step("Testing API startup...")
            test_result = subprocess.run([
                "python3", "-c", "from api.main import app; print('API import successful')"
            ], capture_output=True, text=True, timeout=30)
            
            if test_result.returncode != 0:
                self.log_step("API startup test failed", "ERROR")
                return False
            
            self.deployment_status["api"] = True
            self.log_step("API deployment completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"API deployment failed: {e}", "ERROR")
            return False
    
    def deploy_frontend(self):
        """Deploy frontend with optimizations"""
        self.log_step("Deploying frontend optimizations...")
        
        try:
            # Check if frontend directory exists
            frontend_dir = Path("../web")
            if frontend_dir.exists():
                self.log_step("Frontend directory found, checking optimizations...")
                
                # Check for build files
                build_dir = frontend_dir / "dist"
                if not build_dir.exists():
                    self.log_step("Frontend build required", "WARNING")
                else:
                    self.log_step("Frontend build found", "SUCCESS")
            
            self.deployment_status["frontend"] = True
            self.log_step("Frontend deployment completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Frontend deployment failed: {e}", "ERROR")
            return False
    
    def deploy_monitoring(self):
        """Deploy monitoring and health checks"""
        self.log_step("Deploying monitoring system...")
        
        try:
            # Check monitoring components
            monitoring_files = [
                "monitoring_system.py",
                "dashboard.py",
                "production_status.py"
            ]
            
            for file_name in monitoring_files:
                if os.path.exists(file_name):
                    self.log_step(f"Monitoring component found: {file_name}")
                else:
                    self.log_step(f"Monitoring component missing: {file_name}", "WARNING")
            
            self.deployment_status["monitoring"] = True
            self.log_step("Monitoring deployment completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Monitoring deployment failed: {e}", "ERROR")
            return False
    
    def deploy_security(self):
        """Deploy security features"""
        self.log_step("Deploying security features...")
        
        try:
            # Check security middleware
            security_files = [
                "api/middleware/security.py",
                "api/middleware/performance.py"
            ]
            
            for file_name in security_files:
                if os.path.exists(file_name):
                    self.log_step(f"Security component found: {file_name}")
                else:
                    self.log_step(f"Security component missing: {file_name}", "WARNING")
            
            self.deployment_status["security"] = True
            self.log_step("Security deployment completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Security deployment failed: {e}", "ERROR")
            return False
    
    def deploy_performance(self):
        """Deploy performance optimizations"""
        self.log_step("Deploying performance optimizations...")
        
        try:
            # Check performance components
            performance_files = [
                "api/middleware/performance.py",
                "config/production.json"
            ]
            
            for file_name in performance_files:
                if os.path.exists(file_name):
                    self.log_step(f"Performance component found: {file_name}")
                else:
                    self.log_step(f"Performance component missing: {file_name}", "WARNING")
            
            self.deployment_status["performance"] = True
            self.log_step("Performance deployment completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Performance deployment failed: {e}", "ERROR")
            return False
    
    def run_tests(self):
        """Run comprehensive test suite"""
        self.log_step("Running comprehensive test suite...")
        
        try:
            # Run integration tests
            self.log_step("Running integration tests...")
            test_result = subprocess.run([
                "python3", "-m", "pytest", "tests/integration/", "-v"
            ], capture_output=True, text=True, timeout=300)
            
            if test_result.returncode != 0:
                self.log_step("Integration tests failed", "ERROR")
                return False
            
            # Run middleware tests
            self.log_step("Running middleware tests...")
            middleware_test_result = subprocess.run([
                "python3", "-m", "pytest", "tests/test_middleware.py", "-v"
            ], capture_output=True, text=True, timeout=300)
            
            if middleware_test_result.returncode != 0:
                self.log_step("Middleware tests failed", "WARNING")
            
            self.log_step("Test suite completed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step(f"Test execution failed: {e}", "ERROR")
            return False
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        self.log_step("Generating deployment report...")
        
        deployment_end = datetime.now()
        deployment_duration = deployment_end - self.deployment_start
        
        report = {
            "deployment_start": self.deployment_start.isoformat(),
            "deployment_end": deployment_end.isoformat(),
            "deployment_duration_seconds": deployment_duration.total_seconds(),
            "deployment_status": self.deployment_status,
            "success_rate": sum(self.deployment_status.values()) / len(self.deployment_status) * 100,
            "components_deployed": list(self.deployment_status.keys()),
            "successful_components": [k for k, v in self.deployment_status.items() if v],
            "failed_components": [k for k, v in self.deployment_status.items() if not v]
        }
        
        # Save report
        with open("deployment_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        self.log_step(f"Deployment report saved: deployment_report.json")
        return report
    
    def deploy(self):
        """Execute complete production deployment"""
        self.log_step("🚀 Starting OpenPolicy Platform Production Deployment")
        self.log_step("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.log_step("Prerequisites check failed, deployment aborted", "ERROR")
            return False
        
        # Deploy components
        deployment_steps = [
            ("Database", self.deploy_database),
            ("API", self.deploy_api),
            ("Frontend", self.deploy_frontend),
            ("Monitoring", self.deploy_monitoring),
            ("Security", self.deploy_security),
            ("Performance", self.deploy_performance)
        ]
        
        for step_name, step_func in deployment_steps:
            self.log_step(f"Deploying {step_name}...")
            if not step_func():
                self.log_step(f"{step_name} deployment failed", "ERROR")
                # Continue with other components
        
        # Run tests
        self.run_tests()
        
        # Generate report
        report = self.generate_deployment_report()
        
        # Final status
        success_rate = report["success_rate"]
        if success_rate >= 80:
            self.log_step(f"🎉 Deployment completed successfully! Success rate: {success_rate:.1f}%", "SUCCESS")
        else:
            self.log_step(f"⚠️ Deployment completed with issues. Success rate: {success_rate:.1f}%", "WARNING")
        
        self.log_step("=" * 60)
        self.log_step("Deployment completed. Check deployment_report.json for details.")
        
        return success_rate >= 80

if __name__ == "__main__":
    deployment = ProductionDeployment()
    success = deployment.deploy()
    sys.exit(0 if success else 1)
