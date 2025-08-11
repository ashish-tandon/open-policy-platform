#!/usr/bin/env python3
"""
Continuous Data Collection System
Monitors scrapers, tracks failures, and ensures continuous data loading
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import requests
from config.database import db_config

class ContinuousDataCollectionSystem:
    def __init__(self):
        self.setup_logging()
        self.failure_tracker = {}
        self.success_tracker = {}
        self.data_collection_log = []
        self.system_health_log = []
        self.database_status_log = []
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/continuous_collection_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Continuous Data Collection System Started")
    
    def run_continuous_collection(self, interval=1800):  # 30 minutes
        """Run continuous data collection with monitoring"""
        self.logger.info(f"🔄 Starting continuous collection with {interval}s intervals")
        
        while True:
            try:
                cycle_start = time.time()
                self.logger.info("=" * 60)
                self.logger.info(f"🕐 Starting collection cycle at {datetime.now()}")
                
                # Run comprehensive scraper test
                self.run_scraper_collection()
                
                # Check system health
                self.check_system_health()
                
                # Check database status
                self.check_database_status()
                
                # Generate comprehensive report
                self.generate_collection_report()
                
                # Calculate next run time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, interval - cycle_duration)
                
                self.logger.info(f"✅ Cycle completed in {cycle_duration:.1f}s")
                self.logger.info(f"⏰ Next cycle in {sleep_time:.1f}s")
                
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Continuous collection stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in collection cycle: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def run_scraper_collection(self):
        """Run scraper collection with detailed tracking"""
        self.logger.info("🔍 Running scraper collection...")
        
        try:
            # Run the enhanced scraper framework
            result = subprocess.run([
                sys.executable, "enhanced_scraper_framework.py"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                self.log_success("Scraper collection completed successfully")
                self.parse_scraper_output(result.stdout)
            else:
                self.log_failure(f"Scraper collection failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.log_failure("Scraper collection timed out after 30 minutes")
        except Exception as e:
            self.log_failure(f"Scraper collection error: {e}")
    
    def parse_scraper_output(self, output):
        """Parse scraper output for detailed tracking"""
        lines = output.split('\n')
        
        for line in lines:
            if '✅' in line and 'Success' in line:
                self.log_success(f"Scraper success: {line}")
            elif '❌' in line and 'Failed' in line:
                self.log_failure(f"Scraper failure: {line}")
            elif '📊' in line and 'records' in line:
                self.log_data_collection(line)
    
    def check_system_health(self):
        """Check system health and performance"""
        self.logger.info("💻 Checking system health...")
        
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        self.system_health_log.append(health_data)
        
        # Log warnings for high resource usage
        if health_data['cpu_percent'] > 80:
            self.logger.warning(f"⚠️ High CPU usage: {health_data['cpu_percent']}%")
        if health_data['memory_percent'] > 80:
            self.logger.warning(f"⚠️ High memory usage: {health_data['memory_percent']}%")
        if health_data['disk_usage'] > 90:
            self.logger.warning(f"⚠️ High disk usage: {health_data['disk_usage']}%")
        
        self.logger.info(f"✅ System health: CPU {health_data['cpu_percent']}%, Memory {health_data['memory_percent']}%, Disk {health_data['disk_usage']}%")
    
    def check_database_status(self):
        """Check database status and record counts"""
        self.logger.info("🗄️ Checking database status...")
        
        try:
            # Check database connection and record counts
            result = subprocess.run([
                "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
                "-c", "SELECT 'core_politician' as table_name, COUNT(*) as record_count FROM core_politician UNION ALL SELECT 'bills_bill', COUNT(*) FROM bills_bill UNION ALL SELECT 'hansards_statement', COUNT(*) FROM hansards_statement ORDER BY record_count DESC;",
                "-t", "-A"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                db_status = {}
                
                for line in lines:
                    if '|' in line:
                        table, count = line.split('|')
                        db_status[table.strip()] = int(count.strip())
                
                db_data = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'healthy',
                    'tables': db_status
                }
                
                self.database_status_log.append(db_data)
                
                self.logger.info(f"✅ Database status: {db_status}")
                
                # Check for significant data growth
                if len(self.database_status_log) > 1:
                    prev_status = self.database_status_log[-2]['tables']
                    for table, count in db_status.items():
                        if table in prev_status:
                            growth = count - prev_status[table]
                            if growth > 0:
                                self.logger.info(f"📈 {table}: +{growth} records")
                
            else:
                self.log_failure(f"Database check failed: {result.stderr}")
                
        except Exception as e:
            self.log_failure(f"Database status check error: {e}")
    
    def check_api_health(self):
        """Check API server health"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                self.log_success("API server is healthy")
                return True
            else:
                self.log_failure(f"API server returned status {response.status_code}")
                return False
        except Exception as e:
            self.log_failure(f"API health check failed: {e}")
            return False
    
    def log_success(self, message):
        """Log successful operations"""
        timestamp = datetime.now().isoformat()
        success_entry = {
            'timestamp': timestamp,
            'type': 'success',
            'message': message
        }
        self.success_tracker[timestamp] = success_entry
        self.logger.info(f"✅ {message}")
    
    def log_failure(self, message):
        """Log failures with details"""
        timestamp = datetime.now().isoformat()
        failure_entry = {
            'timestamp': timestamp,
            'type': 'failure',
            'message': message,
            'system_info': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
        self.failure_tracker[timestamp] = failure_entry
        self.logger.error(f"❌ {message}")
    
    def log_data_collection(self, message):
        """Log data collection events"""
        timestamp = datetime.now().isoformat()
        collection_entry = {
            'timestamp': timestamp,
            'message': message
        }
        self.data_collection_log.append(collection_entry)
        self.logger.info(f"📊 {message}")
    
    def generate_collection_report(self):
        """Generate comprehensive collection report"""
        self.logger.info("📊 Generating collection report...")
        
        # Calculate statistics
        total_successes = len(self.success_tracker)
        total_failures = len(self.failure_tracker)
        success_rate = (total_successes / (total_successes + total_failures)) * 100 if (total_successes + total_failures) > 0 else 0
        
        # Recent activity (last 10 entries)
        recent_successes = list(self.success_tracker.values())[-10:]
        recent_failures = list(self.failure_tracker.values())[-10:]
        
        # System health summary
        if self.system_health_log:
            latest_health = self.system_health_log[-1]
        else:
            latest_health = {}
        
        # Database growth
        db_growth = {}
        if len(self.database_status_log) >= 2:
            current = self.database_status_log[-1]['tables']
            previous = self.database_status_log[-2]['tables']
            for table in current:
                if table in previous:
                    db_growth[table] = current[table] - previous[table]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_successes': total_successes,
                'total_failures': total_failures,
                'success_rate': round(success_rate, 2),
                'collection_cycles': len(self.data_collection_log)
            },
            'recent_activity': {
                'successes': recent_successes,
                'failures': recent_failures
            },
            'system_health': latest_health,
            'database_growth': db_growth,
            'data_collection_events': self.data_collection_log[-20:]  # Last 20 events
        }
        
        # Save report
        report_file = f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📄 Collection report saved to {report_file}")
        self.logger.info(f"📈 Success rate: {success_rate:.1f}% ({total_successes} successes, {total_failures} failures)")
        
        # Log database growth
        if db_growth:
            for table, growth in db_growth.items():
                if growth > 0:
                    self.logger.info(f"📊 {table}: +{growth} records")
    
    def get_failure_analysis(self):
        """Analyze failure patterns"""
        if not self.failure_tracker:
            return "No failures recorded"
        
        failure_types = {}
        for failure in self.failure_tracker.values():
            message = failure['message']
            if 'classification' in message:
                failure_types['classification_error'] = failure_types.get('classification_error', 0) + 1
            elif 'SSL' in message:
                failure_types['ssl_error'] = failure_types.get('ssl_error', 0) + 1
            elif 'timeout' in message:
                failure_types['timeout'] = failure_types.get('timeout', 0) + 1
            else:
                failure_types['other'] = failure_types.get('other', 0) + 1
        
        return failure_types

def main():
    """Main function to run continuous data collection"""
    print("🚀 CONTINUOUS DATA COLLECTION SYSTEM")
    print("=" * 50)
    
    # Create and run the collection system
    collection_system = ContinuousDataCollectionSystem()
    
    try:
        collection_system.run_continuous_collection()
    except KeyboardInterrupt:
        print("\n🛑 Continuous collection stopped by user")
        print("📊 Final statistics:")
        print(f"  Successes: {len(collection_system.success_tracker)}")
        print(f"  Failures: {len(collection_system.failure_tracker)}")
        print(f"  Data collection events: {len(collection_system.data_collection_log)}")
        
        # Generate final report
        collection_system.generate_collection_report()

if __name__ == "__main__":
    main()
