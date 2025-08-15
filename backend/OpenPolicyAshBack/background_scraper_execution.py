#!/usr/bin/env python3
"""
Background Scraper Execution System
==================================

This script runs the 35 working scrapers in the background continuously,
with monitoring, error handling, and schedule-based execution.

Following AI Agent Guidance System and TDD Process.
"""

import os
import sys
import time
import json
import logging
import threading
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import signal
import psutil

# Add scrapers path to Python path
scrapers_root = Path(__file__).resolve().parents[2] / "scrapers" / "scrapers-ca"
sys.path.insert(0, str(scrapers_root))

# Resolve writable data directory for outputs
SCRAPERS_DATA_DIR = Path(os.getenv("SCRAPERS_DATA_DIR", "/scrapers/data")).resolve()

class ScraperStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"
    SCHEDULED = "scheduled"

class ScraperSchedule(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CONTINUOUS = "continuous"
    ONE_TIME = "one_time"

@dataclass
class ScraperExecution:
    name: str
    path: str
    schedule: ScraperSchedule
    status: ScraperStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    records_collected: int
    error_count: int
    last_error: Optional[str]
    process_id: Optional[int]
    memory_usage: float
    cpu_usage: float

class BackgroundScraperExecution:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.logs_dir = Path(__file__).parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        SCRAPERS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Working scrapers from testing framework
        self.working_scrapers = {
            # Federal scrapers
            "scrapers/openparliament": ScraperSchedule.DAILY,
            
            # Provincial scrapers
            "scrapers/scrapers-ca/ca": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_on": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_bc": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_ab": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_sk": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_mb": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_ns": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_nb": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_pe": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_nl": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_nt": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_nu": ScraperSchedule.WEEKLY,
            "scrapers/scrapers-ca/ca_yt": ScraperSchedule.WEEKLY,
            
            # Municipal scrapers
            "scrapers/scrapers-ca/ca_on_toronto": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_ab_calgary": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_ab_edmonton": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_on_mississauga": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_on_windsor": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_bc_surrey": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_on_hamilton": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_qc_quebec": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_bc_victoria": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_bc_abbotsford": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_sk_regina": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_mb_winnipeg": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_bc_richmond": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_qc_gatineau": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_ab_lethbridge": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_sk_saskatoon": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_nb_moncton": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_pe_charlottetown": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_bc_burnaby": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_ns_halifax": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_nb_fredericton": ScraperSchedule.MONTHLY,
            "scrapers/scrapers-ca/ca_nl_st_john_s": ScraperSchedule.MONTHLY,
        }
        
        self.executions: Dict[str, ScraperExecution] = {}
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.stop_flag = False
        
        # Initialize executions
        self.initialize_executions()

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / f"background_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Background Scraper Execution System Started")

    def initialize_executions(self):
        """Initialize scraper executions."""
        for scraper_path, schedule_type in self.working_scrapers.items():
            scraper_name = scraper_path.split('/')[-1].replace('_', ' ').title()
            
            execution = ScraperExecution(
                name=scraper_name,
                path=scraper_path,
                schedule=schedule_type,
                status=ScraperStatus.SCHEDULED,
                start_time=None,
                end_time=None,
                records_collected=0,
                error_count=0,
                last_error=None,
                process_id=None,
                memory_usage=0.0,
                cpu_usage=0.0
            )
            
            self.executions[scraper_path] = execution
        
        self.logger.info(f"✅ Initialized {len(self.executions)} scraper executions")

    def run_scraper(self, scraper_path: str) -> bool:
        """Run a single scraper."""
        try:
            execution = self.executions[scraper_path]
            execution.status = ScraperStatus.RUNNING
            execution.start_time = datetime.now()
            execution.error_count = 0
            execution.last_error = None
            
            self.logger.info(f"🔄 Starting scraper: {execution.name}")
            
            # Get the full path to the scraper
            scraper_full_path = self.base_path / scraper_path
            
            # Create a unique run directory under SCRAPERS_DATA_DIR
            run_dir = SCRAPERS_DATA_DIR / execution.name.replace(' ', '_').lower() / datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Run the scraper using the testing framework
            cmd = [
                sys.executable, 
                "scraper_testing_framework.py",
                "--scraper-path", str(scraper_full_path),
                "--max-records", "10",
                "--timeout", "300",
                "--output-dir", str(run_dir)
            ]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            execution.process_id = process.pid
            self.running_processes[scraper_path] = process
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)
                
                if process.returncode == 0:
                    execution.status = ScraperStatus.COMPLETED
                    execution.records_collected = 10  # Estimated
                    self.logger.info(f"✅ Scraper completed: {execution.name} (output: {run_dir})")
                    return True
                else:
                    execution.status = ScraperStatus.FAILED
                    execution.error_count += 1
                    execution.last_error = stderr
                    self.logger.error(f"❌ Scraper failed: {execution.name} - {stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                process.kill()
                execution.status = ScraperStatus.FAILED
                execution.error_count += 1
                execution.last_error = "Timeout after 5 minutes"
                self.logger.error(f"⏰ Scraper timeout: {execution.name}")
                return False
                
        except Exception as e:
            execution.status = ScraperStatus.FAILED
            execution.error_count += 1
            execution.last_error = str(e)
            self.logger.error(f"❌ Error running scraper {scraper_path}: {e}")
            return False
        finally:
            execution.end_time = datetime.now()
            if scraper_path in self.running_processes:
                del self.running_processes[scraper_path]

    def run_daily_scrapers(self):
        """Run daily scrapers."""
        self.logger.info("📅 Running daily scrapers...")
        for scraper_path, execution in self.executions.items():
            if execution.schedule == ScraperSchedule.DAILY:
                self.run_scraper(scraper_path)

    def run_weekly_scrapers(self):
        """Run weekly scrapers."""
        self.logger.info("📅 Running weekly scrapers...")
        for scraper_path, execution in self.executions.items():
            if execution.schedule == ScraperSchedule.WEEKLY:
                self.run_scraper(scraper_path)

    def run_monthly_scrapers(self):
        """Run monthly scrapers."""
        self.logger.info("📅 Running monthly scrapers...")
        for scraper_path, execution in self.executions.items():
            if execution.schedule == ScraperSchedule.MONTHLY:
                self.run_scraper(scraper_path)

    def run_continuous_scrapers(self):
        """Run continuous scrapers."""
        self.logger.info("🔄 Running continuous scrapers...")
        for scraper_path, execution in self.executions.items():
            if execution.schedule == ScraperSchedule.CONTINUOUS:
                self.run_scraper(scraper_path)

    def monitor_system_resources(self):
        """Monitor system resources."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.logger.info(f"💻 System Resources - CPU: {cpu_percent}%, Memory: {memory.percent}%")
            
            # Update resource usage for running processes
            for scraper_path, process in self.running_processes.items():
                try:
                    process_info = psutil.Process(process.pid)
                    execution = self.executions[scraper_path]
                    execution.memory_usage = process_info.memory_info().rss / 1024 / 1024  # MB
                    execution.cpu_usage = process_info.cpu_percent()
                except psutil.NoSuchProcess:
                    pass
                    
        except Exception as e:
            self.logger.error(f"❌ Error monitoring system resources: {e}")

    def save_status_report(self):
        """Save status report."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_scrapers': len(self.executions),
                'running': len([e for e in self.executions.values() if e.status == ScraperStatus.RUNNING]),
                'completed': len([e for e in self.executions.values() if e.status == ScraperStatus.COMPLETED]),
                'failed': len([e for e in self.executions.values() if e.status == ScraperStatus.FAILED]),
                'scheduled': len([e for e in self.executions.values() if e.status == ScraperStatus.SCHEDULED]),
                'executions': [asdict(execution) for execution in self.executions.values()]
            }
            
            # Convert enum values to strings
            for execution_dict in report['executions']:
                execution_dict['status'] = execution_dict['status'].value
                execution_dict['schedule'] = execution_dict['schedule'].value
                if execution_dict['start_time']:
                    execution_dict['start_time'] = execution_dict['start_time'].isoformat()
                if execution_dict['end_time']:
                    execution_dict['end_time'] = execution_dict['end_time'].isoformat()
            
            report_file = self.logs_dir / f"status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"📊 Status report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving status report: {e}")

    def setup_schedules(self):
        """Setup scheduled jobs."""
        # Daily scrapers (run at 6 AM)
        schedule.every().day.at("06:00").do(self.run_daily_scrapers)
        
        # Weekly scrapers (run every Monday at 7 AM)
        schedule.every().monday.at("07:00").do(self.run_weekly_scrapers)
        
        # Monthly scrapers (run on 1st of month at 8 AM)
        # Note: schedule library doesn't support .month, so we'll use a workaround
        schedule.every().day.at("08:00").do(self.check_and_run_monthly_scrapers)
        
        # Continuous scrapers (run every 2 hours)
        schedule.every(2).hours.do(self.run_continuous_scrapers)
        
        # System monitoring (every 5 minutes)
        schedule.every(5).minutes.do(self.monitor_system_resources)
        
        # Status reports (every hour)
        schedule.every().hour.do(self.save_status_report)
        
        self.logger.info("📅 Schedules configured")

    def check_and_run_monthly_scrapers(self):
        """Check if it's the 1st of the month and run monthly scrapers."""
        today = datetime.now()
        if today.day == 1:  # Only run on the 1st of the month
            self.logger.info("📅 First day of month detected, running monthly scrapers...")
            self.run_monthly_scrapers()
        else:
            self.logger.debug(f"📅 Not first day of month (day {today.day}), skipping monthly scrapers")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("🛑 Received shutdown signal, stopping all processes...")
        self.stop_flag = True
        
        # Stop all running processes
        for scraper_path, process in self.running_processes.items():
            try:
                process.terminate()
                self.logger.info(f"🛑 Terminated process for {scraper_path}")
            except Exception as e:
                self.logger.error(f"❌ Error terminating process for {scraper_path}: {e}")

    def run(self):
        """Run the background execution system."""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Setup schedules
            self.setup_schedules()
            
            # Initial run of all scrapers
            self.logger.info("🚀 Starting initial run of all scrapers...")
            self.run_daily_scrapers()
            self.run_weekly_scrapers()
            self.run_monthly_scrapers()
            self.run_continuous_scrapers()
            
            # Main loop
            self.logger.info("🔄 Starting main execution loop...")
            while not self.stop_flag:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"❌ Error in main execution loop: {e}")
        finally:
            self.logger.info("🛑 Background execution system stopped")
            self.save_status_report()

if __name__ == "__main__":
    execution_system = BackgroundScraperExecution()
    execution_system.run()
