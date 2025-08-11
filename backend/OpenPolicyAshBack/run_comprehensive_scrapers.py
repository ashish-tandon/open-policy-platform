#!/usr/bin/env python3
"""
Comprehensive Scraper Execution and Database Loading Script
Runs all scrapers in background and monitors progress
"""

import subprocess
import time
import json
import os
import sys
from datetime import datetime
import psutil
from config.database import db_config

def run_scraper_process(category, max_records=500, log_file=None):
    """Run a scraper process for a specific category"""
    if log_file is None:
        log_file = f"{category}_full_run.log"
    
    cmd = [
        sys.executable, "scraper_testing_framework.py",
        "--max-sample-records", str(max_records),
        "--category", category,
        "--verbose"
    ]
    
    print(f"🚀 Starting {category} scrapers with {max_records} records per scraper...")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    return process, log_file

def check_database_status():
    """Check current database status"""
    try:
        result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", "SELECT 'core_politician' as table_name, COUNT(*) as record_count FROM core_politician UNION ALL SELECT 'bills_bill', COUNT(*) FROM bills_bill UNION ALL SELECT 'hansards_statement', COUNT(*) FROM hansards_statement ORDER BY record_count DESC;",
            "-t", "-A"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            status = {}
            for line in lines:
                if '|' in line:
                    table, count = line.split('|')
                    status[table.strip()] = int(count.strip())
            return status
    except Exception as e:
        print(f"⚠️ Database check error: {e}")
    
    return {}

def monitor_processes(processes):
    """Monitor running scraper processes"""
    print("\n📊 MONITORING SCRAPER PROCESSES")
    print("=" * 50)
    
    while any(p.poll() is None for p, _ in processes):
        print(f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check process status
        for i, (process, log_file) in enumerate(processes):
            if process.poll() is None:
                print(f"✅ Process {i+1} ({log_file}): RUNNING (PID: {process.pid})")
            else:
                print(f"✅ Process {i+1} ({log_file}): COMPLETED (Exit: {process.returncode})")
        
        # Check database status
        db_status = check_database_status()
        if db_status:
            print(f"📊 Database Status:")
            for table, count in db_status.items():
                print(f"   {table}: {count:,} records")
        
        # Check system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        print(f"💻 System: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
        
        time.sleep(30)  # Check every 30 seconds
    
    print("\n🎉 ALL SCRAPER PROCESSES COMPLETED!")

def main():
    """Main execution function"""
    print("🚀 COMPREHENSIVE SCRAPER EXECUTION AND DATABASE LOADING")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Categories to run
    categories = [
        ("provincial", 500),
        ("municipal", 500),
        ("parliamentary", 200),
        ("civic", 100),
        ("update", 50)
    ]
    
    # Start all scraper processes
    processes = []
    for category, max_records in categories:
        process, log_file = run_scraper_process(category, max_records)
        processes.append((process, log_file))
        time.sleep(5)  # Stagger starts
    
    print(f"\n✅ Started {len(processes)} scraper processes")
    
    # Monitor processes
    monitor_processes(processes)
    
    # Final status check
    print("\n📊 FINAL DATABASE STATUS")
    print("=" * 30)
    db_status = check_database_status()
    if db_status:
        for table, count in db_status.items():
            print(f"✅ {table}: {count:,} records")
    
    print(f"\n🎉 Comprehensive scraper execution completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
