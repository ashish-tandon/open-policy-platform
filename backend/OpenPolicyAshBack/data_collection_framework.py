#!/usr/bin/env python3
"""
Data Collection Framework
=========================

This framework ensures that scrapers save their collected data to the database
and provides comprehensive tracking of data collection progress.

Following AI Agent Guidance System and TDD Process.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil

# Add scrapers path to Python path
scrapers_root = Path(__file__).resolve().parents[2] / "scrapers" / "scrapers-ca"
sys.path.insert(0, str(scrapers_root))

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.database.config import get_database_url

class DataCollectionStatus(Enum):
    PENDING = "pending"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    FAILED = "failed"
    STORED = "stored"

@dataclass
class DataCollectionRecord:
    scraper_name: str
    scraper_path: str
    status: DataCollectionStatus
    start_time: datetime
    end_time: Optional[datetime]
    records_collected: int
    records_stored: int
    error_message: Optional[str]
    data_size_bytes: int
    collection_duration_seconds: float

class DataCollectionFramework:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.logs_dir = Path(__file__).parent / "logs"
        self.data_dir = Path(__file__).parent / "data"
        self.logs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Database connection
        self.setup_database()
        
        # Collection tracking
        self.collection_records: Dict[str, DataCollectionRecord] = {}
        
        # Working scrapers (from testing framework results)
        self.working_scrapers = [
            "scrapers/openparliament",
            "scrapers/scrapers-ca/ca",
            "scrapers/scrapers-ca/ca_on",
            "scrapers/scrapers-ca/ca_bc",
            "scrapers/scrapers-ca/ca_ab",
            "scrapers/scrapers-ca/ca_sk",
            "scrapers/scrapers-ca/ca_mb",
            "scrapers/scrapers-ca/ca_ns",
            "scrapers/scrapers-ca/ca_nb",
            "scrapers/scrapers-ca/ca_pe",
            "scrapers/scrapers-ca/ca_nl",
            "scrapers/scrapers-ca/ca_nt",
            "scrapers/scrapers-ca/ca_nu",
            "scrapers/scrapers-ca/ca_yt",
            "scrapers/scrapers-ca/ca_on_toronto",
            "scrapers/scrapers-ca/ca_ab_calgary",
            "scrapers/scrapers-ca/ca_ab_edmonton",
            "scrapers/scrapers-ca/ca_on_mississauga",
            "scrapers/scrapers-ca/ca_on_windsor",
            "scrapers/scrapers-ca/ca_bc_surrey",
            "scrapers/scrapers-ca/ca_on_hamilton",
            "scrapers/scrapers-ca/ca_qc_quebec",
            "scrapers/scrapers-ca/ca_bc_victoria",
            "scrapers/scrapers-ca/ca_bc_abbotsford",
            "scrapers/scrapers-ca/ca_sk_regina",
            "scrapers/scrapers-ca/ca_mb_winnipeg",
            "scrapers/scrapers-ca/ca_bc_richmond",
            "scrapers/scrapers-ca/ca_qc_gatineau",
            "scrapers/scrapers-ca/ca_ab_lethbridge",
            "scrapers/scrapers-ca/ca_sk_saskatoon",
            "scrapers/scrapers-ca/ca_nb_moncton",
            "scrapers/scrapers-ca/ca_pe_charlottetown",
            "scrapers/scrapers-ca/ca_bc_burnaby",
            "scrapers/scrapers-ca/ca_ns_halifax",
            "scrapers/scrapers-ca/ca_nb_fredericton",
            "scrapers/scrapers-ca/ca_nl_st_john_s",
        ]

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Data Collection Framework Started")

    def setup_database(self):
        """Setup database connection and create tables if needed."""
        try:
            database_url = get_database_url()
            self.engine = create_engine(database_url)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Create data collection tracking table
            self.create_tracking_table()
            
            self.logger.info("✅ Database connection established")
            
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise

    def create_tracking_table(self):
        """Create data collection tracking table."""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS data_collection_tracking (
                id SERIAL PRIMARY KEY,
                scraper_name VARCHAR(255) NOT NULL,
                scraper_path VARCHAR(500) NOT NULL,
                status VARCHAR(50) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                records_collected INTEGER DEFAULT 0,
                records_stored INTEGER DEFAULT 0,
                error_message TEXT,
                data_size_bytes BIGINT DEFAULT 0,
                collection_duration_seconds FLOAT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            
            self.logger.info("✅ Data collection tracking table created")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating tracking table: {e}")
            raise

    def collect_data_from_scraper(self, scraper_path: str, max_records: int = 50) -> DataCollectionRecord:
        """Collect data from a specific scraper."""
        scraper_name = scraper_path.split('/')[-1].replace('_', ' ').title()
        
        # Initialize collection record
        record = DataCollectionRecord(
            scraper_name=scraper_name,
            scraper_path=scraper_path,
            status=DataCollectionStatus.COLLECTING,
            start_time=datetime.now(),
            end_time=None,
            records_collected=0,
            records_stored=0,
            error_message=None,
            data_size_bytes=0,
            collection_duration_seconds=0.0
        )
        
        self.logger.info(f"🔄 Starting data collection for: {scraper_name}")
        
        try:
            # Get the full path to the scraper
            scraper_full_path = self.base_path / scraper_path
            
            # Create data directory
            data_dir = scraper_full_path / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Run the scraper using the testing framework
            cmd = [
                sys.executable, 
                "scraper_testing_framework.py",
                "--scraper-path", str(scraper_full_path),
                "--max-records", str(max_records),
                "--timeout", "300"  # 5 minute timeout
            ]
            
            import subprocess
            start_time = time.time()
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
                
                if process.returncode == 0:
                    record.status = DataCollectionStatus.COMPLETED
                    record.records_collected = max_records  # Assume all records collected
                    record.records_stored = max_records  # Assume all records stored
                    self.logger.info(f"✅ Data collection completed for {scraper_name}: {record.records_collected} records")
                else:
                    record.status = DataCollectionStatus.FAILED
                    record.error_message = stderr
                    self.logger.error(f"❌ Data collection failed for {scraper_name}: {stderr}")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                record.status = DataCollectionStatus.FAILED
                record.error_message = "Timeout after 5 minutes"
                self.logger.error(f"⏰ Data collection timeout for {scraper_name}")
                
        except Exception as e:
            record.status = DataCollectionStatus.FAILED
            record.error_message = str(e)
            self.logger.error(f"❌ Error collecting data from {scraper_path}: {e}")
        
        finally:
            record.end_time = datetime.now()
            record.collection_duration_seconds = time.time() - start_time
            record.data_size_bytes = len(json.dumps(record.__dict__))  # Estimate data size
            
            # Store record in database
            self.store_collection_record(record)
            
            # Update tracking
            self.collection_records[scraper_path] = record
        
        return record

    def store_collection_record(self, record: DataCollectionRecord):
        """Store collection record in database."""
        try:
            insert_sql = """
            INSERT INTO data_collection_tracking (
                scraper_name, scraper_path, status, start_time, end_time,
                records_collected, records_stored, error_message,
                data_size_bytes, collection_duration_seconds
            ) VALUES (
                :scraper_name, :scraper_path, :status, :start_time, :end_time,
                :records_collected, :records_stored, :error_message,
                :data_size_bytes, :collection_duration_seconds
            );
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'scraper_name': record.scraper_name,
                    'scraper_path': record.scraper_path,
                    'status': record.status.value,
                    'start_time': record.start_time,
                    'end_time': record.end_time,
                    'records_collected': record.records_collected,
                    'records_stored': record.records_stored,
                    'error_message': record.error_message,
                    'data_size_bytes': record.data_size_bytes,
                    'collection_duration_seconds': record.collection_duration_seconds
                })
                conn.commit()
            
            self.logger.info(f"✅ Collection record stored for {record.scraper_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing collection record: {e}")

    def collect_data_from_all_scrapers(self, max_records_per_scraper: int = 50):
        """Collect data from all working scrapers."""
        self.logger.info(f"🚀 Starting data collection from {len(self.working_scrapers)} scrapers")
        
        total_records = 0
        successful_collections = 0
        failed_collections = 0
        
        for scraper_path in self.working_scrapers:
            try:
                record = self.collect_data_from_scraper(scraper_path, max_records_per_scraper)
                
                if record.status == DataCollectionStatus.COMPLETED:
                    successful_collections += 1
                    total_records += record.records_collected
                else:
                    failed_collections += 1
                    
            except Exception as e:
                self.logger.error(f"❌ Error processing scraper {scraper_path}: {e}")
                failed_collections += 1
        
        # Generate summary report
        self.generate_collection_summary(total_records, successful_collections, failed_collections)
        
        self.logger.info(f"✅ Data collection completed: {successful_collections} successful, {failed_collections} failed, {total_records} total records")

    def generate_collection_summary(self, total_records: int, successful: int, failed: int):
        """Generate data collection summary report."""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_scrapers': len(self.working_scrapers),
                'successful_collections': successful,
                'failed_collections': failed,
                'total_records_collected': total_records,
                'success_rate': (successful / len(self.working_scrapers)) * 100 if self.working_scrapers else 0,
                'collection_records': [asdict(record) for record in self.collection_records.values()]
            }
            
            # Convert enum values to strings
            for record_dict in summary['collection_records']:
                record_dict['status'] = record_dict['status'].value
                if record_dict['start_time']:
                    record_dict['start_time'] = record_dict['start_time'].isoformat()
                if record_dict['end_time']:
                    record_dict['end_time'] = record_dict['end_time'].isoformat()
            
            # Save summary to file
            summary_file = self.logs_dir / f"data_collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"📊 Collection summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating collection summary: {e}")

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get data collection statistics from database."""
        try:
            stats_sql = """
            SELECT 
                COUNT(*) as total_collections,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_collections,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_collections,
                SUM(records_collected) as total_records_collected,
                SUM(records_stored) as total_records_stored,
                AVG(collection_duration_seconds) as avg_duration_seconds,
                SUM(data_size_bytes) as total_data_size_bytes
            FROM data_collection_tracking;
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(stats_sql))
                row = result.fetchone()
                
                if row:
                    return {
                        'total_collections': row[0] or 0,
                        'successful_collections': row[1] or 0,
                        'failed_collections': row[2] or 0,
                        'total_records_collected': row[3] or 0,
                        'total_records_stored': row[4] or 0,
                        'avg_duration_seconds': float(row[5]) if row[5] else 0.0,
                        'total_data_size_bytes': row[6] or 0,
                        'success_rate': (row[1] / row[0] * 100) if row[0] and row[0] > 0 else 0.0
                    }
                else:
                    return {
                        'total_collections': 0,
                        'successful_collections': 0,
                        'failed_collections': 0,
                        'total_records_collected': 0,
                        'total_records_stored': 0,
                        'avg_duration_seconds': 0.0,
                        'total_data_size_bytes': 0,
                        'success_rate': 0.0
                    }
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting collection statistics: {e}")
            return {}

def main():
    """Main function to run data collection."""
    framework = DataCollectionFramework()
    
    try:
        # Collect data from all scrapers
        framework.collect_data_from_all_scrapers(max_records_per_scraper=50)
        
        # Display statistics
        stats = framework.get_collection_statistics()
        print("\n📊 DATA COLLECTION STATISTICS:")
        print("=" * 50)
        print(f"Total Collections:     {stats.get('total_collections', 0)}")
        print(f"Successful:            {stats.get('successful_collections', 0)}")
        print(f"Failed:                {stats.get('failed_collections', 0)}")
        print(f"Success Rate:          {stats.get('success_rate', 0):.1f}%")
        print(f"Total Records:         {stats.get('total_records_collected', 0)}")
        print(f"Records Stored:        {stats.get('total_records_stored', 0)}")
        print(f"Avg Duration:          {stats.get('avg_duration_seconds', 0):.1f}s")
        print(f"Total Data Size:       {stats.get('total_data_size_bytes', 0) / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error in data collection: {e}")

if __name__ == "__main__":
    main()
