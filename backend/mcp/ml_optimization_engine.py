"""
ML-Powered Scraper Optimization Engine - 40by6
Uses machine learning to optimize scraper performance and predict failures
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

from .scraper_management_system import ScraperMetadata, ScraperStatus

logger = logging.getLogger(__name__)


@dataclass
class ScraperFeatures:
    """Features for ML models"""
    hour_of_day: int
    day_of_week: int
    month: int
    response_time_avg: float
    response_time_std: float
    failure_rate_7d: float
    failure_rate_30d: float
    data_volume_avg: float
    platform_encoding: int
    category_encoding: int
    last_success_hours_ago: float
    consecutive_failures: int
    total_runs: int
    data_quality_score: float
    rate_limit_hits: int
    timeout_rate: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.hour_of_day,
            self.day_of_week,
            self.month,
            self.response_time_avg,
            self.response_time_std,
            self.failure_rate_7d,
            self.failure_rate_30d,
            self.data_volume_avg,
            self.platform_encoding,
            self.category_encoding,
            self.last_success_hours_ago,
            self.consecutive_failures,
            self.total_runs,
            self.data_quality_score,
            self.rate_limit_hits,
            self.timeout_rate
        ])


class MLOptimizationEngine:
    """ML-powered optimization for scraper management"""
    
    def __init__(self):
        self.response_time_model = None
        self.failure_prediction_model = None
        self.anomaly_detector = None
        self.optimal_time_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.models_trained = False
        
        # Platform and category encodings
        self.platform_encoding = {
            "legistar": 0, "civic_plus": 1, "granicus": 2,
            "openparliament": 3, "custom": 4, "represent_api": 5
        }
        
        self.category_encoding = {
            "federal_parliament": 0, "provincial_legislature": 1,
            "municipal_council": 2, "civic_platform": 3,
            "federal_elections": 4, "provincial_elections": 5
        }
    
    async def train_models(self, historical_data: pd.DataFrame):
        """Train all ML models with historical scraper data"""
        logger.info("Starting ML model training...")
        
        # Prepare features
        X, y_response_time, y_failure = self._prepare_training_data(historical_data)
        
        # Split data
        X_train, X_test, y_resp_train, y_resp_test, y_fail_train, y_fail_test = train_test_split(
            X, y_response_time, y_failure, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train response time prediction model
        logger.info("Training response time prediction model...")
        self.response_time_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.response_time_model.fit(X_train_scaled, y_resp_train)
        
        # Evaluate
        resp_pred = self.response_time_model.predict(X_test_scaled)
        resp_mae = mean_absolute_error(y_resp_test, resp_pred)
        logger.info(f"Response time MAE: {resp_mae:.2f} seconds")
        
        # Train failure prediction model
        logger.info("Training failure prediction model...")
        self.failure_prediction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.failure_prediction_model.fit(X_train_scaled, y_fail_train)
        
        # Evaluate
        fail_pred = self.failure_prediction_model.predict(X_test_scaled)
        fail_acc = accuracy_score(y_fail_test, fail_pred)
        logger.info(f"Failure prediction accuracy: {fail_acc:.2%}")
        
        # Train anomaly detector
        logger.info("Training anomaly detector...")
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_detector.fit(X_train_scaled[y_fail_train == 0])  # Train on successful runs
        
        # Train optimal time predictor (neural network)
        logger.info("Training optimal time predictor...")
        self.optimal_time_model = self._build_optimal_time_model(X_train_scaled.shape[1])
        
        # Prepare time-based labels (hour of day with best success rate)
        y_optimal_time = self._calculate_optimal_times(historical_data)
        y_time_train = y_optimal_time[:len(X_train)]
        y_time_test = y_optimal_time[len(X_train):len(X_train)+len(X_test)]
        
        self.optimal_time_model.fit(
            X_train_scaled,
            y_time_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        self.models_trained = True
        logger.info("ML model training completed!")
        
        # Save models
        await self._save_models()
    
    def _prepare_training_data(self, historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        features = []
        response_times = []
        failures = []
        
        for _, row in historical_data.iterrows():
            # Extract features
            dt = pd.to_datetime(row['timestamp'])
            
            feature = ScraperFeatures(
                hour_of_day=dt.hour,
                day_of_week=dt.dayofweek,
                month=dt.month,
                response_time_avg=row.get('response_time_avg', 5.0),
                response_time_std=row.get('response_time_std', 1.0),
                failure_rate_7d=row.get('failure_rate_7d', 0.0),
                failure_rate_30d=row.get('failure_rate_30d', 0.0),
                data_volume_avg=row.get('data_volume_avg', 100),
                platform_encoding=self.platform_encoding.get(row.get('platform', 'custom'), 4),
                category_encoding=self.category_encoding.get(row.get('category', 'custom_scraper'), 0),
                last_success_hours_ago=row.get('last_success_hours_ago', 0),
                consecutive_failures=row.get('consecutive_failures', 0),
                total_runs=row.get('total_runs', 100),
                data_quality_score=row.get('data_quality_score', 0.8),
                rate_limit_hits=row.get('rate_limit_hits', 0),
                timeout_rate=row.get('timeout_rate', 0.05)
            )
            
            features.append(feature.to_array())
            response_times.append(row.get('response_time', 5.0))
            failures.append(1 if row.get('status') == 'failed' else 0)
        
        return np.array(features), np.array(response_times), np.array(failures)
    
    def _build_optimal_time_model(self, input_dim: int) -> keras.Model:
        """Build neural network for optimal time prediction"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(24, activation='softmax')  # 24 hours
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _calculate_optimal_times(self, historical_data: pd.DataFrame) -> np.ndarray:
        """Calculate optimal execution times based on success rates"""
        # Group by scraper and hour, calculate success rates
        historical_data['hour'] = pd.to_datetime(historical_data['timestamp']).dt.hour
        historical_data['success'] = (historical_data['status'] != 'failed').astype(int)
        
        hourly_success = historical_data.groupby(['scraper_id', 'hour'])['success'].mean()
        
        # Find best hour for each scraper
        optimal_hours = []
        for _, row in historical_data.iterrows():
            scraper_id = row['scraper_id']
            if scraper_id in hourly_success:
                best_hour = hourly_success[scraper_id].idxmax()
                optimal_hours.append(best_hour)
            else:
                optimal_hours.append(2)  # Default to 2 AM
        
        return np.array(optimal_hours)
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        if self.response_time_model:
            feature_names = [
                'hour_of_day', 'day_of_week', 'month', 'response_time_avg',
                'response_time_std', 'failure_rate_7d', 'failure_rate_30d',
                'data_volume_avg', 'platform', 'category', 'last_success_hours',
                'consecutive_failures', 'total_runs', 'data_quality_score',
                'rate_limit_hits', 'timeout_rate'
            ]
            
            importances = self.response_time_model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
    
    async def predict_response_time(self, scraper: ScraperMetadata, execution_time: datetime) -> float:
        """Predict response time for a scraper at given time"""
        if not self.models_trained:
            return 5.0  # Default
        
        features = self._extract_features(scraper, execution_time)
        features_scaled = self.scaler.transform([features.to_array()])
        
        predicted_time = self.response_time_model.predict(features_scaled)[0]
        
        # Add confidence interval
        predictions = []
        for estimator in self.response_time_model.estimators_:
            predictions.append(estimator.predict(features_scaled)[0])
        
        confidence_interval = np.percentile(predictions, [25, 75])
        
        return {
            'predicted_time': predicted_time,
            'confidence_interval': confidence_interval,
            'confidence': 1 - (confidence_interval[1] - confidence_interval[0]) / predicted_time
        }
    
    async def predict_failure_probability(self, scraper: ScraperMetadata, execution_time: datetime) -> float:
        """Predict probability of failure for a scraper"""
        if not self.models_trained:
            return 0.1  # Default
        
        features = self._extract_features(scraper, execution_time)
        features_scaled = self.scaler.transform([features.to_array()])
        
        failure_prob = self.failure_prediction_model.predict_proba(features_scaled)[0][1]
        
        # Check for anomalies
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        return {
            'failure_probability': failure_prob,
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'risk_level': self._calculate_risk_level(failure_prob, is_anomaly)
        }
    
    async def recommend_optimal_schedule(self, scraper: ScraperMetadata) -> Dict[str, Any]:
        """Recommend optimal execution schedule for a scraper"""
        if not self.models_trained:
            return {'recommended_hour': 2, 'confidence': 0.5}
        
        # Predict success probability for each hour
        current_time = datetime.utcnow()
        hourly_predictions = {}
        
        for hour in range(24):
            test_time = current_time.replace(hour=hour, minute=0, second=0)
            features = self._extract_features(scraper, test_time)
            features_scaled = self.scaler.transform([features.to_array()])
            
            # Get failure probability
            failure_prob = self.failure_prediction_model.predict_proba(features_scaled)[0][1]
            
            # Get response time prediction
            response_time = self.response_time_model.predict(features_scaled)[0]
            
            # Calculate combined score (lower is better)
            score = failure_prob * 10 + response_time / 60  # Weight failure more heavily
            
            hourly_predictions[hour] = {
                'success_probability': 1 - failure_prob,
                'expected_response_time': response_time,
                'score': score
            }
        
        # Find best hours
        sorted_hours = sorted(hourly_predictions.items(), key=lambda x: x[1]['score'])
        best_hour = sorted_hours[0][0]
        
        # Get neural network prediction
        features = self._extract_features(scraper, current_time)
        features_scaled = self.scaler.transform([features.to_array()])
        nn_prediction = self.optimal_time_model.predict(features_scaled)[0]
        nn_best_hour = np.argmax(nn_prediction)
        
        # Combine predictions
        if abs(best_hour - nn_best_hour) <= 2:
            recommended_hour = best_hour
            confidence = 0.9
        else:
            # Average the recommendations
            recommended_hour = (best_hour + nn_best_hour) // 2
            confidence = 0.7
        
        return {
            'recommended_hour': recommended_hour,
            'confidence': confidence,
            'expected_success_rate': hourly_predictions[recommended_hour]['success_probability'],
            'expected_response_time': hourly_predictions[recommended_hour]['expected_response_time'],
            'alternative_hours': [h[0] for h in sorted_hours[:3]],
            'hourly_analysis': hourly_predictions
        }
    
    async def optimize_resource_allocation(self, scrapers: List[ScraperMetadata], max_concurrent: int) -> List[Dict[str, Any]]:
        """Optimize resource allocation for multiple scrapers"""
        logger.info(f"Optimizing resource allocation for {len(scrapers)} scrapers")
        
        current_time = datetime.utcnow()
        scraper_scores = []
        
        for scraper in scrapers:
            # Calculate priority score
            features = self._extract_features(scraper, current_time)
            features_scaled = self.scaler.transform([features.to_array()])
            
            # Predict metrics
            failure_prob = self.failure_prediction_model.predict_proba(features_scaled)[0][1]
            response_time = self.response_time_model.predict(features_scaled)[0]
            
            # Calculate priority (higher is more important)
            days_since_last_run = (current_time - scraper.last_run).days if scraper.last_run else 30
            
            priority_score = (
                (1 - failure_prob) * 10 +  # Success likelihood
                min(days_since_last_run / 7, 5) +  # Staleness
                (10 - scraper.priority) +  # Base priority
                (1 / (response_time + 1)) * 5  # Favor faster scrapers
            )
            
            scraper_scores.append({
                'scraper': scraper,
                'priority_score': priority_score,
                'failure_probability': failure_prob,
                'expected_response_time': response_time,
                'recommended_resources': self._calculate_resource_needs(scraper, response_time)
            })
        
        # Sort by priority
        scraper_scores.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Allocate resources
        allocated = []
        total_resources = 0
        
        for item in scraper_scores:
            if len(allocated) >= max_concurrent:
                break
            
            resources_needed = item['recommended_resources']['total']
            if total_resources + resources_needed <= 100:  # 100% resource cap
                allocated.append(item)
                total_resources += resources_needed
        
        return allocated
    
    def _extract_features(self, scraper: ScraperMetadata, execution_time: datetime) -> ScraperFeatures:
        """Extract features for ML prediction"""
        # Calculate historical metrics (would come from database in production)
        last_success = scraper.last_success or execution_time - timedelta(days=7)
        hours_since_success = (execution_time - last_success).total_seconds() / 3600
        
        return ScraperFeatures(
            hour_of_day=execution_time.hour,
            day_of_week=execution_time.weekday(),
            month=execution_time.month,
            response_time_avg=5.0,  # Would calculate from history
            response_time_std=1.0,
            failure_rate_7d=min(scraper.failure_count / 10, 1.0),
            failure_rate_30d=min(scraper.failure_count / 50, 1.0),
            data_volume_avg=100,
            platform_encoding=self.platform_encoding.get(scraper.platform.value, 4),
            category_encoding=self.category_encoding.get(scraper.category.value, 0),
            last_success_hours_ago=hours_since_success,
            consecutive_failures=scraper.failure_count,
            total_runs=100,  # Would get from history
            data_quality_score=0.8,  # Would calculate from history
            rate_limit_hits=0,
            timeout_rate=0.05
        )
    
    def _calculate_risk_level(self, failure_prob: float, is_anomaly: bool) -> str:
        """Calculate risk level based on predictions"""
        if is_anomaly and failure_prob > 0.7:
            return "critical"
        elif failure_prob > 0.7 or is_anomaly:
            return "high"
        elif failure_prob > 0.3:
            return "medium"
        else:
            return "low"
    
    def _calculate_resource_needs(self, scraper: ScraperMetadata, expected_time: float) -> Dict[str, float]:
        """Calculate resource requirements for a scraper"""
        # Base estimates
        base_cpu = 10  # 10% CPU
        base_memory = 256  # 256 MB
        
        # Adjust based on platform
        platform_multipliers = {
            "legistar": 1.5,
            "granicus": 1.3,
            "civic_plus": 1.2,
            "custom": 2.0,
            "openparliament": 1.0,
            "represent_api": 0.8
        }
        
        multiplier = platform_multipliers.get(scraper.platform.value, 1.0)
        
        # Adjust based on expected execution time
        time_factor = min(expected_time / 30, 2.0)  # Cap at 2x for long-running scrapers
        
        return {
            'cpu_percent': base_cpu * multiplier * time_factor,
            'memory_mb': base_memory * multiplier,
            'total': (base_cpu * multiplier * time_factor) / 2 + (base_memory * multiplier) / 1000 * 50
        }
    
    async def detect_anomalies(self, recent_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in recent scraper runs"""
        if not self.models_trained or not recent_runs:
            return []
        
        anomalies = []
        
        for run in recent_runs:
            # Extract features from run data
            features = ScraperFeatures(
                hour_of_day=pd.to_datetime(run['timestamp']).hour,
                day_of_week=pd.to_datetime(run['timestamp']).dayofweek,
                month=pd.to_datetime(run['timestamp']).month,
                response_time_avg=run.get('response_time', 5.0),
                response_time_std=0.5,
                failure_rate_7d=run.get('recent_failure_rate', 0.1),
                failure_rate_30d=run.get('monthly_failure_rate', 0.1),
                data_volume_avg=run.get('records_scraped', 100),
                platform_encoding=self.platform_encoding.get(run.get('platform', 'custom'), 4),
                category_encoding=self.category_encoding.get(run.get('category', 'custom_scraper'), 0),
                last_success_hours_ago=run.get('hours_since_last_success', 0),
                consecutive_failures=run.get('consecutive_failures', 0),
                total_runs=run.get('total_runs', 100),
                data_quality_score=run.get('quality_score', 0.8),
                rate_limit_hits=run.get('rate_limit_hits', 0),
                timeout_rate=run.get('timeout_rate', 0.05)
            )
            
            features_scaled = self.scaler.transform([features.to_array()])
            
            # Check if anomaly
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            
            if is_anomaly:
                # Determine anomaly type
                anomaly_type = self._classify_anomaly(run, features)
                
                anomalies.append({
                    'scraper_id': run['scraper_id'],
                    'scraper_name': run.get('scraper_name', 'Unknown'),
                    'timestamp': run['timestamp'],
                    'anomaly_score': float(anomaly_score),
                    'anomaly_type': anomaly_type,
                    'details': self._get_anomaly_details(run, features, anomaly_type)
                })
        
        return sorted(anomalies, key=lambda x: x['anomaly_score'])
    
    def _classify_anomaly(self, run: Dict[str, Any], features: ScraperFeatures) -> str:
        """Classify the type of anomaly"""
        if run.get('response_time', 0) > features.response_time_avg * 3:
            return "performance_degradation"
        elif run.get('records_scraped', 0) < features.data_volume_avg * 0.1:
            return "low_data_volume"
        elif features.consecutive_failures > 5:
            return "repeated_failures"
        elif features.rate_limit_hits > 10:
            return "rate_limit_issues"
        elif run.get('quality_score', 1) < 0.5:
            return "data_quality_issues"
        else:
            return "unknown_anomaly"
    
    def _get_anomaly_details(self, run: Dict[str, Any], features: ScraperFeatures, anomaly_type: str) -> Dict[str, Any]:
        """Get detailed information about the anomaly"""
        details = {
            'anomaly_type': anomaly_type,
            'severity': 'high' if features.consecutive_failures > 3 else 'medium',
            'metrics': {
                'response_time': run.get('response_time', 0),
                'records_scraped': run.get('records_scraped', 0),
                'quality_score': run.get('quality_score', 0),
                'consecutive_failures': features.consecutive_failures
            }
        }
        
        # Add recommendations
        if anomaly_type == "performance_degradation":
            details['recommendations'] = [
                "Check target website for changes",
                "Optimize scraper selectors",
                "Consider increasing timeout"
            ]
        elif anomaly_type == "low_data_volume":
            details['recommendations'] = [
                "Verify data source availability",
                "Check for website structure changes",
                "Review scraper logic"
            ]
        elif anomaly_type == "repeated_failures":
            details['recommendations'] = [
                "Investigate root cause of failures",
                "Check authentication if required",
                "Consider disabling scraper temporarily"
            ]
        
        return details
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from the optimization engine"""
        if not self.models_trained:
            return {"status": "models_not_trained"}
        
        return {
            'feature_importance': self.feature_importance,
            'top_factors': list(self.feature_importance.keys())[:5],
            'model_performance': {
                'response_time_mae': getattr(self, 'response_time_mae', 'N/A'),
                'failure_prediction_accuracy': getattr(self, 'failure_accuracy', 'N/A')
            },
            'recommendations': [
                "Schedule scrapers during off-peak hours (2-5 AM)",
                "Monitor scrapers with high failure rates",
                "Optimize scrapers with response times > 10s",
                "Implement retry logic for transient failures"
            ]
        }
    
    async def _save_models(self):
        """Save trained models to disk"""
        import os
        
        model_dir = "models/mcp"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save scikit-learn models
        joblib.dump(self.response_time_model, f"{model_dir}/response_time_model.pkl")
        joblib.dump(self.failure_prediction_model, f"{model_dir}/failure_prediction_model.pkl")
        joblib.dump(self.anomaly_detector, f"{model_dir}/anomaly_detector.pkl")
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        
        # Save neural network
        self.optimal_time_model.save(f"{model_dir}/optimal_time_model.h5")
        
        # Save metadata
        metadata = {
            'feature_importance': self.feature_importance,
            'platform_encoding': self.platform_encoding,
            'category_encoding': self.category_encoding,
            'trained_at': datetime.utcnow().isoformat()
        }
        
        import json
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {model_dir}")
    
    async def load_models(self):
        """Load pre-trained models"""
        import os
        
        model_dir = "models/mcp"
        
        try:
            self.response_time_model = joblib.load(f"{model_dir}/response_time_model.pkl")
            self.failure_prediction_model = joblib.load(f"{model_dir}/failure_prediction_model.pkl")
            self.anomaly_detector = joblib.load(f"{model_dir}/anomaly_detector.pkl")
            self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
            self.optimal_time_model = keras.models.load_model(f"{model_dir}/optimal_time_model.h5")
            
            # Load metadata
            import json
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
                self.feature_importance = metadata['feature_importance']
            
            self.models_trained = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_trained = False


# Automated optimization runner
class AutomatedOptimizer:
    """Automated system for continuous optimization"""
    
    def __init__(self, optimization_engine: MLOptimizationEngine):
        self.engine = optimization_engine
        self.optimization_history = []
    
    async def run_continuous_optimization(self, scraper_registry, orchestrator):
        """Run continuous optimization loop"""
        while True:
            try:
                logger.info("Running optimization cycle...")
                
                # Get all active scrapers
                active_scrapers = [
                    s for s in scraper_registry.scrapers.values()
                    if s.status == ScraperStatus.ACTIVE
                ]
                
                # Optimize schedules
                for scraper in active_scrapers:
                    recommendation = await self.engine.recommend_optimal_schedule(scraper)
                    
                    # Update schedule if confidence is high
                    if recommendation['confidence'] > 0.8:
                        new_hour = recommendation['recommended_hour']
                        scraper.schedule = f"0 {new_hour} * * *"
                        logger.info(f"Updated schedule for {scraper.name} to {new_hour}:00")
                
                # Optimize resource allocation
                allocation = await self.engine.optimize_resource_allocation(
                    active_scrapers,
                    orchestrator.max_concurrent
                )
                
                # Update orchestrator priorities
                for item in allocation:
                    scraper = item['scraper']
                    scraper.priority = min(10, int(item['priority_score']))
                
                # Detect anomalies in recent runs
                recent_runs = await self._get_recent_runs()
                anomalies = await self.engine.detect_anomalies(recent_runs)
                
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} anomalies")
                    await self._handle_anomalies(anomalies)
                
                # Log optimization results
                self.optimization_history.append({
                    'timestamp': datetime.utcnow(),
                    'scrapers_optimized': len(active_scrapers),
                    'anomalies_detected': len(anomalies),
                    'allocations_changed': len(allocation)
                })
                
                # Sleep for 1 hour before next optimization
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in optimization cycle: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _get_recent_runs(self) -> List[Dict[str, Any]]:
        """Get recent scraper runs from database"""
        # This would fetch from database in production
        return []
    
    async def _handle_anomalies(self, anomalies: List[Dict[str, Any]]):
        """Handle detected anomalies"""
        for anomaly in anomalies:
            if anomaly['anomaly_type'] == 'repeated_failures':
                # Disable scraper temporarily
                logger.warning(f"Disabling scraper {anomaly['scraper_name']} due to repeated failures")
                # Would update scraper status in registry
            
            elif anomaly['anomaly_type'] == 'performance_degradation':
                # Increase timeout
                logger.info(f"Increasing timeout for {anomaly['scraper_name']}")
                # Would update scraper configuration


# Example usage
async def setup_ml_optimization():
    """Setup and run ML optimization"""
    engine = MLOptimizationEngine()
    
    # Load historical data (would come from database)
    historical_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='H'),
        'scraper_id': ['scraper_' + str(i % 100) for i in range(1000)],
        'response_time': np.random.gamma(2, 2, 1000),
        'status': np.random.choice(['success', 'failed'], 1000, p=[0.9, 0.1]),
        'platform': np.random.choice(['legistar', 'civic_plus', 'custom'], 1000),
        'category': np.random.choice(['municipal_council', 'federal_parliament'], 1000)
    })
    
    # Train models
    await engine.train_models(historical_data)
    
    # Get insights
    insights = engine.get_optimization_insights()
    logger.info(f"Optimization insights: {insights}")
    
    return engine


if __name__ == "__main__":
    asyncio.run(setup_ml_optimization())