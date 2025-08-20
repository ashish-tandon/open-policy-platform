"""
Comprehensive Alerting and Anomaly Detection System - 40by6
Multi-channel alerting with intelligent anomaly detection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
from twilio.rest import Client as TwilioClient
import slack_sdk
from slack_sdk.webhook.async_client import AsyncWebhookClient
import telegram
from telegram import Bot
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis
from jinja2 import Template
import os
from collections import defaultdict, deque
import yaml

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Available alert channels"""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    DASHBOARD = "dashboard"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(Enum):
    """Types of anomalies"""
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    THRESHOLD = "threshold"
    TREND = "trend"
    CORRELATION = "correlation"
    SEASONAL = "seasonal"


@dataclass
class Alert:
    """Represents an alert"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    category: str
    source: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    channels: List[AlertChannel] = field(default_factory=list)
    anomaly_type: Optional[AnomalyType] = None
    anomaly_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'category': self.category,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'context': self.context,
            'anomaly_type': self.anomaly_type.value if self.anomaly_type else None,
            'anomaly_score': self.anomaly_score
        }


@dataclass
class AlertRule:
    """Defines an alert rule"""
    id: str
    name: str
    description: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown: int = 300  # seconds
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertChannelHandler:
    """Base class for alert channel handlers"""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        raise NotImplementedError


class EmailAlertHandler(AlertChannelHandler):
    """Email alert handler"""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_server = smtp_config.get('server', 'smtp.gmail.com')
        self.smtp_port = smtp_config.get('port', 587)
        self.username = smtp_config.get('username')
        self.password = smtp_config.get('password')
        self.from_email = smtp_config.get('from_email')
        self.to_emails = smtp_config.get('to_emails', [])
        
        # Email templates
        self.templates = self._load_email_templates()
    
    def _load_email_templates(self) -> Dict[str, Template]:
        """Load email templates"""
        return {
            'default': Template("""
                <html>
                <body style="font-family: Arial, sans-serif;">
                    <div style="background-color: {{ color }}; padding: 20px; color: white;">
                        <h2>{{ alert.title }}</h2>
                        <p>Severity: {{ alert.severity.value | upper }}</p>
                    </div>
                    <div style="padding: 20px;">
                        <p><strong>Description:</strong> {{ alert.description }}</p>
                        <p><strong>Source:</strong> {{ alert.source }}</p>
                        <p><strong>Time:</strong> {{ alert.timestamp }}</p>
                        
                        {% if alert.metrics %}
                        <h3>Metrics:</h3>
                        <ul>
                        {% for key, value in alert.metrics.items() %}
                            <li><strong>{{ key }}:</strong> {{ value }}</li>
                        {% endfor %}
                        </ul>
                        {% endif %}
                        
                        {% if alert.anomaly_type %}
                        <p><strong>Anomaly Type:</strong> {{ alert.anomaly_type.value }}</p>
                        <p><strong>Anomaly Score:</strong> {{ "%.2f"|format(alert.anomaly_score) }}</p>
                        {% endif %}
                    </div>
                </body>
                </html>
            """)
        }
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send email alert"""
        try:
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: '#17a2b8',
                AlertSeverity.WARNING: '#ffc107',
                AlertSeverity.ERROR: '#dc3545',
                AlertSeverity.CRITICAL: '#721c24',
                AlertSeverity.EMERGENCY: '#000000'
            }
            
            # Render email content
            template = self.templates.get(alert.category, self.templates['default'])
            html_content = template.render(
                alert=alert,
                color=color_map.get(alert.severity, '#6c757d')
            )
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class SlackAlertHandler(AlertChannelHandler):
    """Slack alert handler"""
    
    def __init__(self, slack_config: Dict[str, Any]):
        self.webhook_url = slack_config.get('webhook_url')
        self.channel = slack_config.get('channel', '#alerts')
        self.username = slack_config.get('username', 'Alert Bot')
        self.client = AsyncWebhookClient(self.webhook_url) if self.webhook_url else None
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send Slack alert"""
        if not self.client:
            return False
        
        try:
            # Determine emoji based on severity
            emoji_map = {
                AlertSeverity.INFO: ':information_source:',
                AlertSeverity.WARNING: ':warning:',
                AlertSeverity.ERROR: ':x:',
                AlertSeverity.CRITICAL: ':rotating_light:',
                AlertSeverity.EMERGENCY: ':sos:'
            }
            
            # Build message blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji_map.get(alert.severity, ':bell:')} {alert.title}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": alert.description
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:* {alert.severity.value} | *Source:* {alert.source} | *Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        }
                    ]
                }
            ]
            
            # Add metrics if present
            if alert.metrics:
                metric_text = "\n".join([f"• *{k}:* {v}" for k, v in alert.metrics.items()])
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Metrics:*\n{metric_text}"
                    }
                })
            
            # Add anomaly info if present
            if alert.anomaly_type:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Anomaly Detected:* {alert.anomaly_type.value} (score: {alert.anomaly_score:.2f})"
                    }
                })
            
            # Send message
            response = await self.client.send(
                text=f"{alert.severity.value.upper()}: {alert.title}",
                blocks=blocks,
                username=self.username
            )
            
            logger.info(f"Slack alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class SMSAlertHandler(AlertChannelHandler):
    """SMS alert handler using Twilio"""
    
    def __init__(self, twilio_config: Dict[str, Any]):
        self.account_sid = twilio_config.get('account_sid')
        self.auth_token = twilio_config.get('auth_token')
        self.from_number = twilio_config.get('from_number')
        self.to_numbers = twilio_config.get('to_numbers', [])
        self.client = TwilioClient(self.account_sid, self.auth_token) if self.account_sid else None
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send SMS alert"""
        if not self.client:
            return False
        
        try:
            # Create concise message
            message = f"[{alert.severity.value.upper()}] {alert.title}\n{alert.description[:100]}"
            if alert.anomaly_type:
                message += f"\nAnomaly: {alert.anomaly_type.value}"
            
            # Send to all recipients
            for to_number in self.to_numbers:
                self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=to_number
                )
            
            logger.info(f"SMS alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
            return False


class WebhookAlertHandler(AlertChannelHandler):
    """Generic webhook alert handler"""
    
    def __init__(self, webhook_config: Dict[str, Any]):
        self.url = webhook_config.get('url')
        self.headers = webhook_config.get('headers', {})
        self.method = webhook_config.get('method', 'POST')
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send webhook alert"""
        if not self.url:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=self.method,
                    url=self.url,
                    json=alert.to_dict(),
                    headers=self.headers
                ) as response:
                    success = response.status < 400
                    
                    if success:
                        logger.info(f"Webhook alert sent: {alert.title}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
                    
                    return success
                    
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class AnomalyDetector:
    """Advanced anomaly detection system"""
    
    def __init__(self):
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.seasonal_patterns: Dict[str, Any] = {}
    
    async def detect_anomalies(self, metric_name: str, value: float, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies in metric"""
        # Add to historical data
        self.historical_data[metric_name].append({
            'timestamp': datetime.utcnow(),
            'value': value,
            'context': context
        })
        
        # Need enough data for detection
        if len(self.historical_data[metric_name]) < 50:
            return None
        
        anomalies = []
        
        # Statistical anomaly detection
        stat_anomaly = self._detect_statistical_anomaly(metric_name, value)
        if stat_anomaly:
            anomalies.append(stat_anomaly)
        
        # Pattern anomaly detection
        pattern_anomaly = self._detect_pattern_anomaly(metric_name, value)
        if pattern_anomaly:
            anomalies.append(pattern_anomaly)
        
        # Trend anomaly detection
        trend_anomaly = self._detect_trend_anomaly(metric_name)
        if trend_anomaly:
            anomalies.append(trend_anomaly)
        
        # Correlation anomaly detection
        if self.correlation_matrix is not None:
            corr_anomaly = self._detect_correlation_anomaly(metric_name, value)
            if corr_anomaly:
                anomalies.append(corr_anomaly)
        
        # Return most severe anomaly
        if anomalies:
            return max(anomalies, key=lambda x: x['score'])
        
        return None
    
    def _detect_statistical_anomaly(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """Detect statistical anomalies using Isolation Forest"""
        # Prepare data
        data = [d['value'] for d in self.historical_data[metric_name]]
        X = np.array(data).reshape(-1, 1)
        
        # Train or update model
        if metric_name not in self.models:
            self.models[metric_name] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.scalers[metric_name] = StandardScaler()
            X_scaled = self.scalers[metric_name].fit_transform(X)
            self.models[metric_name].fit(X_scaled)
        
        # Predict
        value_scaled = self.scalers[metric_name].transform([[value]])
        anomaly_score = self.models[metric_name].decision_function(value_scaled)[0]
        is_anomaly = self.models[metric_name].predict(value_scaled)[0] == -1
        
        if is_anomaly:
            # Calculate z-score for context
            mean = np.mean(data)
            std = np.std(data)
            z_score = (value - mean) / std if std > 0 else 0
            
            return {
                'type': AnomalyType.STATISTICAL,
                'score': abs(anomaly_score),
                'details': {
                    'value': value,
                    'mean': mean,
                    'std': std,
                    'z_score': z_score,
                    'isolation_score': anomaly_score
                }
            }
        
        return None
    
    def _detect_pattern_anomaly(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """Detect pattern-based anomalies"""
        data = [d['value'] for d in self.historical_data[metric_name]]
        
        if len(data) < 10:
            return None
        
        # Check for sudden spikes
        recent_avg = np.mean(data[-10:-1])
        if abs(value - recent_avg) > 3 * np.std(data[-10:]):
            return {
                'type': AnomalyType.PATTERN,
                'score': abs(value - recent_avg) / (np.std(data[-10:]) + 1),
                'details': {
                    'pattern': 'sudden_spike',
                    'value': value,
                    'recent_average': recent_avg,
                    'deviation': abs(value - recent_avg)
                }
            }
        
        # Check for unusual patterns
        # Implement more sophisticated pattern detection here
        
        return None
    
    def _detect_trend_anomaly(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Detect trend anomalies"""
        data = [d['value'] for d in self.historical_data[metric_name]]
        
        if len(data) < 20:
            return None
        
        # Simple trend detection using linear regression
        x = np.arange(len(data))
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        # Check for significant trend change
        recent_data = data[-10:]
        recent_x = np.arange(len(recent_data))
        recent_slope, _, _, _, _ = stats.linregress(recent_x, recent_data)
        
        if abs(recent_slope - slope) > 2 * std_err:
            return {
                'type': AnomalyType.TREND,
                'score': abs(recent_slope - slope) / (std_err + 1),
                'details': {
                    'overall_trend': slope,
                    'recent_trend': recent_slope,
                    'trend_change': recent_slope - slope,
                    'significance': p_value
                }
            }
        
        return None
    
    def _detect_correlation_anomaly(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """Detect correlation-based anomalies"""
        # This would check if correlated metrics are behaving unusually
        # Implementation depends on having correlation matrix built
        return None
    
    def update_correlation_matrix(self, metrics_df: pd.DataFrame):
        """Update correlation matrix for multi-metric anomaly detection"""
        self.correlation_matrix = metrics_df.corr()
    
    def learn_seasonal_patterns(self, metric_name: str, period: str = 'daily'):
        """Learn seasonal patterns for better anomaly detection"""
        data = [d for d in self.historical_data[metric_name]]
        
        if len(data) < 168:  # Need at least a week of hourly data
            return
        
        df = pd.DataFrame(data)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Calculate hourly patterns
        hourly_pattern = df.groupby('hour')['value'].agg(['mean', 'std'])
        
        # Calculate daily patterns
        daily_pattern = df.groupby('day_of_week')['value'].agg(['mean', 'std'])
        
        self.seasonal_patterns[metric_name] = {
            'hourly': hourly_pattern,
            'daily': daily_pattern
        }


class ComprehensiveAlertingSystem:
    """Main alerting system coordinator"""
    
    def __init__(self, config_path: str = 'alerting_config.yaml'):
        self.config = self._load_config(config_path)
        self.handlers = self._setup_handlers()
        self.rules = self._load_rules()
        self.anomaly_detector = AnomalyDetector()
        self.alert_history: deque = deque(maxlen=1000)
        self.cooldowns: Dict[str, datetime] = {}
        self.redis_client = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load alerting configuration"""
        # Default configuration
        default_config = {
            'channels': {
                'email': {
                    'enabled': False,
                    'server': 'smtp.gmail.com',
                    'port': 587
                },
                'slack': {
                    'enabled': False
                },
                'sms': {
                    'enabled': False
                },
                'webhook': {
                    'enabled': False
                }
            },
            'rules': [],
            'anomaly_detection': {
                'enabled': True,
                'sensitivity': 0.1
            }
        }
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    default_config.update(loaded_config)
        
        return default_config
    
    def _setup_handlers(self) -> Dict[AlertChannel, AlertChannelHandler]:
        """Setup alert channel handlers"""
        handlers = {}
        
        if self.config['channels']['email'].get('enabled'):
            handlers[AlertChannel.EMAIL] = EmailAlertHandler(self.config['channels']['email'])
        
        if self.config['channels']['slack'].get('enabled'):
            handlers[AlertChannel.SLACK] = SlackAlertHandler(self.config['channels']['slack'])
        
        if self.config['channels']['sms'].get('enabled'):
            handlers[AlertChannel.SMS] = SMSAlertHandler(self.config['channels']['sms'])
        
        if self.config['channels']['webhook'].get('enabled'):
            handlers[AlertChannel.WEBHOOK] = WebhookAlertHandler(self.config['channels']['webhook'])
        
        return handlers
    
    def _load_rules(self) -> List[AlertRule]:
        """Load alert rules"""
        rules = []
        
        # Default rules
        default_rules = [
            AlertRule(
                id='high_error_rate',
                name='High Error Rate',
                description='System error rate exceeds threshold',
                condition='error_rate > 0.2',
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
            ),
            AlertRule(
                id='critical_error_rate',
                name='Critical Error Rate',
                description='System error rate critically high',
                condition='error_rate > 0.5',
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS]
            ),
            AlertRule(
                id='disk_space_warning',
                name='Low Disk Space',
                description='Disk space running low',
                condition='disk_usage_percent > 85',
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL]
            ),
            AlertRule(
                id='scraper_failures',
                name='Multiple Scraper Failures',
                description='Multiple scrapers failing',
                condition='failed_scrapers > 10',
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
            )
        ]
        
        rules.extend(default_rules)
        
        # Load custom rules from config
        for rule_config in self.config.get('rules', []):
            rule = AlertRule(**rule_config)
            rules.append(rule)
        
        return rules
    
    async def initialize(self):
        """Initialize the alerting system"""
        self.redis_client = await redis.from_url(
            os.getenv('REDIS_URL', 'redis://localhost:6379')
        )
        logger.info("Comprehensive alerting system initialized")
    
    async def check_metrics(self, metrics: Dict[str, float]):
        """Check metrics against rules and anomalies"""
        # Check rules
        for rule in self.rules:
            if rule.enabled:
                await self._check_rule(rule, metrics)
        
        # Check anomalies if enabled
        if self.config['anomaly_detection']['enabled']:
            for metric_name, value in metrics.items():
                anomaly = await self.anomaly_detector.detect_anomalies(
                    metric_name, value, {'source': 'system'}
                )
                
                if anomaly:
                    await self._handle_anomaly(metric_name, value, anomaly)
    
    async def _check_rule(self, rule: AlertRule, metrics: Dict[str, float]):
        """Check if rule condition is met"""
        try:
            # Simple evaluation (in production, use safe expression evaluator)
            local_vars = metrics.copy()
            condition_met = eval(rule.condition, {"__builtins__": {}}, local_vars)
            
            if condition_met:
                # Check cooldown
                if rule.id in self.cooldowns:
                    if datetime.utcnow() < self.cooldowns[rule.id]:
                        return  # Still in cooldown
                
                # Create alert
                alert = Alert(
                    id=f"{rule.id}_{datetime.utcnow().timestamp()}",
                    title=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    category='rule',
                    source='rule_engine',
                    timestamp=datetime.utcnow(),
                    metrics={k: v for k, v in metrics.items() if k in rule.condition},
                    context={'rule_id': rule.id},
                    channels=rule.channels
                )
                
                await self.send_alert(alert)
                
                # Set cooldown
                self.cooldowns[rule.id] = datetime.utcnow() + timedelta(seconds=rule.cooldown)
                
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.id}: {e}")
    
    async def _handle_anomaly(self, metric_name: str, value: float, anomaly: Dict[str, Any]):
        """Handle detected anomaly"""
        # Determine severity based on anomaly score
        score = anomaly['score']
        if score > 5:
            severity = AlertSeverity.CRITICAL
        elif score > 3:
            severity = AlertSeverity.ERROR
        elif score > 2:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        # Create alert
        alert = Alert(
            id=f"anomaly_{metric_name}_{datetime.utcnow().timestamp()}",
            title=f"Anomaly Detected: {metric_name}",
            description=f"Unusual behavior detected in {metric_name} metric",
            severity=severity,
            category='anomaly',
            source='anomaly_detector',
            timestamp=datetime.utcnow(),
            metrics={metric_name: value},
            context=anomaly['details'],
            channels=self._get_anomaly_channels(severity),
            anomaly_type=anomaly['type'],
            anomaly_score=score
        )
        
        await self.send_alert(alert)
    
    def _get_anomaly_channels(self, severity: AlertSeverity) -> List[AlertChannel]:
        """Determine channels for anomaly alerts based on severity"""
        if severity == AlertSeverity.CRITICAL:
            return [AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS]
        elif severity == AlertSeverity.ERROR:
            return [AlertChannel.EMAIL, AlertChannel.SLACK]
        elif severity == AlertSeverity.WARNING:
            return [AlertChannel.SLACK]
        else:
            return [AlertChannel.DASHBOARD]
    
    async def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        logger.info(f"Sending alert: {alert.title} (severity: {alert.severity.value})")
        
        # Record alert
        self.alert_history.append(alert)
        
        # Store in Redis for dashboard
        await self._store_alert(alert)
        
        # Send through each channel
        results = {}
        for channel in alert.channels:
            if channel in self.handlers:
                try:
                    success = await self.handlers[channel].send_alert(alert)
                    results[channel] = success
                except Exception as e:
                    logger.error(f"Error sending alert via {channel.value}: {e}")
                    results[channel] = False
        
        # Log results
        success_count = sum(1 for s in results.values() if s)
        logger.info(f"Alert sent successfully to {success_count}/{len(results)} channels")
    
    async def _store_alert(self, alert: Alert):
        """Store alert in Redis for dashboard access"""
        if self.redis_client:
            # Store in sorted set by timestamp
            await self.redis_client.zadd(
                'alerts:history',
                {json.dumps(alert.to_dict()): alert.timestamp.timestamp()}
            )
            
            # Keep only last 10000 alerts
            await self.redis_client.zremrangebyrank('alerts:history', 0, -10001)
            
            # Update counters
            await self.redis_client.hincrby('alerts:stats', alert.severity.value, 1)
            await self.redis_client.hincrby('alerts:stats', 'total', 1)
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting statistics"""
        if not self.redis_client:
            return {}
        
        # Get stats from Redis
        stats = await self.redis_client.hgetall('alerts:stats')
        
        # Get recent alerts
        recent = await self.redis_client.zrevrange('alerts:history', 0, 99, withscores=True)
        recent_alerts = []
        for alert_json, score in recent:
            alert_data = json.loads(alert_json)
            recent_alerts.append(alert_data)
        
        return {
            'total_alerts': int(stats.get(b'total', 0)),
            'by_severity': {
                severity.value: int(stats.get(severity.value.encode(), 0))
                for severity in AlertSeverity
            },
            'recent_alerts': recent_alerts[:20],
            'active_rules': len([r for r in self.rules if r.enabled]),
            'anomaly_detection_enabled': self.config['anomaly_detection']['enabled']
        }
    
    def add_custom_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.rules.append(rule)
        logger.info(f"Added custom rule: {rule.name}")
    
    def update_anomaly_sensitivity(self, sensitivity: float):
        """Update anomaly detection sensitivity"""
        self.config['anomaly_detection']['sensitivity'] = sensitivity
        # Update models
        for model in self.anomaly_detector.models.values():
            model.contamination = sensitivity


# Example usage
async def setup_alerting_system():
    """Setup and start the alerting system"""
    alerting = ComprehensiveAlertingSystem()
    await alerting.initialize()
    
    # Add custom rule
    custom_rule = AlertRule(
        id='custom_data_quality',
        name='Low Data Quality',
        description='Data quality score below threshold',
        condition='data_quality_score < 0.6',
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
    )
    alerting.add_custom_rule(custom_rule)
    
    # Simulate metrics check
    metrics = {
        'error_rate': 0.15,
        'disk_usage_percent': 75,
        'failed_scrapers': 5,
        'data_quality_score': 0.85,
        'response_time': 5.2
    }
    
    await alerting.check_metrics(metrics)
    
    # Get stats
    stats = await alerting.get_alert_stats()
    logger.info(f"Alert stats: {stats}")
    
    return alerting


if __name__ == "__main__":
    asyncio.run(setup_alerting_system())