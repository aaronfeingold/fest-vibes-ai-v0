"""Comprehensive monitoring and alerting system."""

import asyncio
import smtplib
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from src.models.data_models import BotMetrics


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class MonitoringRule:
    """Monitoring rule configuration."""
    name: str
    metric_pattern: str
    condition: str  # e.g., "> 0.8", "< 0.1"
    threshold: float
    severity: AlertSeverity
    cooldown_minutes: int = 30
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.health_status = {
            "overall": "healthy",
            "components": {},
            "last_check": datetime.utcnow()
        }
        
        # Health check functions
        self.health_checks: Dict[str, Callable] = {}
        
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    async def perform_health_checks(self) -> Dict[str, Any]:
        """Perform all registered health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                
                results[name] = {
                    "status": "healthy" if result["healthy"] else "unhealthy",
                    "details": result.get("details", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if not result["healthy"]:
                    overall_healthy = False
                    
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                overall_healthy = False
        
        self.health_status = {
            "overall": "healthy" if overall_healthy else "unhealthy",
            "components": results,
            "last_check": datetime.utcnow()
        }
        
        return self.health_status
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health_status.copy()


class AlertManager:
    """Alert management system."""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        
        # Rate limiting for alerts
        self.alert_rate_limits: Dict[str, datetime] = {}
        
    def add_notification_handler(self, handler: Callable) -> None:
        """Add a notification handler."""
        self.notification_handlers.append(handler)
        
    async def create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create and process a new alert."""
        
        # Check rate limiting
        if self._is_rate_limited(alert_id):
            logger.debug(f"Alert {alert_id} rate limited")
            return None
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Apply rate limiting
        self.alert_rate_limits[alert_id] = datetime.utcnow()
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"Alert created: {severity.value.upper()} - {title}")
        return alert
    
    def _is_rate_limited(self, alert_id: str, cooldown_minutes: int = 30) -> bool:
        """Check if alert is rate limited."""
        if alert_id not in self.alert_rate_limits:
            return False
        
        last_sent = self.alert_rate_limits[alert_id]
        cooldown = timedelta(minutes=cooldown_minutes)
        
        return datetime.utcnow() - last_sent < cooldown
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_timestamp = datetime.utcnow()
            
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications."""
        for handler in self.notification_handlers:
            try:
                await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_count,
            "severity_breakdown": severity_counts,
            "alert_rate": len([
                a for a in self.alert_history
                if a.timestamp > datetime.utcnow() - timedelta(hours=24)
            ])
        }


class MetricsCollector:
    """Advanced metrics collection and analysis."""
    
    def __init__(self):
        self.custom_registry = CollectorRegistry()
        
        # Custom metrics
        self.error_rate = Gauge('bot_error_rate', 'Bot error rate', registry=self.custom_registry)
        self.agent_performance = Gauge('agent_performance_score', 'Agent performance score', ['agent'], registry=self.custom_registry)
        self.safety_violations = Counter('safety_violations_total', 'Total safety violations', ['type'], registry=self.custom_registry)
        self.api_response_time = Histogram('api_response_time_seconds', 'API response time', ['service'], registry=self.custom_registry)
        
        # Metric thresholds
        self.thresholds = {
            "error_rate": 0.1,  # 10%
            "response_time": 5.0,  # 5 seconds
            "memory_usage": 0.8,  # 80%
            "disk_usage": 0.9   # 90%
        }
    
    def record_custom_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a custom metric."""
        try:
            if hasattr(self, metric_name):
                metric = getattr(self, metric_name)
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")
    
    def get_metrics_export(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.custom_registry).decode('utf-8')
    
    async def analyze_metrics_for_alerts(self, mongodb_manager) -> List[Dict[str, Any]]:
        """Analyze recent metrics for potential alerts."""
        alerts = []
        
        try:
            # Get recent metrics
            recent_metrics = await mongodb_manager.get_metrics(hours=1)
            
            # Analyze error rates by agent
            agent_metrics = {}
            for metric in recent_metrics:
                agent = metric.agent
                if agent not in agent_metrics:
                    agent_metrics[agent] = {"errors": 0, "total": 0}
                
                agent_metrics[agent]["total"] += 1
                if "error" in metric.metric_type:
                    agent_metrics[agent]["errors"] += 1
            
            # Check error rate thresholds
            for agent, stats in agent_metrics.items():
                if stats["total"] > 0:
                    error_rate = stats["errors"] / stats["total"]
                    if error_rate > self.thresholds["error_rate"]:
                        alerts.append({
                            "id": f"high_error_rate_{agent}",
                            "severity": AlertSeverity.WARNING,
                            "title": f"High Error Rate for {agent}",
                            "description": f"Error rate is {error_rate:.1%} (threshold: {self.thresholds['error_rate']:.1%})",
                            "metadata": {"agent": agent, "error_rate": error_rate}
                        })
        
        except Exception as e:
            logger.error(f"Metrics analysis failed: {e}")
        
        return alerts


class SafetyMonitor:
    """Safety monitoring and violation detection."""
    
    def __init__(self):
        self.safety_checks = {
            "rate_limit_violations": self._check_rate_limits,
            "content_safety": self._check_content_safety,
            "engagement_patterns": self._check_engagement_patterns,
            "api_errors": self._check_api_errors
        }
        
        self.violation_counts = {}
        self.safety_thresholds = {
            "rate_limit_violations": 5,      # per hour
            "content_safety_issues": 3,      # per day
            "suspicious_patterns": 2,        # per hour
            "api_error_rate": 0.2           # 20%
        }
    
    async def perform_safety_checks(self, mongodb_manager) -> List[Dict[str, Any]]:
        """Perform comprehensive safety checks."""
        violations = []
        
        for check_name, check_func in self.safety_checks.items():
            try:
                result = await check_func(mongodb_manager)
                if result["violations"]:
                    violations.extend(result["violations"])
                    
                    # Update violation counts
                    self.violation_counts[check_name] = self.violation_counts.get(check_name, 0) + len(result["violations"])
                    
            except Exception as e:
                logger.error(f"Safety check {check_name} failed: {e}")
        
        return violations
    
    async def _check_rate_limits(self, mongodb_manager) -> Dict[str, Any]:
        """Check for rate limit violations."""
        violations = []
        
        try:
            # Get recent engagements
            recent_engagements = await mongodb_manager.get_recent_engagements(hours=1)
            
            # Count actions by type
            action_counts = {}
            for engagement in recent_engagements:
                action_type = engagement.action_type.value
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
            
            # Check against safety thresholds
            hourly_limits = {
                "like": 30,
                "repost": 15,
                "comment": 10,
                "follow": 10
            }
            
            for action, count in action_counts.items():
                if action in hourly_limits and count > hourly_limits[action]:
                    violations.append({
                        "type": "rate_limit_violation",
                        "action": action,
                        "count": count,
                        "limit": hourly_limits[action],
                        "severity": "high"
                    })
        
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
        
        return {"violations": violations}
    
    async def _check_content_safety(self, mongodb_manager) -> Dict[str, Any]:
        """Check for content safety issues."""
        violations = []
        
        try:
            # Get recent content
            recent_content = await mongodb_manager.get_unused_content("post", limit=20)
            
            # Check for problematic content patterns
            problematic_patterns = [
                r'http[s]?://[^\s]+',  # URLs
                r'@[a-zA-Z0-9_]+\s+@[a-zA-Z0-9_]+',  # Multiple mentions
                r'#\w+\s+#\w+\s+#\w+\s+#\w+',  # Too many hashtags
                r'(follow.*back|dm.*me|check.*out)',  # Spam patterns
            ]
            
            import re
            for content in recent_content:
                for pattern in problematic_patterns:
                    if re.search(pattern, content.content, re.IGNORECASE):
                        violations.append({
                            "type": "content_safety_issue",
                            "content_id": str(content.id),
                            "pattern": pattern,
                            "content_preview": content.content[:50] + "...",
                            "severity": "medium"
                        })
        
        except Exception as e:
            logger.error(f"Content safety check failed: {e}")
        
        return {"violations": violations}
    
    async def _check_engagement_patterns(self, mongodb_manager) -> Dict[str, Any]:
        """Check for suspicious engagement patterns."""
        violations = []
        
        try:
            # Get recent successful engagements
            recent_engagements = await mongodb_manager.get_recent_engagements(hours=4)
            successful_engagements = [e for e in recent_engagements if e.success]
            
            if len(successful_engagements) > 50:  # Too many engagements
                violations.append({
                    "type": "high_engagement_volume",
                    "count": len(successful_engagements),
                    "time_window": "4 hours",
                    "severity": "medium"
                })
            
            # Check for rapid-fire engagement (same user multiple times quickly)
            user_engagement_times = {}
            for engagement in successful_engagements:
                user_id = engagement.target_user_id
                if user_id not in user_engagement_times:
                    user_engagement_times[user_id] = []
                user_engagement_times[user_id].append(engagement.timestamp)
            
            for user_id, times in user_engagement_times.items():
                if len(times) > 3:  # More than 3 engagements with same user
                    times.sort()
                    if (times[-1] - times[0]).total_seconds() < 3600:  # Within 1 hour
                        violations.append({
                            "type": "rapid_engagement_pattern",
                            "user_id": user_id,
                            "engagement_count": len(times),
                            "time_span": (times[-1] - times[0]).total_seconds(),
                            "severity": "high"
                        })
        
        except Exception as e:
            logger.error(f"Engagement pattern check failed: {e}")
        
        return {"violations": violations}
    
    async def _check_api_errors(self, mongodb_manager) -> Dict[str, Any]:
        """Check for high API error rates."""
        violations = []
        
        try:
            # Get recent metrics
            recent_metrics = await mongodb_manager.get_metrics(hours=2)
            
            # Calculate error rates
            total_requests = 0
            error_requests = 0
            
            for metric in recent_metrics:
                if "execution" in metric.metric_type:
                    total_requests += 1
                    if "error" in metric.metric_type:
                        error_requests += 1
            
            if total_requests > 0:
                error_rate = error_requests / total_requests
                if error_rate > self.safety_thresholds["api_error_rate"]:
                    violations.append({
                        "type": "high_api_error_rate",
                        "error_rate": error_rate,
                        "total_requests": total_requests,
                        "error_requests": error_requests,
                        "threshold": self.safety_thresholds["api_error_rate"],
                        "severity": "high"
                    })
        
        except Exception as e:
            logger.error(f"API error check failed: {e}")
        
        return {"violations": violations}
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            "violation_counts": self.violation_counts,
            "safety_thresholds": self.safety_thresholds,
            "last_check": datetime.utcnow().isoformat(),
            "overall_status": "safe" if all(
                count < threshold 
                for count, threshold in zip(
                    self.violation_counts.values(),
                    self.safety_thresholds.values()
                )
            ) else "at_risk"
        }


class ComprehensiveMonitor:
    """Main monitoring coordinator."""
    
    def __init__(self, config):
        self.config = config
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.metrics_collector = MetricsCollector()
        self.safety_monitor = SafetyMonitor()
        
        # Monitoring intervals
        self.health_check_interval = 300  # 5 minutes
        self.safety_check_interval = 600  # 10 minutes
        self.metrics_analysis_interval = 900  # 15 minutes
        
        # Task tracking
        self.monitoring_tasks = {}
        self.running = False
    
    async def start(self, mongodb_manager) -> None:
        """Start comprehensive monitoring."""
        self.running = True
        
        # Register default health checks
        self._register_default_health_checks(mongodb_manager)
        
        # Start monitoring tasks
        self.monitoring_tasks = {
            "health_check": asyncio.create_task(
                self._health_check_loop(mongodb_manager)
            ),
            "safety_monitor": asyncio.create_task(
                self._safety_monitor_loop(mongodb_manager)
            ),
            "metrics_analysis": asyncio.create_task(
                self._metrics_analysis_loop(mongodb_manager)
            )
        }
        
        logger.info("Comprehensive monitoring started")
    
    def _register_default_health_checks(self, mongodb_manager) -> None:
        """Register default health check functions."""
        
        async def database_health():
            return {
                "healthy": mongodb_manager._connected,
                "details": {"connection": "active" if mongodb_manager._connected else "inactive"}
            }
        
        async def memory_health():
            import psutil
            memory = psutil.virtual_memory()
            return {
                "healthy": memory.percent < 90,
                "details": {"usage_percent": memory.percent, "available_gb": memory.available / (1024**3)}
            }
        
        async def disk_health():
            import psutil
            disk = psutil.disk_usage('/')
            return {
                "healthy": disk.percent < 90,
                "details": {"usage_percent": disk.percent, "free_gb": disk.free / (1024**3)}
            }
        
        self.health_checker.register_health_check("database", database_health)
        self.health_checker.register_health_check("memory", memory_health)
        self.health_checker.register_health_check("disk", disk_health)
    
    async def _health_check_loop(self, mongodb_manager) -> None:
        """Health check monitoring loop."""
        while self.running:
            try:
                health_status = await self.health_checker.perform_health_checks()
                
                # Create alerts for unhealthy components
                for component, status in health_status["components"].items():
                    if status["status"] != "healthy":
                        await self.alert_manager.create_alert(
                            alert_id=f"health_{component}",
                            severity=AlertSeverity.WARNING if status["status"] == "unhealthy" else AlertSeverity.ERROR,
                            title=f"Health Check Failed: {component}",
                            description=f"Component {component} is {status['status']}",
                            metadata=status
                        )
                    else:
                        # Resolve alert if component is now healthy
                        await self.alert_manager.resolve_alert(f"health_{component}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)
    
    async def _safety_monitor_loop(self, mongodb_manager) -> None:
        """Safety monitoring loop."""
        while self.running:
            try:
                violations = await self.safety_monitor.perform_safety_checks(mongodb_manager)
                
                # Create alerts for safety violations
                for violation in violations:
                    severity = AlertSeverity.ERROR if violation["severity"] == "high" else AlertSeverity.WARNING
                    
                    await self.alert_manager.create_alert(
                        alert_id=f"safety_{violation['type']}",
                        severity=severity,
                        title=f"Safety Violation: {violation['type'].replace('_', ' ').title()}",
                        description=f"Safety violation detected: {violation}",
                        metadata=violation
                    )
                
                await asyncio.sleep(self.safety_check_interval)
                
            except Exception as e:
                logger.error(f"Safety monitor loop error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_analysis_loop(self, mongodb_manager) -> None:
        """Metrics analysis loop."""
        while self.running:
            try:
                alerts = await self.metrics_collector.analyze_metrics_for_alerts(mongodb_manager)
                
                # Create alerts from metrics analysis
                for alert_data in alerts:
                    await self.alert_manager.create_alert(**alert_data)
                
                await asyncio.sleep(self.metrics_analysis_interval)
                
            except Exception as e:
                logger.error(f"Metrics analysis loop error: {e}")
                await asyncio.sleep(60)
    
    async def stop(self) -> None:
        """Stop monitoring."""
        self.running = False
        
        # Cancel monitoring tasks
        for task_name, task in self.monitoring_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Comprehensive monitoring stopped")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get overall monitoring status."""
        return {
            "running": self.running,
            "health_status": self.health_checker.get_health_status(),
            "alert_stats": self.alert_manager.get_alert_stats(),
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alert_manager.get_active_alerts()
            ],
            "safety_status": self.safety_monitor.get_safety_status(),
            "monitoring_tasks": {
                name: not task.done()
                for name, task in self.monitoring_tasks.items()
            }
        }