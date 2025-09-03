"""Main scheduler for orchestrating bot agents with asyncio."""

import asyncio
import signal
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from loguru import logger
from prometheus_client import start_http_server, Counter, Gauge, Histogram

from src.config.settings import BotConfig
from src.database.mongodb_manager import MongoDBManager
from src.database.postgres_manager import PostgreSQLManager
from src.utils.rate_limiter import RateLimiter
from src.agents.follow_agent import FollowAgent
from src.agents.content_agent import ContentAgent
from src.agents.engagement_agent import EngagementAgent
from src.models.data_models import BotMetrics


# Prometheus metrics
scheduler_cycles_total = Counter('scheduler_cycles_total', 'Total scheduler cycles')
scheduler_cycle_duration = Histogram('scheduler_cycle_duration_seconds', 'Scheduler cycle duration')
agent_last_run_timestamp = Gauge('agent_last_run_timestamp', 'Last run timestamp for each agent', ['agent'])
bot_uptime_seconds = Gauge('bot_uptime_seconds', 'Bot uptime in seconds')


@dataclass
class AgentSchedule:
    """Agent execution schedule configuration."""
    agent_name: str
    interval_minutes: int
    last_run: Optional[datetime] = None
    enabled: bool = True
    priority: int = 0  # Lower number = higher priority


class BotScheduler:
    """Main scheduler for orchestrating all bot agents."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.running = False
        self.start_time = datetime.utcnow()

        # Database managers
        self.mongodb = MongoDBManager(config.database)
        self.postgres = PostgreSQLManager(config.database)

        # Rate limiter
        self.rate_limiter = RateLimiter()

        # Agents
        self.agents: Dict[str, Any] = {}
        self.agent_schedules: List[AgentSchedule] = []

        # Execution tracking
        self.cycle_count = 0
        self.last_health_check = datetime.utcnow()

        # Task management
        self.running_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            logger.info("Initializing bot scheduler...")

            # Initialize databases
            await self.mongodb.connect()
            await self.postgres.connect()

            # Initialize agents
            await self._initialize_agents()

            # Setup agent schedules
            self._setup_schedules()

            # Start metrics server
            if self.config.monitoring.prometheus_enabled:
                start_http_server(self.config.monitoring.metrics_port)
                logger.info(f"Metrics server started on port {self.config.monitoring.metrics_port}")

            # Setup signal handlers
            self._setup_signal_handlers()

            logger.info("Bot scheduler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            raise

    async def _initialize_agents(self) -> None:
        """Initialize all bot agents."""
        try:
            # Initialize agents with shared dependencies
            self.agents = {
                "follow": FollowAgent(
                    config=self.config,
                    mongodb=self.mongodb,
                    postgres=self.postgres,
                    rate_limiter=self.rate_limiter
                ),
                "content": ContentAgent(
                    config=self.config,
                    mongodb=self.mongodb,
                    postgres=self.postgres,
                    rate_limiter=self.rate_limiter
                ),
                "engagement": EngagementAgent(
                    config=self.config,
                    mongodb=self.mongodb,
                    postgres=self.postgres,
                    rate_limiter=self.rate_limiter
                )
            }

            logger.info(f"Initialized {len(self.agents)} agents")

        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise

    def _setup_schedules(self) -> None:
        """Setup execution schedules for each agent."""
        self.agent_schedules = [
            # Content agent runs every 12 hours to minimize API usage
            AgentSchedule(
                agent_name="content", interval_minutes=720, priority=1, enabled=True
            ),
            # Follow agent runs moderately to discover and manage users
            AgentSchedule(
                agent_name="follow", interval_minutes=60, priority=2, enabled=True
            ),
            # Engagement agent runs frequently for timeline monitoring
            AgentSchedule(
                agent_name="engagement", interval_minutes=45, priority=1, enabled=True
            ),
        ]

        # Adjust intervals based on development mode
        if self.config.development.development_mode:
            for schedule in self.agent_schedules:
                schedule.interval_minutes = max(5, schedule.interval_minutes // 6)  # 6x faster

        logger.info(f"Setup {len(self.agent_schedules)} agent schedules")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers setup complete")
        except Exception as e:
            logger.error(f"Failed to setup signal handlers: {e}")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")

        # Create shutdown task
        if self.running:
            asyncio.create_task(self.shutdown())

    async def start(self) -> None:
        """Start the bot scheduler."""
        try:
            await self.initialize()

            self.running = True
            logger.info("Bot scheduler started")

            # Start main execution loop
            await self._main_loop()

        except Exception as e:
            logger.error(f"Scheduler failed: {e}")
            raise
        finally:
            await self.shutdown()

    async def _main_loop(self) -> None:
        """Main execution loop."""
        logger.info("Starting main execution loop")

        while self.running:
            cycle_start = asyncio.get_event_loop().time()

            try:
                with scheduler_cycle_duration.time():
                    await self._execute_cycle()

                scheduler_cycles_total.inc()
                self.cycle_count += 1

                # Update uptime metric
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                bot_uptime_seconds.set(uptime)

                # Periodic health checks
                if self.cycle_count % 10 == 0:  # Every 10 cycles
                    await self._health_check()

                # Periodic cleanup
                if self.cycle_count % 100 == 0:  # Every 100 cycles
                    await self._cleanup()

            except Exception as e:
                logger.error(f"Error in main loop cycle: {e}")

                # Record error metric
                await self._record_system_metric("scheduler_error", 1.0, {"error": str(e)})

                # Brief pause on error to avoid tight loop
                await asyncio.sleep(30)

            # Calculate sleep time for next cycle (aim for 1-minute cycles)
            cycle_duration = asyncio.get_event_loop().time() - cycle_start
            sleep_time = max(0, 60 - cycle_duration)

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _execute_cycle(self) -> None:
        """Execute one scheduler cycle."""
        logger.debug(f"Starting scheduler cycle {self.cycle_count + 1}")

        # Get agents ready to run
        ready_agents = self._get_ready_agents()

        if not ready_agents:
            logger.debug("No agents ready to run")
            return

        # Sort by priority (lower number = higher priority)
        ready_agents.sort(key=lambda x: x.priority)

        # Execute agents concurrently (up to a limit)
        max_concurrent = 2  # Limit concurrent agent execution

        for i in range(0, len(ready_agents), max_concurrent):
            batch = ready_agents[i:i + max_concurrent]

            # Create tasks for this batch
            tasks = []
            for schedule in batch:
                if schedule.agent_name in self.running_tasks:
                    # Skip if agent is already running
                    logger.debug(f"Skipping {schedule.agent_name} - already running")
                    continue

                task = asyncio.create_task(
                    self._execute_agent(schedule),
                    name=f"agent_{schedule.agent_name}"
                )
                tasks.append(task)
                self.running_tasks[schedule.agent_name] = task

            # Wait for batch to complete
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Error executing agent batch: {e}")
                finally:
                    # Clean up completed tasks
                    for schedule in batch:
                        self.running_tasks.pop(schedule.agent_name, None)

    def _get_ready_agents(self) -> List[AgentSchedule]:
        """Get list of agents ready to run."""
        ready = []
        now = datetime.utcnow()

        for schedule in self.agent_schedules:
            if not schedule.enabled:
                continue

            # Check if enough time has passed since last run
            if schedule.last_run is None:
                ready.append(schedule)
            else:
                time_since_last = now - schedule.last_run
                interval = timedelta(minutes=schedule.interval_minutes)

                if time_since_last >= interval:
                    ready.append(schedule)

        return ready

    async def _execute_agent(self, schedule: AgentSchedule) -> None:
        """Execute a single agent."""
        agent_name = schedule.agent_name
        agent = self.agents.get(agent_name)

        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return

        start_time = datetime.utcnow()

        try:
            logger.info(f"Executing {agent_name} agent")

            # Run agent
            result = await agent.run_once()

            # Update schedule
            schedule.last_run = start_time

            # Update metrics
            agent_last_run_timestamp.labels(agent=agent_name).set(start_time.timestamp())

            # Record execution metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._record_agent_metric(
                agent_name,
                "execution_time",
                execution_time,
                {"success": result.get("success", False)}
            )

            if result.get("success"):
                logger.info(f"{agent_name} agent completed successfully in {execution_time:.1f}s")
            else:
                logger.warning(f"{agent_name} agent completed with issues: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Agent {agent_name} execution failed: {e}")

            # Record error
            await self._record_agent_metric(
                agent_name,
                "execution_error",
                1.0,
                {"error": str(e)}
            )

            # Update last_run even on failure to prevent rapid retries
            schedule.last_run = start_time

    async def _health_check(self) -> None:
        """Perform system health check."""
        try:
            logger.debug("Performing health check")

            # Check database connections
            mongodb_healthy = self.mongodb._connected
            postgres_healthy = self.postgres._connected

            # Check agent health
            agent_health = {}
            for name, agent in self.agents.items():
                status = agent.get_health_status()
                agent_health[name] = {
                    "error_rate": status["error_rate"],
                    "last_run": status["last_run"],
                    "backoff_active": status["backoff_active"]
                }

            # Check rate limiter status
            rate_limit_status = self.rate_limiter.get_status()

            # Record health metrics
            await self._record_system_metric("mongodb_healthy", 1.0 if mongodb_healthy else 0.0)
            await self._record_system_metric("postgres_healthy", 1.0 if postgres_healthy else 0.0)

            # Log health summary
            unhealthy_agents = [
                name for name, health in agent_health.items()
                if health["error_rate"] > 0.5 or health["backoff_active"]
            ]

            if unhealthy_agents:
                logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")

            self.last_health_check = datetime.utcnow()

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    async def _cleanup(self) -> None:
        """Perform periodic cleanup tasks."""
        try:
            logger.info("Performing periodic cleanup")

            # Clean up old data
            cleanup_stats = await self.mongodb.cleanup_old_data(days_to_keep=30)
            logger.info(f"MongoDB cleanup: {cleanup_stats}")

            # Clean expired cache
            cache_cleaned = await self.postgres.clean_expired_cache()
            logger.info(f"PostgreSQL cache cleanup: {cache_cleaned} entries removed")

            # Record cleanup metrics
            await self._record_system_metric("cleanup_completed", 1.0, cleanup_stats)

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def _record_system_metric(
        self, metric_type: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record system-level metric."""
        try:
            metric = BotMetrics(
                metric_type=metric_type,
                agent="scheduler",
                value=value,
                metadata=metadata or {}
            )

            await self.mongodb.create_metric(metric)

        except Exception as e:
            logger.error(f"Failed to record system metric: {e}")

    async def _record_agent_metric(
        self,
        agent_name: str,
        metric_type: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record agent-specific metric."""
        try:
            metric = BotMetrics(
                metric_type=metric_type,
                agent=agent_name,
                value=value,
                metadata=metadata or {}
            )

            await self.mongodb.create_metric(metric)

        except Exception as e:
            logger.error(f"Failed to record agent metric: {e}")

    async def shutdown(self) -> None:
        """Gracefully shutdown the scheduler."""
        if not self.running:
            return

        logger.info("Initiating scheduler shutdown...")
        self.running = False

        try:
            # Cancel running agent tasks
            if self.running_tasks:
                logger.info(f"Cancelling {len(self.running_tasks)} running tasks")

                for task_name, task in self.running_tasks.items():
                    if not task.done():
                        task.cancel()
                        logger.debug(f"Cancelled task: {task_name}")

                # Wait for tasks to complete cancellation
                if self.running_tasks:
                    await asyncio.gather(
                        *self.running_tasks.values(),
                        return_exceptions=True
                    )

            # Cleanup agents
            logger.info("Cleaning up agents...")
            for name, agent in self.agents.items():
                try:
                    await agent.cleanup()
                    logger.debug(f"Cleaned up agent: {name}")
                except Exception as e:
                    logger.error(f"Error cleaning up agent {name}: {e}")

            # Disconnect databases
            logger.info("Disconnecting databases...")
            await self.mongodb.disconnect()
            await self.postgres.disconnect()

            # Record final metrics
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            await self._record_system_metric("scheduler_shutdown", 1.0, {
                "uptime_seconds": uptime,
                "cycles_completed": self.cycle_count
            })

            logger.info(f"Scheduler shutdown complete. Uptime: {uptime:.1f}s, Cycles: {self.cycle_count}")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "cycle_count": self.cycle_count,
            "start_time": self.start_time.isoformat(),
            "last_health_check": self.last_health_check.isoformat(),
            "agents": {
                name: agent.get_health_status()
                for name, agent in self.agents.items()
            },
            "agent_schedules": [
                {
                    "name": s.agent_name,
                    "interval_minutes": s.interval_minutes,
                    "last_run": s.last_run.isoformat() if s.last_run else None,
                    "enabled": s.enabled,
                    "priority": s.priority
                }
                for s in self.agent_schedules
            ],
            "running_tasks": list(self.running_tasks.keys()),
            "rate_limiter": self.rate_limiter.get_status(),
            "database_status": {
                "mongodb_connected": self.mongodb._connected,
                "postgres_connected": self.postgres._connected
            }
        }
