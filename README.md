# Autonomous Twitter Bot System

An intelligent, multi-agent Twitter bot system designed for the New Orleans music and culture domain. The system uses AI agents for autonomous decision-making across user discovery, content generation, and strategic engagement.

## Features

- **Multi-Agent Architecture**: Specialized agents for following, content creation, and engagement
- **AI-Powered Content**: LLM integration for authentic, contextual content generation
- **Smart User Discovery**: Vector-based similarity matching for relevant user targeting
- **Safety-First Design**: Comprehensive rate limiting and spam prevention
- **Real-Time Monitoring**: Prometheus metrics with Grafana dashboards
- **Semantic Search**: PostgreSQL with pgvector for content similarity analysis

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Scheduler     │    │  Agent Manager   │    │  Twitter API    │
│   (Main Loop)   │───▶│   (Orchestrator) │───▶│   Interface     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Follow Agent │ │Content Agent │ │Engage Agent  │
            │              │ │              │ │              │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                    ┌──────────────────────┐
                    │   Data Layer        │
                    │ MongoDB + PostgreSQL │
                    └──────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Twitter Developer Account
- OpenAI or Anthropic API key

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd fest-vibes-ai-v0
```

2. **Install dependencies**

```bash
poetry install
```

3. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. **Start services**

```bash
docker-compose up -d
```

5. **Run the bot**

```bash
poetry run python src/main.py
```

### Environment Variables

Create a `.env` file with the following variables:

```env
# Twitter API Configuration
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
TWITTER_BEARER_TOKEN=your_bearer_token

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key

# Bot Configuration
BOT_USERNAME=your_bot_username
TARGET_LOCATION=New Orleans
FOLLOW_THRESHOLD=0.7
LIKE_THRESHOLD=0.6
REPOST_THRESHOLD=0.8
COMMENT_THRESHOLD=0.9

# Rate Limits
MAX_FOLLOWS_PER_DAY=50
MAX_POSTS_PER_DAY=10
MAX_LIKES_PER_HOUR=30
```

## Agents

### Follow Agent

- **Purpose**: Discovers and manages user relationships
- **Capabilities**:
  - Location-based user discovery
  - Relevance scoring using ML
  - Intelligent follow/unfollow decisions
  - Relationship tracking and analytics

### Content Agent

- **Purpose**: Generates authentic, engaging content
- **Capabilities**:
  - LLM-powered post generation
  - Contextual comment creation
  - Content cache management
  - Brand voice consistency
  - Performance tracking

### Engagement Agent

- **Purpose**: Strategic social media interactions
- **Capabilities**:
  - Timeline analysis and scoring
  - Smart like/repost decisions
  - Contextual commenting
  - Engagement pattern optimization

## Monitoring

Access monitoring dashboards at:

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Bot Metrics**: http://localhost:8000/metrics

### Key Metrics

- Agent execution rates and success rates
- Rate limit utilization
- Content generation performance
- Engagement effectiveness
- System health indicators

## Safety Features

- **Rate Limiting**: Multi-layer rate limiting for Twitter API compliance
- **Content Validation**: Automated content safety checks
- **Pattern Detection**: Spam behavior prevention
- **Backoff Strategies**: Exponential backoff on errors
- **Dry Run Mode**: Test without making actual API calls

## Testing

Run the test suite:

```bash
# All tests
poetry run pytest

# Unit tests only
poetry run pytest tests/test_agents.py

# Integration tests
poetry run pytest -m integration

# With coverage
poetry run pytest --cov=src
```

## Project Structure

```
src/
├── agents/           # AI agent implementations
│   ├── base_agent.py
│   ├── follow_agent.py
│   ├── content_agent.py
│   └── engagement_agent.py
├── config/           # Configuration management
│   └── settings.py
├── database/         # Database managers
│   ├── mongodb_manager.py
│   └── postgres_manager.py
├── models/           # Data models
│   └── data_models.py
├── utils/            # Utilities
│   ├── llm_client.py
│   ├── rate_limiter.py
│   └── monitoring.py
├── scheduler.py      # Main orchestrator
└── main.py          # Entry point

tests/               # Test suite
├── test_agents.py
├── test_database.py
├── test_utils.py
└── conftest.py

monitoring/          # Monitoring configuration
├── grafana/
│   ├── dashboards/
│   └── datasources/
└── prometheus.yml

scripts/             # Database initialization
├── mongo-init.js
└── init-postgres.sql
```

## Configuration

### Agent Thresholds

- **Follow Threshold**: 0.7 (70% relevance score required)
- **Like Threshold**: 0.6 (60% engagement score required)
- **Repost Threshold**: 0.8 (80% quality score required)
- **Comment Threshold**: 0.9 (90% relevance score required)

### Content Domains

- New Orleans culture and events
- Music scene and local artists
- GenZ trends and humor
- Community engagement

### Rate Limits (Conservative)

- **Daily**: 50 follows, 10 posts
- **Hourly**: 30 likes, 15 reposts, 10 comments

## Development

### Development Mode

Set `DEVELOPMENT_MODE=true` to:

- Reduce agent execution intervals (6x faster)
- Enable verbose logging
- Use lower rate limits

### Adding New Agents

1. Extend `BaseAgent` class
2. Implement required methods (`execute`, `get_agent_status`)
3. Add to scheduler configuration
4. Update monitoring dashboards

### Custom Content Themes

Modify `content_themes` in `ContentAgent` to add new content categories and inspiration prompts.

## Performance Optimization

### Database Optimization

- MongoDB indexes on frequently queried fields
- PostgreSQL pgvector indexes for similarity search
- Connection pooling for both databases
- Automated cleanup of old data

### Rate Limit Optimization

- Intelligent queuing with priority
- Exponential backoff on failures
- Multiple rate limit strategies
- Real-time limit monitoring

### Content Optimization

- Semantic caching of similar content
- Performance-based content selection
- A/B testing of content strategies

## Troubleshooting

### Common Issues

**Bot not starting**

- Check API credentials in `.env`
- Verify database connections
- Review logs for specific errors

**Low engagement rates**

- Adjust agent thresholds in configuration
- Review content generation prompts
- Check target audience relevance scores

**Rate limit errors**

- Reduce daily/hourly limits in configuration
- Check rate limiter status in monitoring
- Verify Twitter API plan limits

**Content quality issues**

- Adjust LLM temperature and prompts
- Review content validation rules
- Check sentiment analysis results

### Logs and Debugging

- Main logs: `logs/twitter_bot.log`
- Enable debug mode: `DEBUG=true`
- Dry run mode: `DRY_RUN=true`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Disclaimer

This bot system is designed for legitimate marketing and community engagement purposes. Users are responsible for:

- Complying with Twitter's Terms of Service
- Following applicable laws and regulations
- Using the system ethically and responsibly
- Monitoring bot behavior and content

## Support

For issues and questions:

- Check the troubleshooting guide above
- Review GitHub Issues
- Monitor system health dashboards
- Check agent execution logs

---

**Built for the New Orleans music community**
