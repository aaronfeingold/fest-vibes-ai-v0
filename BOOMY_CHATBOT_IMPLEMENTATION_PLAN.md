# Boomy RAG Chatbot Project - Complete Implementation Plan

## Project Overview
Building an authenticated MCP-powered chatbot agent that can query your NOLA events database intelligently, with proper shared library architecture for code reuse across multiple projects.

## Research Summary Completed

### Agent Architecture
- **MCP Tools Approach**: Converting existing RAG manager to authenticated MCP server tools
- **Pure Python**: Avoiding LangChain overhead, using existing robust database layer
- **Text-to-SQL**: LLM generates queries using full database schema awareness

### Deployment & Cost Analysis
- **AWS Lambda**: Cost-effective starting point ($10-50/month low traffic)
- **Scaling Path**: Lambda → EC2 when traffic > 1000 requests/day
- **LLM Hosting**: Hugging Face API more cost-effective than OpenAI for high volume
- **Infrastructure**: Message queue + pub/sub for streaming responses (avoid Vercel $20/mo plan)

### Current Assets to Leverage
- **RAG Manager**: Well-structured semantic search & route optimization (`src/database/rag_manager.py`)
- **LLM Client**: Rate limiting, multi-provider support (`src/utils/llm_client.py`) 
- **Boomy Prompt**: Comprehensive but needs token optimization (274 lines → ~150 lines)
- **DB Schema**: Available in ETL pipeline shared directory

## Proposed Architecture

### 1. Shared Library (`fest-vibes-ai-shared`)
**Create installable Python package with:**
- Database schema definitions & models
- RAG query operations & optimization algorithms
- Database connection utilities & configuration
- Common data types (EventSearchResult, OptimizedSchedule, etc.)
- LLM client abstraction layer

### 2. MCP Server (`fest-vibes-ai-chatbot/mcp-server`)
**Authenticated MCP server exposing tools:**
- `search_events_by_query`: Semantic event search
- `build_event_schedule`: Multi-venue route optimization  
- `get_events_by_timeframe`: Time-based filtering
- `query_database`: Text-to-SQL with schema validation
- Authentication middleware for secure database access

### 3. Boomy Chatbot Agent (`fest-vibes-ai-chatbot/agent`)
**Pure Python conversational agent:**
- Optimized system prompt (reduced token count)
- MCP tool integration
- Gen-Z authentic communication style
- Response streaming via pub/sub architecture

### 4. AWS Infrastructure
**Terraform-managed deployment:**
- Lambda functions for message processing
- SQS/SNS for pub/sub messaging
- API Gateway for client connections  
- Authenticated MCP server hosting

## Implementation Phases

### Phase 1: Shared Library Creation
- Extract reusable components from current project
- Package as installable library with proper setup.py
- Move DB schema from ETL shared to this library
- Create clean import structure

### Phase 2: MCP Server Development
- Build authenticated MCP server framework
- Convert RAG operations to MCP tools
- Implement text-to-SQL with full schema awareness
- Add security layers (read-only, query validation)

### Phase 3: Chatbot Agent Implementation
- Optimize Boomy system prompt for token efficiency
- Integrate MCP tools with conversational flow
- Implement response streaming architecture
- Add user preference learning capabilities

### Phase 4: Infrastructure & Deployment
- Terraform AWS infrastructure
- Lambda + message queue deployment
- Monitoring, logging, and error handling
- Performance testing and optimization

## Key Benefits
- **Reusable Architecture**: Shared library used by ETL, chatbot, future projects
- **Cost Optimization**: AWS Lambda scaling + Hugging Face API pricing
- **Clean Separation**: Database ops via authenticated MCP, agent logic separate
- **Schema Consistency**: Single source of truth for database structure
- **Production Ready**: Proper authentication, monitoring, scalable infrastructure

## Next Steps
- Navigate to parent directory (`/home/aaronfeingold/Code/ajf/fest-vibes-ai/`)
- Examine ETL shared directory structure
- Design shared library package structure
- Begin extraction and refactoring process

This plan leverages all your existing work while building toward a production-ready, cost-effective chatbot architecture that can scale with your needs.