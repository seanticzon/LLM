"""
INCIDENT AI WITH RAG - ENHANCED COMPETITION EDITION
With Conversation-Aware System to Fix Repetition
PLUS: Comprehensive Analytics API for External Dashboards
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
import httpx
import time
import json
from typing import Optional, Dict, Any, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG system
try:
    from rag_system import IncidentRAG, seed_example_data
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("⚠️  RAG system not available.")

app = FastAPI(
    title="Incident AI - Competition Edition",
    description="🏆 Advanced AI-powered incident analysis with RAG and auto-tuning",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"

# 🔥 COMPETITION TIP: Upgrade to better model for 2x better responses!
# Uncomment one of these for better quality:
# MODEL_NAME = "qwen2.5:7b"         # ⭐⭐⭐⭐⭐ BEST - Great reasoning, less repetitive
# MODEL_NAME = "llama3.1:8b"        # ⭐⭐⭐⭐ Good all-rounder
# MODEL_NAME = "mistral:7b"         # ⭐⭐⭐⭐ Fast and smart

MODEL_NAME = "llama3.2:3b-instruct-q8_0"  # Current model (change this!)

# Analytics data storage
analytics_data = {
    "total_analyses": 0,
    "successful_analyses": 0,
    "failed_analyses": 0,
    "response_times": [],
    "incidents_by_service": defaultdict(int),
    "incidents_by_hour": defaultdict(int),
    "confidence_scores": [],
    "rag_matches": 0,
    "rag_no_matches": 0,
    "tools_used": defaultdict(int),
    "agentic_calls": 0,
    "quality_scores": [],
    "last_updated": datetime.now().isoformat()
}

# ============================================================================
# CONVERSATION MANAGER - FIXES REPETITION
# ============================================================================

class ConversationManager:
    """Manages conversation history to prevent repetitive responses"""
    
    def __init__(self):
        self.conversations = {}  # incident_id -> conversation history
        self.max_history = 10
        logger.info("💬 Conversation Manager initialized")
    
    def add_message(self, incident_id: str, role: str, content: str):
        """Add a message to conversation history"""
        if incident_id not in self.conversations:
            self.conversations[incident_id] = []
        
        self.conversations[incident_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.conversations[incident_id]) > self.max_history:
            self.conversations[incident_id] = self.conversations[incident_id][-self.max_history:]
    
    def get_history(self, incident_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.conversations.get(incident_id, [])
    
    def clear_history(self, incident_id: str):
        """Clear conversation history"""
        if incident_id in self.conversations:
            del self.conversations[incident_id]

# ============================================================================
# ADAPTIVE LLM CONFIGURATION SYSTEM
# ============================================================================

class AdaptiveLLMConfig:
    """Auto-tunes LLM parameters based on response quality"""
    
    def __init__(self):
        self.config = {
            "temperature": 0.75,
            "top_p": 0.92,
            "top_k": 40,
            "repeat_penalty": 1.2,
            "num_predict": 600,
            "num_ctx": 4096,
            "stop": ["---", "\n\n\n", "Similar Past Incidents:"]
        }
        self.response_history = deque(maxlen=20)
        self.total_adjustments = 0
        logger.info("🤖 Adaptive LLM Config initialized")
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
    
    def analyze_quality(self, text: str) -> float:
        if len(text) < 30:
            return 0.3
        
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        word_count = len(words)
        length_score = 1.0 if 80 <= word_count <= 400 else 0.7
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        structure_score = 1.0 if len(sentences) >= 3 else 0.6
        
        repetitive_phrases = ["root cause assessment", "immediate action steps", "prevention measure"]
        phrase_count = sum(text.lower().count(phrase) for phrase in repetitive_phrases)
        repetition_penalty = 1.0 if phrase_count <= 1 else 0.8
        
        quality = (unique_ratio * 0.4 + length_score * 0.25 + structure_score * 0.25 + repetition_penalty * 0.1)
        return min(1.0, quality)
    
    def update(self, response_text: str) -> Dict[str, Any]:
        quality = self.analyze_quality(response_text)
        
        self.response_history.append({
            "timestamp": datetime.now().isoformat(),
            "quality": quality,
            "config": self.config.copy()
        })
        
        adjustments = {}
        
        if quality < 0.6:
            old_penalty = self.config["repeat_penalty"]
            old_temp = self.config["temperature"]
            
            self.config["repeat_penalty"] = min(1.3, old_penalty + 0.05)
            self.config["temperature"] = min(0.85, old_temp + 0.05)
            
            adjustments["repeat_penalty"] = f"{old_penalty:.2f} → {self.config['repeat_penalty']:.2f}"
            adjustments["temperature"] = f"{old_temp:.2f} → {self.config['temperature']:.2f}"
            self.total_adjustments += 1
            logger.warning(f"⚠️ Low quality ({quality:.2f}), adjusted parameters")
        
        return {
            "quality": quality,
            "adjustments": adjustments,
            "config": self.config
        }
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.response_history:
            return {
                "total_responses": 0,
                "average_quality": 0.0,
                "total_adjustments": self.total_adjustments,
                "current_config": self.config
            }
        
        qualities = [r["quality"] for r in self.response_history]
        return {
            "total_responses": len(self.response_history),
            "average_quality": round(sum(qualities) / len(qualities), 2),
            "min_quality": round(min(qualities), 2),
            "max_quality": round(max(qualities), 2),
            "total_adjustments": self.total_adjustments,
            "current_config": self.config,
            "trend": "improving" if len(qualities) > 5 and qualities[-1] > qualities[0] else "stable"
        }

# ============================================================================
# CONTEXT GATHERING SYSTEM
# ============================================================================

class ContextGatherer:
    """Gathers context for better AI analysis"""
    
    async def gather(self, title: str, description: str, service: str, similar_incidents: List[Dict]) -> Dict[str, Any]:
        return {
            "rag_context": similar_incidents,
            "service_info": self._get_service_info(service),
            "time_context": self._get_time_context(),
            "keywords": self._extract_keywords(title, description)
        }
    
    def _get_service_info(self, service: str) -> Dict[str, Any]:
        services = {
            "api-gateway": {
                "replicas": 3,
                "cpu": "45%",
                "memory": "1.2GB",
                "common_issues": ["timeout", "rate limiting", "502 errors"]
            },
            "database": {
                "replicas": 2,
                "cpu": "65%",
                "memory": "8GB",
                "common_issues": ["slow queries", "connection pool", "deadlocks"]
            },
            "auth-service": {
                "replicas": 2,
                "cpu": "30%",
                "memory": "512MB",
                "common_issues": ["token expiry", "redis cache", "rate limiting"]
            },
            "payment-service": {
                "replicas": 4,
                "cpu": "55%",
                "memory": "2GB",
                "common_issues": ["timeout", "api errors", "webhook failures"]
            },
            "system": {
                "replicas": "N/A",
                "cpu": "N/A",
                "memory": "N/A",
                "common_issues": ["general system errors", "configuration issues"]
            }
        }
        return services.get(service.lower(), services["system"])
    
    def _get_time_context(self) -> Dict[str, Any]:
        now = datetime.now()
        hour = now.hour
        day = now.strftime("%A")
        
        return {
            "current_time": now.isoformat(),
            "hour": hour,
            "day": day,
            "is_deployment_window": hour in [9, 10, 14, 15],
            "is_business_hours": 9 <= hour <= 17,
            "risk_level": "high" if hour in [9, 10, 14, 15] else "normal"
        }
    
    def _extract_keywords(self, title: str, description: str) -> List[str]:
        text = f"{title} {description}".lower()
        terms = [
            "timeout", "503", "502", "500", "404", "memory", "cpu",
            "disk", "latency", "slow", "error", "crash", "restart",
            "database", "redis", "cache", "queue", "deployment",
            "connection", "pool", "exhausted", "high load"
        ]
        return [term for term in terms if term in text]

# ============================================================================
# INITIALIZE SYSTEMS
# ============================================================================

rag = None
if RAG_AVAILABLE:
    try:
        rag = IncidentRAG()
        logger.info(f"✅ RAG initialized with {rag.count_incidents()} incidents")
    except Exception as e:
        logger.error(f"⚠️  Could not initialize RAG: {e}")

adaptive_config = AdaptiveLLMConfig()
context_gatherer = ContextGatherer()
conversation_manager = ConversationManager()

# Webhook subscribers for real-time updates
webhook_subscribers = []

# ============================================================================
# CONVERSATION-AWARE PROMPT BUILDER
# ============================================================================

def build_conversation_aware_prompt(
    title: str,
    description: str,
    service: str,
    context: Dict[str, Any],
    user_question: str,
    conversation_history: List[Dict]
) -> str:
    """Build prompts that understand conversation context - PREVENTS REPETITION"""
    
    similar = context["rag_context"]
    
    # Track what was already mentioned
    already_mentioned = {
        "commands": set(),
        "topics": set(),
        "incident_ids": set()
    }
    
    for msg in conversation_history:
        if msg["role"] == "assistant":
            content_lower = msg["content"].lower()
            
            # Track commands
            if "pg_stat_activity" in content_lower:
                already_mentioned["commands"].add("pg_stat_activity")
            if "pg_settings" in content_lower:
                already_mentioned["commands"].add("pg_settings")
            if "ps aux" in content_lower:
                already_mentioned["commands"].add("ps aux")
            
            # Track topics
            if "root cause" in content_lower:
                already_mentioned["topics"].add("root_cause")
            if "prevention" in content_lower:
                already_mentioned["topics"].add("prevention")
            
            # Track mentioned incidents
            for inc in similar:
                if inc["incident_id"].lower() in content_lower:
                    already_mentioned["incident_ids"].add(inc["incident_id"])
    
    # Analyze user question
    question_lower = user_question.lower()
    
    asking_about_similar = any(word in question_lower for word in 
                               ["similar", "past", "previous", "related", "other", "cases", "guide"])
    
    asking_without_similar = any(phrase in question_lower for phrase in 
                                 ["without", "instead of", "other than", "different", "alternative"])
    
    # Build context-aware response
    if asking_without_similar and similar:
        # User wants alternative approaches (not using similar incident)
        best = similar[0]
        prompt = f"""You're helping with: {title}

The user asked: "{user_question}"

**Important:** The user explicitly wants to solve this WITHOUT using the similar incident approach.

**Previous conversation:**
"""
        for msg in conversation_history[-4:]:
            role = "User" if msg["role"] == "user" else "You"
            prompt += f"{role}: {msg['content'][:120]}...\n"
        
        prompt += f"""

**Commands already suggested (DON'T repeat these):**
{', '.join(already_mentioned['commands']) if already_mentioned['commands'] else 'None yet'}

**Provide COMPLETELY DIFFERENT troubleshooting:**

1. **Alternative diagnostic** (different from what you said before):
   - Check connection pooler logs: `tail -f /var/log/pgbouncer/pgbouncer.log`
   - Analyze network latency: `ping -c 5 database-host`
   - Look for app-level leaks: `lsof -p <app_pid> | grep TCP`

2. **Different root cause angle**:
   - Could be a sudden traffic spike (DDoS or viral event)
   - Memory leak in app keeping connections open
   - Slow queries blocking the pool

3. **Quick fixes**:
   - Restart connection pooler for immediate relief
   - Kill long-running transactions: `SELECT pg_terminate_backend(pid)`
   - Enable statement timeout if not set
   - Scale horizontally with read replicas

Keep under 150 words. Be specific."""

    elif asking_about_similar and similar:
        # User is asking about the similar incident
        best = similar[0]
        prompt = f"""User asked: "{user_question}"

**Found: Incident {best['incident_id']} (73% match)**

What happened: {best['title']}
Root cause: {best['root_cause']}
How we fixed it: {best['resolution']}

**Answer the user's question about this similar case:**

1. Confirm YES, there IS a similar case ({best['incident_id']})
2. Explain what happened in that incident
3. Describe specifically how it was solved
4. Point out what parts apply to current situation
5. Mention differences to watch for

Start with: "Yes, we had incident {best['incident_id']} which is very similar..."

Be conversational. Under 150 words."""

    else:
        # General follow-up
        prompt = f"""Conversation about: {title}

**Recent chat:**
"""
        for msg in conversation_history[-4:]:
            role = "User" if msg["role"] == "user" else "You"
            prompt += f"{role}: {msg['content'][:100]}...\n"
        
        prompt += f"""

User asked: "{user_question}"

**Already covered (don't repeat):**
- Commands: {', '.join(already_mentioned['commands']) if already_mentioned['commands'] else 'none'}
- Topics: {', '.join(already_mentioned['topics']) if already_mentioned['topics'] else 'none'}

Answer directly. Reference what was discussed. Add NEW information.

Under 120 words."""
    
    return prompt

def build_enhanced_prompt(
    title: str, 
    description: str, 
    service: str, 
    context: Dict[str, Any],
    user_question: Optional[str] = None
) -> str:
    """Enhanced prompt for first message"""
    
    similar = context["rag_context"]
    
    prompt = f"""You're an expert SRE assistant analyzing this incident.

**Incident:**
Service: {service}
Title: {title}
Description: {description}

"""
    
    if similar:
        best = similar[0]
        sim_pct = int(best['similarity_score'] * 100)
        
        if sim_pct >= 75:
            prompt += f"""**Found incident {best['incident_id']} ({sim_pct}% match):**
Cause: {best['root_cause']}
Fix: {best['resolution']}

Start with: "This looks very similar to incident {best['incident_id']} we had before."
"""
        elif sim_pct >= 60:
            prompt += f"""**Found incident {best['incident_id']} ({sim_pct}% similar):**
Cause: {best['root_cause']}
Fix: {best['resolution']}

Start with: "I found a related incident ({best['incident_id']})."
"""
        else:
            prompt += f"""**Past incidents only {sim_pct}% similar.**

Start with: "I found some loosely related incidents, but this looks like a new pattern."
"""
    else:
        prompt += """**No similar past incidents.**

Start with: "This appears to be a new type of incident."
"""
    
    service_info = context.get("service_info", {})
    if service_info.get("common_issues"):
        prompt += f"\nNote: {service} often has {', '.join(service_info['common_issues'][:2])}.\n"
    
    keywords = context.get("keywords", [])
    if keywords:
        prompt += f"Detected: {', '.join(keywords[:4])}\n"
    
    prompt += """
Give:
1. Quick diagnosis (2 sentences)
2. Immediate steps (3-4 specific commands)
3. Prevention tip (1 sentence)

Conversational style. Under 200 words."""
    
    return prompt

# ============================================================================
# API ENDPOINTS - HOME & HEALTH
# ============================================================================

@app.get("/")
def home():
    """Home endpoint"""
    incident_count = rag.count_incidents() if rag else 0
    
    ollama_status = "unknown"
    qdrant_status = "unknown"
    
    try:
        import requests
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        ollama_status = "✅ running" if response.status_code == 200 else "❌ error"
    except:
        ollama_status = "❌ not running"
    
    try:
        import requests
        response = requests.get(f"{QDRANT_URL}/collections", timeout=2)
        qdrant_status = "✅ running" if response.status_code == 200 else "❌ error"
    except:
        qdrant_status = "❌ not running"
    
    return {
        "status": "🚀 Incident AI - Competition Edition",
        "version": "2.0 - Conversation-Aware + Enhanced Analytics",
        "model": MODEL_NAME,
        "services": {
            "ollama": ollama_status,
            "qdrant": qdrant_status,
            "rag": "✅ enabled" if rag else "❌ disabled"
        },
        "knowledge_base": {
            "incidents": incident_count,
            "auto_learning": True
        },
        "dashboards": {
            "api_docs": "http://localhost:8000/docs",
            "qdrant_ui": f"{QDRANT_URL}/dashboard",
            "health": "http://localhost:8000/health"
        },
        "endpoints": {
            "analysis": ["/analyze", "/analyze/stream", "/analyze/agentic-stream"],
            "incidents": ["/incidents/add", "/incidents/search", "/incidents/list"],
            "analytics": [
                "/api/analytics/overview", "/api/analytics/performance",
                "/api/analytics/quality", "/api/analytics/export"
            ],
            "config": ["/config/current", "/config/reset"],
            "conversation": ["/conversation/{incident_id}"]
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            health_status["services"]["ollama"] = {
                "status": "✅ running" if response.status_code == 200 else "❌ error",
                "url": OLLAMA_URL
            }
    except Exception as e:
        health_status["services"]["ollama"] = {
            "status": "❌ not reachable",
            "url": OLLAMA_URL
        }
        health_status["status"] = "degraded"
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL)
        collections = client.get_collections()
        health_status["services"]["qdrant"] = {
            "status": "✅ running",
            "url": QDRANT_URL,
            "dashboard": f"{QDRANT_URL}/dashboard",
            "collections": len(collections.collections)
        }
    except Exception as e:
        health_status["services"]["qdrant"] = {
            "status": "❌ not reachable",
            "url": QDRANT_URL
        }
        health_status["status"] = "degraded"
    
    health_status["services"]["rag"] = {
        "status": "✅ enabled" if rag else "❌ disabled",
        "incidents_count": rag.count_incidents() if rag else 0
    }
    
    return health_status

@app.get("/qdrant")
async def qdrant_redirect():
    """Redirect to Qdrant dashboard"""
    return RedirectResponse(url=f"{QDRANT_URL}/dashboard")

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/analyze")
async def analyze_basic(incident: dict):
    """Basic non-streaming analysis"""
    title = incident.get('title', '')
    description = incident.get('description', '')
    service = incident.get('service', 'unknown')
    
    if not title or not description:
        raise HTTPException(status_code=400, detail="Missing title or description")
    
    start_time = time.time()
    analytics_data["total_analyses"] += 1
    
    similar_incidents = []
    if rag:
        try:
            similar_incidents = rag.search_similar(
                query=f"{title}. {description}",
                service=service,
                limit=3
            )
            if similar_incidents:
                analytics_data["rag_matches"] += 1
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
    
    context = await context_gatherer.gather(title, description, service, similar_incidents)
    prompt = build_enhanced_prompt(title, description, service, context)
    llm_config = adaptive_config.get_config()
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": llm_config
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get('response', '').strip()
                elapsed = time.time() - start_time
                
                quality_report = adaptive_config.update(analysis_text)
                
                analytics_data["successful_analyses"] += 1
                analytics_data["response_times"].append(elapsed)
                analytics_data["quality_scores"].append(quality_report["quality"])
                
                return {
                    "incident_title": title,
                    "service": service,
                    "analysis": analysis_text,
                    "quality_score": quality_report["quality"],
                    "similar_incidents": [
                        {
                            "id": inc["incident_id"],
                            "similarity": inc["similarity_score"],
                            "title": inc["title"]
                        }
                        for inc in similar_incidents[:3]
                    ],
                    "response_time": round(elapsed, 2),
                    "mode": "basic"
                }
            
            raise HTTPException(status_code=response.status_code, detail="AI service error")
    
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama")
    except Exception as e:
        analytics_data["failed_analyses"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/stream")
async def analyze_stream(incident: dict):
    """Streaming analysis"""
    title = incident.get('title', '')
    description = incident.get('description', '')
    service = incident.get('service', 'unknown')
    
    if not title or not description:
        async def error_gen():
            yield f"data: {json.dumps({'error': 'Missing title or description'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    
    async def generate_stream():
        try:
            start_time = time.time()
            analytics_data["total_analyses"] += 1
            
            similar_incidents = []
            if rag:
                try:
                    similar_incidents = rag.search_similar(
                        query=f"{title}. {description}",
                        service=service,
                        limit=3
                    )
                except Exception as e:
                    logger.error(f"RAG search: {e}")
            
            context = await context_gatherer.gather(title, description, service, similar_incidents)
            prompt = build_enhanced_prompt(title, description, service, context)
            llm_config = adaptive_config.get_config()
            
            metadata = {
                "type": "metadata",
                "incident": title,
                "service": service,
                "similar_count": len(similar_incidents)
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            full_response = ""
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": True,
                        "options": llm_config
                    }
                ) as response:
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if token := data.get('response'):
                                    full_response += token
                                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                                    await asyncio.sleep(0.01)
                                
                                if data.get('done'):
                                    elapsed = time.time() - start_time
                                    quality_report = adaptive_config.update(full_response)
                                    
                                    analytics_data["successful_analyses"] += 1
                                    analytics_data["response_times"].append(elapsed)
                                    analytics_data["quality_scores"].append(quality_report["quality"])
                                    
                                    completion = {
                                        "type": "done",
                                        "response_time": round(elapsed, 2),
                                        "quality": quality_report["quality"]
                                    }
                                    yield f"data: {json.dumps(completion)}\n\n"
                                    break
                            
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/analyze/agentic-stream")
async def analyze_agentic_stream(incident: dict):
    """
    ENHANCED: Conversation-aware streaming - FIXES REPETITION
    """
    title = incident.get('title', '')
    description = incident.get('description', '')
    service = incident.get('service', 'unknown')
    severity = incident.get('severity', 'medium')
    incident_id = incident.get('incident_id', title)
    
    # Extract user question from description
    user_question = ""
    if "\n\nUser Question: " in description:
        parts = description.split("\n\nUser Question: ")
        description = parts[0]
        user_question = parts[1] if len(parts) > 1 else ""
    
    if not title:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Missing title'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    
    async def generate_agentic_stream():
        try:
            start_time = time.time()
            logger.info(f"🚀 Agentic analysis: {title}")
            
            analytics_data["total_analyses"] += 1
            analytics_data["agentic_calls"] += 1
            
            # Get conversation history
            conversation_history = conversation_manager.get_history(incident_id)
            
            # Add user question to history
            if user_question:
                conversation_manager.add_message(incident_id, "user", user_question)
                logger.info(f"📜 Conversation has {len(conversation_history)} messages")
            
            # Search
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing context...'})}\n\n"
            await asyncio.sleep(0.1)
            
            similar_incidents = []
            if rag:
                try:
                    similar_incidents = rag.search_similar(
                        query=f"{title}. {description}",
                        service=service,
                        limit=3
                    )
                    if similar_incidents:
                        analytics_data["rag_matches"] += 1
                except Exception as e:
                    logger.error(f"RAG error: {e}")
            
            context = await context_gatherer.gather(title, description, service, similar_incidents)
            
            # Build conversation-aware prompt
            if user_question and conversation_history:
                logger.info(f"💬 Using conversation-aware prompt")
                prompt = build_conversation_aware_prompt(
                    title, description, service, context,
                    user_question, conversation_history
                )
            else:
                prompt = build_enhanced_prompt(title, description, service, context, user_question)
            
            llm_config = adaptive_config.get_config()
            
            confidence = 0.5
            if similar_incidents and similar_incidents[0]["similarity_score"] > 0.8:
                confidence = 0.9
            elif similar_incidents and similar_incidents[0]["similarity_score"] > 0.6:
                confidence = 0.75
            
            metadata = {
                "type": "metadata",
                "incident_title": title,
                "service": service,
                "confidence": confidence,
                "conversation_length": len(conversation_history),
                "context": {
                    "similar_incidents": len(similar_incidents),
                    "keywords": context.get("keywords", [])
                },
                "similar_past_incidents": [
                    {
                        "id": inc["incident_id"],
                        "similarity": round(inc["similarity_score"], 2),
                        "title": inc["title"],
                        "root_cause": inc.get("root_cause", "N/A")[:100],
                        "resolution": inc.get("resolution", "N/A")[:100]
                    }
                    for inc in similar_incidents[:2]
                ]
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            await asyncio.sleep(0.1)
            
            full_response = ""
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": True,
                        "options": llm_config
                    }
                ) as response:
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if token := data.get('response'):
                                    full_response += token
                                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                                    await asyncio.sleep(0.01)
                                
                                if data.get('done'):
                                    elapsed = time.time() - start_time
                                    
                                    # Add assistant response to history
                                    conversation_manager.add_message(incident_id, "assistant", full_response)
                                    
                                    quality_report = adaptive_config.update(full_response)
                                    
                                    analytics_data["successful_analyses"] += 1
                                    analytics_data["response_times"].append(elapsed)
                                    analytics_data["quality_scores"].append(quality_report["quality"])
                                    
                                    completion = {
                                        "type": "done",
                                        "response_time": round(elapsed, 2),
                                        "quality_score": quality_report["quality"],
                                        "conversation_messages": len(conversation_history) + 2,
                                        "adjustments": quality_report["adjustments"],
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    yield f"data: {json.dumps(completion)}\n\n"
                                    break
                            
                            except json.JSONDecodeError:
                                continue
        
        except httpx.ConnectError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Cannot connect to Ollama'})}\n\n"
        except Exception as e:
            logger.error(f"Agentic stream error: {e}")
            analytics_data["failed_analyses"] += 1
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_agentic_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================================
# CONVERSATION ENDPOINTS
# ============================================================================

@app.get("/conversation/{incident_id}")
async def get_conversation(incident_id: str):
    """Get conversation history"""
    history = conversation_manager.get_history(incident_id)
    return {
        "incident_id": incident_id,
        "message_count": len(history),
        "history": history
    }

@app.delete("/conversation/{incident_id}")
async def clear_conversation(incident_id: str):
    """Clear conversation history"""
    conversation_manager.clear_history(incident_id)
    return {
        "message": "✅ Conversation history cleared",
        "incident_id": incident_id
    }

# ============================================================================
# ENHANCED ANALYTICS API - FOR EXTERNAL DASHBOARDS
# ============================================================================

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """
    OVERVIEW: High-level metrics for dashboard cards
    Perfect for displaying key metrics at a glance
    """
    total = analytics_data["total_analyses"]
    avg_time = sum(analytics_data["response_times"]) / len(analytics_data["response_times"]) if analytics_data["response_times"] else 0
    success_rate = (analytics_data["successful_analyses"] / total * 100) if total > 0 else 0
    avg_quality = sum(analytics_data["quality_scores"]) / len(analytics_data["quality_scores"]) if analytics_data["quality_scores"] else 0
    
    return {
        "total_analyses": total,
        "successful_analyses": analytics_data["successful_analyses"],
        "failed_analyses": analytics_data["failed_analyses"],
        "success_rate": round(success_rate, 1),
        "average_response_time": round(avg_time, 2),
        "average_quality_score": round(avg_quality, 2),
        "knowledge_base_size": rag.count_incidents() if rag else 0,
        "rag_match_rate": round(analytics_data["rag_matches"] / total * 100, 1) if total > 0 else 0,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/analytics/performance")
async def get_performance_metrics():
    """
    PERFORMANCE: Response time statistics
    Great for performance monitoring graphs
    """
    times = analytics_data["response_times"]
    
    if not times:
        return {
            "average": 0,
            "min": 0,
            "max": 0,
            "median": 0,
            "p95": 0,
            "p99": 0,
            "total_requests": 0,
            "response_times": []
        }
    
    sorted_times = sorted(times)
    n = len(sorted_times)
    
    return {
        "average": round(sum(times) / n, 2),
        "min": round(min(times), 2),
        "max": round(max(times), 2),
        "median": round(sorted_times[n // 2], 2),
        "p95": round(sorted_times[int(n * 0.95)], 2) if n > 0 else 0,
        "p99": round(sorted_times[int(n * 0.99)], 2) if n > 0 else 0,
        "total_requests": n,
        "response_times": [round(t, 2) for t in times[-50:]]
    }

@app.get("/api/analytics/quality")
async def get_quality_metrics():
    """
    QUALITY: AI response quality tracking
    Shows how well the AI is performing
    """
    scores = analytics_data["quality_scores"]
    
    if not scores:
        return {
            "average": 0,
            "min": 0,
            "max": 0,
            "trend": "stable",
            "total_scored": 0,
            "excellent_count": 0,
            "good_count": 0,
            "poor_count": 0,
            "quality_scores": []
        }
    
    excellent = sum(1 for s in scores if s >= 0.8)
    good = sum(1 for s in scores if 0.6 <= s < 0.8)
    poor = sum(1 for s in scores if s < 0.6)
    
    # Calculate trend
    trend = "stable"
    if len(scores) >= 20:
        recent_avg = sum(scores[-10:]) / 10
        previous_avg = sum(scores[-20:-10]) / 10
        if recent_avg > previous_avg + 0.05:
            trend = "improving"
        elif recent_avg < previous_avg - 0.05:
            trend = "declining"
    
    return {
        "average": round(sum(scores) / len(scores), 2),
        "min": round(min(scores), 2),
        "max": round(max(scores), 2),
        "trend": trend,
        "total_scored": len(scores),
        "excellent_count": excellent,
        "good_count": good,
        "poor_count": poor,
        "quality_scores": [round(s, 2) for s in scores[-50:]]
    }

@app.get("/api/analytics/rag")
async def get_rag_metrics():
    """
    RAG INTELLIGENCE: How well RAG is performing
    Shows knowledge base effectiveness
    """
    total = analytics_data["total_analyses"]
    match_rate = (analytics_data["rag_matches"] / total * 100) if total > 0 else 0
    
    return {
        "knowledge_base_size": rag.count_incidents() if rag else 0,
        "total_queries": total,
        "matches_found": analytics_data["rag_matches"],
        "no_matches": analytics_data["rag_no_matches"],
        "match_rate": round(match_rate, 1),
        "is_enabled": rag is not None,
        "confidence_scores": [round(s, 2) for s in analytics_data["confidence_scores"][-50:]]
    }

@app.get("/api/analytics/patterns")
async def get_pattern_analysis():
    """
    PATTERNS: Incident patterns by service and time
    Useful for identifying trends
    """
    top_services = sorted(
        analytics_data["incidents_by_service"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    hourly_data = dict(analytics_data["incidents_by_hour"])
    peak_hour = max(hourly_data.items(), key=lambda x: x[1])[0] if hourly_data else 0
    
    return {
        "by_service": dict(top_services),
        "by_hour": hourly_data,
        "peak_hour": peak_hour,
        "total_services": len(analytics_data["incidents_by_service"]),
        "busiest_service": top_services[0][0] if top_services else "none",
        "total_incidents": sum(analytics_data["incidents_by_service"].values())
    }

@app.get("/api/analytics/health")
async def get_system_health():
    """
    HEALTH CHECK: System component status
    Shows if all services are running properly
    """
    health_status = {
        "overall": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            health_status["components"]["ollama"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "reachable": True,
                "url": OLLAMA_URL
            }
    except Exception as e:
        health_status["components"]["ollama"] = {
            "status": "unhealthy",
            "reachable": False,
            "url": OLLAMA_URL,
            "error": str(e)
        }
        health_status["overall"] = "degraded"
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL)
        collections = client.get_collections()
        health_status["components"]["qdrant"] = {
            "status": "healthy",
            "reachable": True,
            "url": QDRANT_URL,
            "collections_count": len(collections.collections)
        }
    except Exception as e:
        health_status["components"]["qdrant"] = {
            "status": "unhealthy",
            "reachable": False,
            "url": QDRANT_URL,
            "error": str(e)
        }
        health_status["overall"] = "degraded"
    
    # Check RAG
    health_status["components"]["rag"] = {
        "status": "healthy" if rag else "disabled",
        "enabled": rag is not None,
        "incidents_count": rag.count_incidents() if rag else 0
    }
    
    return health_status

@app.get("/api/analytics/conversations")
async def get_conversation_metrics():
    """
    CONVERSATIONS: Track conversation patterns
    Shows how users are interacting with the AI
    """
    total_conversations = len(conversation_manager.conversations)
    
    avg_length = 0
    if total_conversations > 0:
        lengths = [len(conv) for conv in conversation_manager.conversations.values()]
        avg_length = sum(lengths) / len(lengths)
    
    active_conversations = sum(
        1 for conv in conversation_manager.conversations.values() 
        if len(conv) > 0
    )
    
    return {
        "total_conversations": total_conversations,
        "active_conversations": active_conversations,
        "average_conversation_length": round(avg_length, 1),
        "max_history_size": conversation_manager.max_history,
        "anti_repetition_enabled": True
    }

@app.get("/api/analytics/realtime")
async def get_realtime_stats():
    """
    REALTIME: Live statistics for dashboards
    Updates frequently for real-time monitoring
    """
    recent_times = analytics_data["response_times"][-10:] if analytics_data["response_times"] else []
    recent_avg = sum(recent_times) / len(recent_times) if recent_times else 0
    
    recent_quality = analytics_data["quality_scores"][-10:] if analytics_data["quality_scores"] else []
    recent_quality_avg = sum(recent_quality) / len(recent_quality) if recent_quality else 0
    
    return {
        "current_timestamp": datetime.now().isoformat(),
        "recent_performance": {
            "last_10_avg_response_time": round(recent_avg, 2),
            "last_10_avg_quality": round(recent_quality_avg, 2)
        },
        "status": {
            "system": "operational",
            "ai_model": MODEL_NAME,
            "requests_today": analytics_data["total_analyses"]
        },
        "live_metrics": {
            "success_rate": round(
                (analytics_data["successful_analyses"] / analytics_data["total_analyses"] * 100) 
                if analytics_data["total_analyses"] > 0 else 0,
                1
            ),
            "rag_enabled": rag is not None,
            "adaptive_learning": True
        }
    }

@app.get("/api/analytics/export")
async def export_all_analytics():
    """
    EXPORT: Complete analytics dump
    Get everything in one call - perfect for external dashboards
    """
    total = analytics_data["total_analyses"]
    avg_time = sum(analytics_data["response_times"]) / len(analytics_data["response_times"]) if analytics_data["response_times"] else 0
    success_rate = (analytics_data["successful_analyses"] / total * 100) if total > 0 else 0
    avg_quality = sum(analytics_data["quality_scores"]) / len(analytics_data["quality_scores"]) if analytics_data["quality_scores"] else 0
    
    return {
        "export_timestamp": datetime.now().isoformat(),
        "overview": {
            "total_analyses": total,
            "successful": analytics_data["successful_analyses"],
            "failed": analytics_data["failed_analyses"],
            "success_rate": round(success_rate, 1),
            "avg_quality": round(avg_quality, 2)
        },
        "performance": {
            "avg_response_time": round(avg_time, 2),
            "response_times": [round(t, 2) for t in analytics_data["response_times"][-100:]]
        },
        "quality": {
            "scores": [round(s, 2) for s in analytics_data["quality_scores"][-100:]],
            "average": round(avg_quality, 2)
        },
        "rag": {
            "knowledge_base_size": rag.count_incidents() if rag else 0,
            "matches": analytics_data["rag_matches"],
            "no_matches": analytics_data["rag_no_matches"]
        },
        "patterns": {
            "by_service": dict(analytics_data["incidents_by_service"]),
            "by_hour": dict(analytics_data["incidents_by_hour"])
        },
        "adaptive": adaptive_config.get_stats(),
        "conversations": {
            "total": len(conversation_manager.conversations),
            "max_history": conversation_manager.max_history
        },
        "model": {
            "name": MODEL_NAME,
            "features": ["RAG", "Streaming", "Adaptive Config", "Anti-Repetition"]
        }
    }

# ============================================================================
# WEBHOOK SUPPORT - FOR REAL-TIME UPDATES
# ============================================================================

@app.post("/api/analytics/webhook/subscribe")
async def subscribe_to_analytics_webhook(webhook: dict):
    """
    WEBHOOK: Subscribe to real-time analytics updates
    Your external app can register a webhook URL to receive updates
    """
    url = webhook.get("url")
    events = webhook.get("events", ["all"])
    
    if not url:
        raise HTTPException(status_code=400, detail="Missing webhook URL")
    
    webhook_subscribers.append({
        "url": url,
        "events": events,
        "subscribed_at": datetime.now().isoformat()
    })
    
    return {
        "message": "✅ Webhook subscribed",
        "url": url,
        "events": events,
        "total_subscribers": len(webhook_subscribers)
    }

@app.get("/api/analytics/webhook/list")
async def list_webhook_subscribers():
    """List all webhook subscribers"""
    return {
        "subscribers": webhook_subscribers,
        "total": len(webhook_subscribers)
    }

async def notify_webhooks(event_type: str, data: dict):
    """
    Internal function to notify webhooks
    Call this after important events
    """
    for subscriber in webhook_subscribers:
        if "all" in subscriber["events"] or event_type in subscriber["events"]:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        subscriber["url"],
                        json={
                            "event": event_type,
                            "timestamp": datetime.now().isoformat(),
                            "data": data
                        }
                    )
            except Exception as e:
                logger.error(f"Webhook notification failed for {subscriber['url']}: {e}")

# ============================================================================
# INCIDENT MANAGEMENT
# ============================================================================

@app.post("/incidents/seed")
async def seed_incidents():
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        seed_example_data(rag)
        count = rag.count_incidents()
        return {
            "message": "✅ Incidents seeded successfully",
            "total": count,
            "qdrant_dashboard": f"{QDRANT_URL}/dashboard"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/incidents/add")
async def add_incident(incident: dict):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    required = ['incident_id', 'title', 'description', 'service']
    if not all(k in incident for k in required):
        raise HTTPException(status_code=400, detail=f"Missing required fields: {required}")
    
    try:
        doc_id = rag.add_incident(
            incident_id=incident['incident_id'],
            title=incident['title'],
            description=incident['description'],
            service=incident['service'],
            root_cause=incident.get('root_cause', ''),
            resolution=incident.get('resolution', ''),
            severity=incident.get('severity', 'medium'),
            tags=incident.get('tags', [])
        )
        return {
            "message": "✅ Incident added",
            "document_id": doc_id,
            "total_incidents": rag.count_incidents(),
            "view_in_qdrant": f"{QDRANT_URL}/dashboard"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents/search")
async def search_incidents(query: str, service: Optional[str] = None, limit: int = 5):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        results = rag.search_similar(query=query, service=service, limit=limit)
        return {
            "query": query,
            "service_filter": service,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents/count")
async def count_incidents():
    if not rag:
        return {"count": 0, "rag_available": False}
    return {
        "count": rag.count_incidents(),
        "rag_available": True,
        "qdrant_dashboard": f"{QDRANT_URL}/dashboard"
    }

@app.get("/incidents/list")
async def list_incidents(limit: int = 50, offset: int = 0):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL)
        
        result = client.scroll(
            collection_name="incidents",
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        points, next_offset = result
        
        incidents = [
            {
                "incident_id": p.payload.get('incident_id'),
                "title": p.payload.get('title'),
                "service": p.payload.get('service'),
                "severity": p.payload.get('severity', 'unknown'),
                "root_cause": p.payload.get('root_cause', 'N/A'),
                "resolution": p.payload.get('resolution', 'N/A')
            }
            for p in points
        ]
        
        return {
            "incidents": incidents,
            "count": len(incidents),
            "next_offset": next_offset,
            "has_more": next_offset is not None,
            "view_all_in_qdrant": f"{QDRANT_URL}/dashboard"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LEGACY ANALYTICS (KEEPING FOR COMPATIBILITY)
# ============================================================================

@app.get("/analytics/dashboard")
async def analytics_dashboard():
    total = analytics_data["total_analyses"]
    avg_time = sum(analytics_data["response_times"]) / len(analytics_data["response_times"]) if analytics_data["response_times"] else 0
    success_rate = (analytics_data["successful_analyses"] / total * 100) if total > 0 else 0
    avg_quality = sum(analytics_data["quality_scores"]) / len(analytics_data["quality_scores"]) if analytics_data["quality_scores"] else 0
    
    return {
        "overview": {
            "total_analyses": total,
            "successful": analytics_data["successful_analyses"],
            "failed": analytics_data["failed_analyses"],
            "success_rate": round(success_rate, 1),
            "average_quality": round(avg_quality, 2)
        },
        "performance": {
            "avg_response_time": round(avg_time, 2),
            "fastest": round(min(analytics_data["response_times"]), 2) if analytics_data["response_times"] else 0,
            "slowest": round(max(analytics_data["response_times"]), 2) if analytics_data["response_times"] else 0
        },
        "intelligence": {
            "knowledge_base_size": rag.count_incidents() if rag else 0,
            "rag_matches": analytics_data["rag_matches"],
            "rag_no_matches": analytics_data["rag_no_matches"],
            "match_rate": round(analytics_data["rag_matches"] / total * 100, 1) if total > 0 else 0
        },
        "patterns": {
            "by_service": dict(analytics_data["incidents_by_service"]),
            "by_hour": dict(analytics_data["incidents_by_hour"])
        },
        "last_updated": analytics_data["last_updated"]
    }

@app.get("/analytics/adaptive")
async def analytics_adaptive():
    return {
        "adaptive_system": adaptive_config.get_stats(),
        "auto_tuning": "enabled"
    }

# ============================================================================
# CONFIGURATION
# ============================================================================

@app.get("/config/current")
async def get_config():
    return {
        "llm_config": adaptive_config.get_config(),
        "model": MODEL_NAME,
        "adaptive_tuning": True
    }

@app.post("/config/reset")
async def reset_config():
    global adaptive_config
    adaptive_config = AdaptiveLLMConfig()
    return {
        "message": "✅ Configuration reset",
        "config": adaptive_config.get_config()
    }

# ============================================================================
# TEST DATA GENERATOR
# ============================================================================

@app.post("/test/generate-analytics")
async def generate_test_analytics():
    """Generate realistic test analytics data for demo"""
    import random
    
    for i in range(50):
        analytics_data["total_analyses"] += 1
        
        if random.random() < 0.9:
            analytics_data["successful_analyses"] += 1
            analytics_data["response_times"].append(random.uniform(1.5, 4.5))
            analytics_data["quality_scores"].append(random.uniform(0.6, 0.95))
        else:
            analytics_data["failed_analyses"] += 1
        
        service = random.choice(["database", "api-gateway", "auth-service", "payment-service"])
        analytics_data["incidents_by_service"][service] += 1
        
        hour = random.randint(8, 18)
        analytics_data["incidents_by_hour"][hour] += 1
        
        if random.random() < 0.7:
            analytics_data["rag_matches"] += 1
            analytics_data["confidence_scores"].append(random.uniform(0.65, 0.95))
        else:
            analytics_data["rag_no_matches"] += 1
    
    analytics_data["agentic_calls"] = int(analytics_data["total_analyses"] * 0.8)
    analytics_data["last_updated"] = datetime.now().isoformat()
    
    for _ in range(20):
        adaptive_config.update(f"This is a test response with quality score simulation. " * random.randint(10, 30))
    
    return {
        "message": "✅ Test analytics data generated",
        "stats": {
            "total_analyses": analytics_data["total_analyses"],
            "successful": analytics_data["successful_analyses"],
            "failed": analytics_data["failed_analyses"],
            "rag_matches": analytics_data["rag_matches"]
        },
        "note": "Refresh your dashboard to see the data"
    }

@app.post("/test/reset-analytics")
async def reset_test_analytics():
    """Reset analytics data to zero"""
    global analytics_data, adaptive_config
    
    analytics_data = {
        "total_analyses": 0,
        "successful_analyses": 0,
        "failed_analyses": 0,
        "response_times": [],
        "incidents_by_service": defaultdict(int),
        "incidents_by_hour": defaultdict(int),
        "confidence_scores": [],
        "rag_matches": 0,
        "rag_no_matches": 0,
        "tools_used": defaultdict(int),
        "agentic_calls": 0,
        "quality_scores": [],
        "last_updated": datetime.now().isoformat()
    }
    
    adaptive_config = AdaptiveLLMConfig()
    
    return {
        "message": "✅ Analytics data reset",
        "note": "All metrics are now at zero"
    }

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🏆 INCIDENT AI - ENHANCED EDITION")
    print("="*70)
    print(f"\n🤖 Model: {MODEL_NAME}")
    print(f"🔍 RAG: {'✅ Enabled' if rag else '❌ Disabled'}")
    print(f"💬 Conversation Tracking: ✅ Enabled (Anti-Repetition)")
    print(f"🎯 Adaptive Tuning: ✅ Enabled")
    print(f"📊 Knowledge Base: {rag.count_incidents() if rag else 0} incidents")
    
    print("\n🌐 DASHBOARDS:")
    print(f"  • API Docs:        http://localhost:8000/docs")
    print(f"  • Health Check:    http://localhost:8000/health")
    print(f"  • Qdrant UI:       {QDRANT_URL}/dashboard")
    
    print("\n📍 ANALYSIS ENDPOINTS:")
    print("  • POST /analyze                - Basic analysis")
    print("  • POST /analyze/stream         - Smooth streaming")
    print("  • POST /analyze/agentic-stream - 🏆 Competition (Anti-Repetition)")
    
    print("\n💬 CONVERSATION:")
    print("  • GET  /conversation/{id}      - View history")
    print("  • DELETE /conversation/{id}    - Clear history")
    
    print("\n📍 INCIDENT MANAGEMENT:")
    print("  • POST /incidents/seed")
    print("  • POST /incidents/add")
    print("  • GET  /incidents/search")
    print("  • GET  /incidents/list")
    
    print("\n📊 ANALYTICS (Legacy):")
    print("  • GET  /analytics/dashboard    - Full dashboard")
    print("  • GET  /analytics/adaptive     - Adaptive stats")
    
    print("\n🎯 ENHANCED ANALYTICS API (For External Apps):")
    print("  • GET  /api/analytics/overview      - High-level metrics")
    print("  • GET  /api/analytics/performance   - Response time stats")
    print("  • GET  /api/analytics/quality       - AI quality tracking")
    print("  • GET  /api/analytics/rag           - Knowledge base metrics")
    print("  • GET  /api/analytics/patterns      - Incident patterns")
    print("  • GET  /api/analytics/health        - System health")
    print("  • GET  /api/analytics/conversations - Conversation stats")
    print("  • GET  /api/analytics/realtime      - Live statistics")
    print("  • GET  /api/analytics/export        - Complete data dump")
    
    print("\n🔔 WEBHOOKS:")
    print("  • POST /api/analytics/webhook/subscribe - Subscribe to events")
    print("  • GET  /api/analytics/webhook/list      - List subscribers")
    
    print("\n🧪 TEST DATA:")
    print("  • POST /test/generate-analytics - Generate sample data")
    print("  • POST /test/reset-analytics    - Reset all metrics")
    
    print("\n⚙️  CONFIGURATION:")
    print("  • GET  /config/current         - View config")
    print("  • POST /config/reset           - Reset config")
    
    print("\n🏆 Ready for competition - NO MORE REPETITION!")
    print("📊 Enhanced Analytics API ready for external dashboards!")
    print("="*70 + "\n")
    
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")