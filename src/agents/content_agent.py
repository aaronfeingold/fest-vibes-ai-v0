"""Content Agent for generating original posts and contextual comments.

Revamped for: Gen‑Z (18–25) NOLA nightlife + demo fest‑guide ads
- Brand voice, lexicon, and prompt packs
- Event‑driven demo threads (best‑effort; uses optional DB calls)
- No emojis, light slang, authentic, under‑21 safe (focus on music/venues)
"""

import asyncio
import random
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import (
    ContentCache,
    ContentType,
    ContentGenerationRequest,
    EngagementHistory,
    ActionType,
)
from src.utils.llm_client import LLMClient
from src.database.rag_manager import RAGManager, EventSearchResult, OptimizedSchedule
from src.utils.route_calculator import RouteCalculator, VenueCoordinate
from src.prompts.rag_prompts import RAGPrompts


class ContentAgent(BaseAgent):
    """Agent responsible for content generation and management."""

    # -----------------------------
    # Brand Voice + Lexicon
    # -----------------------------
    BRAND_VOICE = {
        "who_we_are": (
            "You're writing for Fest Vibes NOLA — a Gen‑Z nightlife compass for New Orleans. "
            "We help students and young creatives assemble their own decentralized music fest any night."
        ),
        "audience": "18–25 in/around NOLA (students, young creatives). Diverse, inclusive, budget‑aware.",
        "what_we_do": (
            "We match micro‑genres and vibes to venues and build hop‑friendly lineups across the city. "
            "Focus on music discovery, not drinking. Avoid brand‑speak."
        ),
        "style_rules": [
            "No emojis.",
            "Avoid hashtags unless 1 purposeful tag improves discovery (default to none).",
            "Keep it conversational, first/second‑person.",
            "Be specific to NOLA culture: brass, bounce, funk, zydeco, queer‑friendly scenes, late‑night sets.",
            "Never imply under‑21 drinking; say '18+ venues' when relevant.",
            "No hard sells. Invite with curiosity and agency.",
            "Tweets must be <= 280 chars. Prefer 160–240.",
            "One crisp CTA max (reply keyword, tap to build, or share with crew).",
        ],
        "tone": "curious, warm, a little mischievous, absolutely local",
        "banned": [
            "emojis",
            "🔥",
            "💯",
            "best in town",
            "limited time offer",
            "sign up now",
        ],
    }

    CITY_LEXICON = {
        "micro_genres": [
            "brass",
            "bounce",
            "swamp funk",
            "zydeco",
            "second‑line grooves",
            "neo‑soul",
            "alt‑indie on Freret",
            "vinyl DJ nights",
            "Bywater art‑house sets",
            "warehouse afters",
            "Afrobeat",
            "house",
            "bass",
            "singer‑songwriter",
            "jam‑funk",
            "trad jazz",
        ],
        "neighborhoods": [
            "Uptown",
            "Bywater",
            "Marigny",
            "Treme",
            "Mid‑City",
            "CBD",
            "Frenchmen",
            "Freret",
        ],
        "venue_types": [
            "18+ venue",
            "all‑ages room",
            "listening room",
            "backyard stage",
            "dive bar",
        ],
        "slang": [
            "pull up",
            "post up",
            "lineup",
            "set",
            "after",
            "crew",
            "vibe check",
            "roll through",
        ],
    }

    CTA_PATTERNS = [
        "Reply VIBE for a mini lineup.",
        "DM 'map' and we’ll stitch your hop route.",
        "Share this with your crew and split the hop.",
        "Want your version? Reply 'guide' and your micro‑genre.",
    ]

    # Post modes help diversify the feed
    TWEET_MODES = [
        "hook",  # short, culture‑rooted hook + CTA
        "guide",  # mini demo fest‑guide built from events
        "rag_schedule",  # RAG-powered real event schedules and routes
        "callout",  # call‑and‑response prompt (UGC starter)
        "hot_take",  # specific opinion about a scene w/ invite to discuss
        "poll",  # either/or choice about venues/genres
        "memeish",  # playful observation (no copyrighted memes)
    ]

    # Few‑shot examples to steer style (kept short; not real posts)
    FEW_SHOTS = [
        # Hook
        (
            "hook",
            "Frenchmen brass at 7, river breeze at 9, bounce set after midnight. Your city, your lineup. "
            "Want a custom hop? Reply VIBE.",
        ),
        # Guide
        (
            "guide",
            "Tonight's DIY fest: 7:30 Tip’s (brass), 9:45 d.b.a. (funk), 12:15 Hi‑Ho (bounce). "
            "We’ll ping you when it’s time to roll. Want your version? Reply 'guide' + your micro‑genre.",
        ),
        # Callout
        (
            "callout",
            "Where do you go Uptown on a first date that has the best vibes to music ratio?",
        ),
        # Hot take
        (
            "hot_take",
            "St. Claude has awesome DJ sets. Prove me wrong.",
        ),
        # Poll
        (
            "poll",
            "Pick your Friday: brass warm‑up on Frenchmen or emo-core off Freret?",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__("ContentAgent", *args, **kwargs)

        # Initialize LLM client
        self.llm_client = LLMClient(self.config.llm)

        # Initialize RAG components
        self.rag_manager = RAGManager(self.config.database, self.llm_client)
        self.route_calculator = RouteCalculator()

        # Queues
        self.post_queue = asyncio.Queue()
        self.comment_queue = asyncio.Queue()

        # Templates and inspiration (re‑tuned)
        self.content_themes = {
            "music": [
                "Frenchmen brass pre‑game then bounce afters",
                "Bywater vinyl DJ night with backyard lights",
                "Freret indie then a late funk sit‑in",
                "Treme second‑line grooves rolling into a jam",
            ],
            "culture": [
                "Neighborhood‑first: each block has a different BPM",
                "Lineups over chaos: choose your night, don’t chase it",
                "DIY fest energy any day of the week",
            ],
            "guides": [
                "3‑stop hop for funk heads, budget‑friendly",
                "Queer‑friendly dance route with two 18+ venues",
                "Zero‑small‑talk date plan: listening rooms + late snack",
            ],
            "genz": [
                "We built a city‑as‑festival switchboard for your crew",
                "Don’t scroll for hours; tell us your micro‑genre",
                "Your vibe isn’t random — map it",
            ],
            "events": [
                "Sample tonight with brass → funk → bounce",
                "Late‑night house off St. Claude after the show",
                "Zero‑cover openers, pay‑what‑you‑can finales",
            ],
        }

        # Contextual response patterns (more specific, still safe)
        self.response_patterns = {
            "supportive": [
                "Real. NOLA’s best nights start with one good pick.",
                "Facts — lineups beat FOMO every time.",
                "Love this. Which neighborhood are you starting in?",
            ],
            "enthusiastic": [
                "Say less. We’re already mapping a route.",
                "This is the energy. What’s the anchor set?",
            ],
            "curious": [
                "If you had two hours, which micro‑genre gets the slot?",
                "Crew size tonight? We can make the hop smoother.",
            ],
            "relatable": [
                "We’ve all done the venue roulette. Never again.",
                "When the opener hits and the whole plan changes — we get it.",
            ],
        }

    # -----------------------------
    # Public entrypoint
    # -----------------------------
    async def execute(self) -> Dict[str, Any]:
        results = {
            "posts_generated": 0,
            "comments_generated": 0,
            "posts_published": 0,
            "cache_maintained": False,
            "errors": [],
        }
        try:
            cache_results = await self._maintain_content_cache()
            results.update(
                {
                    "posts_generated": cache_results["posts_generated"],
                    "comments_generated": cache_results["comments_generated"],
                    "cache_maintained": True,
                }
            )

            publish_results = await self._publish_scheduled_content()
            results["posts_published"] = publish_results["published"]

            await self._generate_contextual_comments()
            await self._update_content_performance()
        except Exception as e:
            logger.error(f"Content agent execution failed: {e}")
            results["errors"].append(str(e))
            raise
        return results

    # -----------------------------
    # Cache maintenance
    # -----------------------------
    async def _maintain_content_cache(self) -> Dict[str, Any]:
        posts_generated = 0
        comments_generated = 0
        try:
            unused_posts = await self.mongodb.get_unused_content(ContentType.POST.value, limit=100)
            unused_comments = await self.mongodb.get_unused_content(ContentType.COMMENT.value, limit=100)

            if len(unused_posts) < self.config.agents.min_content_cache_size:
                needed = self.config.agents.max_content_cache_size - len(unused_posts)
                posts_generated = await self._generate_original_posts(needed)

            if len(unused_comments) < self.config.agents.min_content_cache_size:
                needed = self.config.agents.max_content_cache_size - len(unused_comments)
                comments_generated = await self._generate_template_comments(needed)

            logger.info(
                f"Content cache maintained: {posts_generated} posts, {comments_generated} comments generated"
            )
        except Exception as e:
            logger.error(f"Failed to maintain content cache: {e}")
            raise
        return {
            "posts_generated": posts_generated,
            "comments_generated": comments_generated,
        }

    # -----------------------------
    # Generation helpers
    # -----------------------------
    def _pick_mode(self) -> str:
        # Weighted pick: RAG schedule and guide more frequent for showcasing functionality
        weights = {
            "hook": 0.25,
            "guide": 0.20,
            "rag_schedule": 0.30,  # High weight for RAG-powered schedules
            "callout": 0.10,
            "hot_take": 0.10,
            "poll": 0.03,
            "memeish": 0.02,
        }
        modes, probs = zip(*weights.items())
        r = random.random()
        cum = 0.0
        for m, p in zip(modes, probs):
            cum += p
            if r <= cum:
                return m
        return "rag_schedule"

    async def _generate_original_posts(self, count: int) -> int:
        generated = 0
        for _ in range(count):
            if not self.safety_limits.can_perform_action("posts"):
                logger.warning("Daily post generation limit reached")
                break

            mode = self._pick_mode()
            theme = random.choice(list(self.content_themes.keys()))
            inspiration = random.choice(self.content_themes[theme])

            # Handle RAG-powered schedule generation
            if mode == "rag_schedule":
                try:
                    rag_result = await self._generate_rag_schedule_post()
                    if rag_result and rag_result.get("valid"):
                        cache = ContentCache(
                            content_type=ContentType.POST,
                            content=self._postprocess_tweet(rag_result["content"]),
                            prompt_used=rag_result.get("prompt", ""),
                            tags=[mode, "rag", theme],
                            word_count=rag_result.get("word_count", 0),
                            character_count=rag_result.get("character_count", 0),
                            sentiment_score=rag_result.get("sentiment_score", 0.0),
                        )
                        await self.mongodb.create_content(cache)
                        generated += 1
                        logger.debug(
                            f"Generated RAG schedule post: {cache.content[:80]}..."
                        )
                        await asyncio.sleep(
                            random.uniform(2.0, 4.0)
                        )  # Longer delay for RAG calls
                        continue
                except Exception as e:
                    logger.warning(
                        f"RAG schedule generation failed, falling back to regular mode: {e}"
                    )
                    mode = "guide"  # Fallback to guide mode

            # Try to include live event snippets for guides
            events: List[Dict[str, Any]] = []
            if mode == "guide":
                try:
                    events = await self._fetch_event_candidates(
                        genres=random.sample(self.CITY_LEXICON["micro_genres"], k=2),
                        days_ahead=3,
                    )
                except Exception as e:
                    logger.debug(f"Event fetch skipped: {e}")

            context = self._compose_post_context(
                mode=mode, theme=theme, inspiration=inspiration, events=events
            )

            request = ContentGenerationRequest(
                content_type=ContentType.POST,
                context=context,
                tone="NOLA‑casual",
                max_length=280,
            )

            content_result = await self._generate_content_with_validation(request)

            if content_result.get("valid"):
                cache = ContentCache(
                    content_type=ContentType.POST,
                    content=self._postprocess_tweet(content_result["content"]),
                    prompt_used=content_result.get("prompt", ""),
                    tags=[mode, theme],
                    word_count=content_result.get("word_count", 0),
                    character_count=content_result.get("character_count", 0),
                    sentiment_score=content_result.get("sentiment_score", 0.0),
                )
                await self.mongodb.create_content(cache)
                generated += 1
                logger.debug(f"Generated [{mode}] post: {cache.content[:80]}...")

            await asyncio.sleep(random.uniform(1.5, 3.5))
        return generated

    async def _generate_template_comments(self, count: int) -> int:
        generated = 0
        for _ in range(count):
            pattern_type = random.choice(list(self.response_patterns.keys()))
            base = random.choice(self.response_patterns[pattern_type])

            request = ContentGenerationRequest(
                content_type=ContentType.COMMENT,
                context=self._compose_comment_context(base, pattern_type),
                tone="conversational",
                max_length=140,
            )
            result = await self._generate_content_with_validation(request)
            if result.get("valid"):
                cache = ContentCache(
                    content_type=ContentType.COMMENT,
                    content=self._postprocess_tweet(result["content"]),
                    prompt_used=result.get("prompt", ""),
                    tags=[pattern_type],
                    word_count=result.get("word_count", 0),
                    character_count=result.get("character_count", 0),
                    sentiment_score=result.get("sentiment_score", 0.0),
                )
                await self.mongodb.create_content(cache)
                generated += 1
            await asyncio.sleep(random.uniform(1, 2))
        return generated



    def _postprocess_tweet(self, text: str) -> str:
        # Strip emojis & enforce char limit politely
        text = re.sub(
            r"[\U00010000-\U0010ffff]", "", text
        )  # remove non‑BMP chars (emojis)
        text = re.sub(r"\s+", " ", text).strip()
        # Remove double spaces before punctuation
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # Keep under 280
        if len(text) > 280:
            text = text[:277].rstrip() + "…"
        # Avoid multiple hashtags; keep max 1
        if text.count("#") > 1:
            parts = text.split()
            kept = 0
            pruned = []
            for w in parts:
                if w.startswith("#"):
                    if kept == 0:
                        pruned.append(w)
                        kept += 1
                    # else drop
                else:
                    pruned.append(w)
            text = " ".join(pruned)
        return text

    async def _generate_content_with_validation(
        self, request: ContentGenerationRequest
    ) -> Dict[str, Any]:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Use the provided context as the complete prompt (already formatted by ContentPrompts)
                # or fall back to config-based prompt if no context provided
                if request.context:
                    prompt = request.context
                else:
                    # Fallback to config-based prompt for backward compatibility
                    base_prompt = self.config.get_content_generation_prompt(
                        request.content_type.value
                    )
                    prompt = base_prompt

                llm_result = await self.llm_client.generate_content(
                    prompt,
                    content_type=request.content_type.value,
                    context=request.context,
                    max_tokens=min(request.max_length * 2, 500),
                    temperature=0.7 + (attempt * 0.1),
                )
                content = llm_result["content"]

                # Additional local validations using centralized validation
                validation_result = BasePrompts.validate_content_rules(content)
                if not validation_result["valid"]:
                    logger.warning(f"Content validation failed: {validation_result['issues']}; retrying")
                    raise ValueError("validation failed")
                if len(content) > 280:
                    logger.debug(
                        "Content exceeds char limit; will retry with higher temp for brevity"
                    )

                validation = self.llm_client.validate_content(content, request.content_type.value)
                if validation.get("valid"):
                    sentiment = await self.llm_client.analyze_sentiment(content)
                    return {
                        "content": content,
                        "valid": True,
                        "prompt": prompt,
                        "word_count": validation.get("word_count", 0),
                        "character_count": validation.get("character_count", 0),
                        "sentiment_score": sentiment.get("sentiment_score", 0.0),
                        "model_used": llm_result.get("model", "unknown"),
                        "attempt": attempt + 1,
                    }
                else:
                    logger.warning(
                        f"Content validation failed (attempt {attempt + 1}): {validation.get('issues')}"
                    )
                    if attempt == max_attempts - 1:
                        return {
                            "content": content,
                            "valid": False,
                            "issues": validation.get("issues"),
                            "attempt": attempt + 1,
                        }
            except Exception as e:
                logger.error(f"Content generation attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2**attempt)
        return {"content": "", "valid": False, "attempt": max_attempts}

    # -----------------------------
    # Publishing & selection
    # -----------------------------
    async def _publish_scheduled_content(self) -> Dict[str, Any]:
        published = 0
        try:
            if not self.safety_limits.can_perform_action("posts"):
                logger.info("Daily post limit reached, skipping content publishing")
                return {"published": 0}

            available_posts = await self.mongodb.get_unused_content(
                ContentType.POST.value, limit=8
            )
            if not available_posts:
                logger.warning("No posts available for publishing")
                return {"published": 0}

            selected_post = self._select_best_post(available_posts)
            if selected_post:
                success = await self._publish_post(selected_post)
                if success:
                    published = 1
        except Exception as e:
            logger.error(f"Failed to publish scheduled content: {e}")
            raise
        return {"published": published}

    def _select_best_post(self, posts: List[ContentCache]) -> Optional[ContentCache]:
        if not posts:
            return None
        scored: List[Tuple[ContentCache, float]] = []
        for post in posts:
            score = 0.0
            # Prefer positive sentiment
            if post.sentiment_score and post.sentiment_score > 0:
                score += post.sentiment_score * 0.3
            # Prefer thread/guide cues
            if any(
                k in (post.content or "").lower()
                for k in ["tonight", "lineup", "guide", "route"]
            ):
                score += 0.25
            # Prefer optimal length
            n = post.character_count or len(post.content)
            if 120 <= n <= 240:
                score += 0.25
            elif 240 < n <= 280:
                score += 0.15
            # Light randomness
            score += random.uniform(0, 0.2)
            scored.append((post, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    async def _publish_post(self, post: ContentCache) -> bool:
        try:
            result = await self.safe_twitter_call(
                "create_tweet", "bot_post", text=post.content
            )
            await self.mongodb.mark_content_used(
                str(post.id), {"published_at": datetime.utcnow().isoformat()}
            )

            engagement = EngagementHistory(
                target_tweet_id=str(result.get("id", "")) if result else "",
                target_user_id="",
                action_type=ActionType.COMMENT,  # Keeping COMMENT for posts as in original
                success=True,
                response_data={"result": str(result) if result else "dry_run"},
                content_used=post.content,
            )
            await self.mongodb.create_engagement(engagement)
            logger.info(f"Published post: {post.content[:80]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to publish post: {e}")
            engagement = EngagementHistory(
                target_tweet_id="",
                target_user_id="",
                action_type=ActionType.COMMENT,
                success=False,
                error_message=str(e),
                content_used=post.content,
            )
            await self.mongodb.create_engagement(engagement)
            return False

    # -----------------------------
    # Contextual comments
    # -----------------------------
    async def _generate_contextual_comments(self) -> None:
        try:
            unused = await self.mongodb.get_unused_content(
                ContentType.COMMENT.value, limit=20
            )
            if len(unused) < 10:
                logger.info("Generating additional contextual comments")
                await self._generate_template_comments(10)
        except Exception as e:
            logger.error(f"Failed to generate contextual comments: {e}")

    async def generate_contextual_comment(
        self, tweet_context: Dict[str, Any]
    ) -> Optional[str]:
        try:
            tweet_text = tweet_context.get("text", "")
            author_info = tweet_context.get("author", {})
            prompt = ContentPrompts.get_contextual_reply_prompt(tweet_text, author_info)
            request = ContentGenerationRequest(
                content_type=ContentType.COMMENT,
                context=prompt,
                tone="conversational",
                max_length=140,
            )
            result = await self._generate_content_with_validation(request)
            if result.get("valid"):
                return self._postprocess_tweet(result["content"])
            else:
                template_comments = await self.mongodb.get_unused_content(
                    ContentType.COMMENT.value, limit=5
                )
                if template_comments:
                    return random.choice(template_comments).content
                else:
                    # Use fallback from ContentPrompts
                    fallback_responses = ContentPrompts.RESPONSE_PATTERNS.get("supportive", ["Thanks for sharing!"])
                    return random.choice(fallback_responses)
            return None
        except Exception as e:
            logger.error(f"Failed to generate contextual comment: {e}")
            return None

    # -----------------------------
    # Performance update (stub same as original)
    # -----------------------------
    async def _update_content_performance(self) -> None:
        try:
            recent = await self.mongodb.get_recent_engagements(
                action_type=ActionType.COMMENT.value, hours=24
            )
            for engagement in recent:
                if engagement.content_used and engagement.success:
                    performance_update = {
                        "engagement_rate": random.uniform(0.02, 0.06),
                        "last_updated": datetime.utcnow().isoformat(),
                    }
                    # TODO: persist to content item if you track it; left as no‑op as in original
        except Exception as e:
            logger.error(f"Failed to update content performance: {e}")

    async def _should_run(self) -> bool:
        if not await super()._should_run():
            return False
        remaining = self.safety_limits.get_remaining_actions("posts")
        if remaining["daily"] <= 0:
            logger.debug("Daily post limit reached")
            return False
        return True

    def get_agent_status(self) -> Dict[str, Any]:
        base_status = self.get_health_status()
        total_queue_size = self.post_queue.qsize() + self.comment_queue.qsize()
        agent_queue_size.labels(agent=self.name).set(total_queue_size)
        return {
            **base_status,
            "queue_sizes": {
                "posts": self.post_queue.qsize(),
                "comments": self.comment_queue.qsize(),
            },
            "llm_client": self.llm_client.get_client_status(),
            "content_themes": len(BasePrompts.CONTENT_THEMES),
            "response_patterns": len(ContentPrompts.RESPONSE_PATTERNS),
            "safety_limits": {
                "posts": self.safety_limits.get_remaining_actions("posts")
            },
        }

    # -----------------------------
    # RAG-powered content generation
    # -----------------------------
    async def _generate_rag_schedule_post(self) -> Optional[Dict[str, Any]]:
        """Generate a post using RAG to find real events and create optimized schedules."""
        try:
            # Connect RAG manager if not connected
            if not self.rag_manager._connected:
                await self.rag_manager.connect()

            # Generate a search query based on current themes
            search_queries = [
                "jazz brass funk music tonight New Orleans",
                "bounce hip hop live music venue",
                "indie rock alternative live show NOLA",
                "electronic house dance music event",
                "acoustic singer songwriter intimate venue",
            ]
            query = random.choice(search_queries)

            # Search for events using config settings
            events = await self.rag_manager.search_events_by_query(
                query=query,
                days_ahead=self.config.agents.rag_search_days_ahead,
                similarity_threshold=self.config.agents.rag_similarity_threshold,
                limit=self.config.agents.rag_max_events_per_search,
            )

            if not events or len(events) < 2:
                logger.debug("Not enough events found for RAG schedule generation")
                return None

            # Build optimized schedule
            schedule = await self.rag_manager.build_event_schedule(
                events=events,
                max_venues=self.config.agents.rag_max_venues_per_schedule,
                schedule_type=random.choice(
                    ["distance_optimized", "time_optimized", "genre_focused"]
                ),
            )

            if not schedule or len(schedule.events) < 2:
                logger.debug("Could not build viable schedule from events")
                return None

            # Convert to format for prompt generation
            event_dicts = []
            for event in schedule.events:
                event_dict = {
                    "performance_time": event.performance_time.isoformat(),
                    "venue_name": event.venue_name,
                    "artist_name": event.artist_name,
                    "genres": event.genres,
                    "description": event.description,
                }
                event_dicts.append(event_dict)
                logger.debug(f"Event dict created: {event_dict}")

            logger.debug(f"Total event_dicts created: {len(event_dicts)}")

            # Generate route summary
            route_summary = self.route_calculator.generate_route_summary(
                self._convert_schedule_to_route(schedule)
            )

            route_info = {
                "total_distance_miles": schedule.total_distance_miles,
                "total_travel_time_minutes": schedule.total_travel_time_minutes,
                "total_estimated_cost": 0.0,  # Will be calculated by route_calculator
                "route_summary": route_summary,
                "schedule_type": schedule.schedule_type,
            }

            # Choose prompt type based on schedule characteristics
            prompt_type = self._choose_rag_prompt_type(schedule)
            prompt = self._get_rag_prompt(
                prompt_type, event_dicts, route_info, schedule
            )

            # Generate content using LLM
            llm_result = await self.llm_client.generate_content(
                prompt,
                content_type="post",
                context="RAG schedule generation",
                max_tokens=400,
                temperature=0.7,
            )

            content = llm_result["content"]

            # Validate content
            validation = self.llm_client.validate_content(content, "post")
            if not validation.get("valid"):
                logger.warning(
                    f"RAG content validation failed: {validation.get('issues')}"
                )
                return None

            # Analyze sentiment
            sentiment = await self.llm_client.analyze_sentiment(content)

            return {
                "content": content,
                "valid": True,
                "prompt": prompt,
                "word_count": validation.get("word_count", 0),
                "character_count": validation.get("character_count", 0),
                "sentiment_score": sentiment.get("sentiment_score", 0.0),
                "model_used": llm_result.get("model", "unknown"),
                "events_used": len(schedule.events),
                "schedule_type": schedule.schedule_type,
            }

        except Exception as e:
            logger.error(f"RAG schedule generation failed: {e}")
            return None

    def _choose_rag_prompt_type(self, schedule: OptimizedSchedule) -> str:
        """Choose the most appropriate prompt type based on schedule characteristics."""
        if schedule.schedule_type == "genre_focused":
            # Check if there's a dominant genre
            genre_counts = {}
            for event in schedule.events:
                for genre in event.genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

            if genre_counts:
                max_genre = max(genre_counts.items(), key=lambda x: x[1])
                if max_genre[1] >= 2:  # At least 2 events of same genre
                    return "genre_focus"

        elif schedule.schedule_type == "distance_optimized":
            if schedule.total_distance_miles <= 1.5:
                return "neighborhood_focus"
            else:
                return "route_optimization"

        elif schedule.schedule_type == "time_optimized":
            return "time_optimization"

        # Default to general schedule generation
        return "schedule_generation"

    def _get_rag_prompt(
        self,
        prompt_type: str,
        event_dicts: List[Dict[str, Any]],
        route_info: Dict[str, Any],
        schedule: OptimizedSchedule,
    ) -> str:
        """Get the appropriate RAG prompt based on type."""
        if prompt_type == "genre_focus":
            # Find dominant genre
            genre_counts = {}
            for event in schedule.events:
                for genre in event.genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            primary_genre = (
                max(genre_counts.items(), key=lambda x: x[1])[0]
                if genre_counts
                else "music"
            )
            return RAGPrompts.get_genre_focus_prompt(
                event_dicts, primary_genre, route_info
            )

        elif prompt_type == "route_optimization":
            return RAGPrompts.get_route_optimization_prompt(
                event_dicts, route_info, "distance"
            )

        elif prompt_type == "time_optimization":
            return RAGPrompts.get_time_optimization_prompt(event_dicts, route_info)

        elif prompt_type == "neighborhood_focus":
            # Determine primary neighborhood (simplified)
            neighborhood = (
                "Frenchmen"
                if "frenchmen" in str(schedule.events).lower()
                else "Marigny"
            )
            return RAGPrompts.get_neighborhood_focus_prompt(
                event_dicts, neighborhood, route_info
            )

        else:
            return RAGPrompts.get_schedule_generation_prompt(event_dicts, route_info)

    def _convert_schedule_to_route(self, schedule: OptimizedSchedule):
        """Convert OptimizedSchedule to route format for route_calculator."""
        # Create a mock optimized route for route summary generation
        from src.utils.route_calculator import (
            OptimizedRoute,
            RouteSegment,
            TransportMode,
        )

        segments = []
        for i, transition in enumerate(schedule.venue_transitions):
            segment = RouteSegment(
                from_venue_id=transition.from_venue_id,
                to_venue_id=transition.to_venue_id,
                from_venue_name=transition.from_venue_name,
                to_venue_name=transition.to_venue_name,
                from_address="",
                to_address="",
                distance_miles=transition.distance_miles,
                travel_time_minutes=transition.walking_time_minutes,
                transport_mode=TransportMode.WALKING,
                instructions="",
                estimated_cost=0.0,
            )
            segments.append(segment)

        return OptimizedRoute(
            venue_ids=[e.venue_id for e in schedule.events],
            segments=segments,
            total_distance_miles=schedule.total_distance_miles,
            total_travel_time_minutes=schedule.total_travel_time_minutes,
            total_estimated_cost=0.0,
            recommended_transport_modes=[TransportMode.WALKING],
            route_efficiency_score=1.0,
        )

    # -----------------------------
    # Event fetch (best‑effort helper)
    # -----------------------------
    async def _fetch_event_candidates(
        self, *, genres: List[str], days_ahead: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Try to pull a few relevant events from your DB to seed demo guides.
        This assumes your Mongo wrapper exposes something like `search_events` with filters.
        Fallback is an empty list (the LLM will still write a guide using seeds).
        Expected return shape per item: {title, venue, start_time, time_short, genre, neighborhood}
        """
        try:
            now = datetime.utcnow()
            end = now + timedelta(days=days_ahead)
            # Adjust this call to your actual DB API
            events = await self.mongodb.search_events(
                genres=genres,
                start_time_from=now.isoformat(),
                start_time_to=end.isoformat(),
                limit=6,
            )
            # Optionally normalize
            normd = []
            for e in events:
                t = e.get("start_time")
                time_short = e.get("time_short")
                if not time_short and t:
                    try:
                        dt = datetime.fromisoformat(t)
                        time_short = dt.strftime("%I:%M%p").lstrip("0").lower()
                    except Exception:
                        time_short = ""
                normd.append(
                    {
                        "title": e.get("title"),
                        "venue": e.get("venue_name") or e.get("venue"),
                        "start_time": t,
                        "time_short": time_short,
                        "genre": e.get("genre") or ", ".join(e.get("tags", [])[:2]),
                        "neighborhood": e.get("neighborhood"),
                    }
                )
            return normd
        except AttributeError:
            # search_events not available; silently skip
            return []
        except Exception as e:
            logger.debug(f"search_events failed: {e}")
            return []
