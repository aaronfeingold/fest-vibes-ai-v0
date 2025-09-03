"""Agent-specific prompts for content generation and engagement.

This module contains prompt classes for different agent types, following the
pattern established by RAGPrompts but for agent-specific functionality.
"""

import random
from typing import Dict, List, Any, Optional

from src.prompts.base_prompts import BasePrompts


class ContentPrompts:
    """Prompt generation for content agent posts and comments."""

    # Tweet modes for diversifying content feed - from content_agent.py
    TWEET_MODES = [
        "hook",  # short, culture‑rooted hook + CTA
        "guide",  # mini demo fest‑guide built from events
        "rag_schedule",  # RAG-powered real event schedules and routes
        "callout",  # call‑and‑response prompt (UGC starter)
        "hot_take",  # specific opinion about a scene w/ invite to discuss
        "poll",  # either/or choice about venues/genres
        "memeish",  # playful observation (no copyrighted memes)
    ]

    # Few‑shot examples to steer style - from content_agent.py
    FEW_SHOT_EXAMPLES = [
        # Hook
        (
            "hook",
            "Frenchmen brass at 7, river breeze at 9, bounce set after midnight. Your city, your lineup. "
            "Want a custom hop? Reply VIBE.",
        ),
        # Guide
        (
            "guide",
            "Tonight's DIY fest: 7:30 Tip's (brass), 9:45 d.b.a. (funk), 12:15 Hi‑Ho (bounce). "
            "We'll ping you when it's time to roll. Want your version? Reply 'guide' + your micro‑genre.",
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

    # Response patterns for template comments - from content_agent.py
    RESPONSE_PATTERNS = {
        "supportive": [
            "Real. NOLA's best nights start with one good pick.",
            "Facts — lineups beat FOMO every time.",
            "Love this. Which neighborhood are you starting in?",
        ],
        "enthusiastic": [
            "Say less. We're already mapping a route.",
            "This is the energy. What's the anchor set?",
        ],
        "curious": [
            "If you had two hours, which micro‑genre gets the slot?",
            "Crew size tonight? We can make the hop smoother.",
        ],
        "relatable": [
            "We've all done the venue roulette. Never again.",
            "When the opener hits and the whole plan changes — we get it.",
        ],
    }

    @staticmethod
    def get_post_generation_prompt(
        mode: str, theme: str, inspiration: str, events: List[Dict[str, Any]] = None
    ) -> str:
        """Generate prompt for creating posts - extracted from _compose_post_context()."""

        # Get few-shot examples for this mode
        few_shot_examples = [
            ex for (m, ex) in ContentPrompts.FEW_SHOT_EXAMPLES if m == mode
        ]
        fewshot = (
            f"Examples for {mode}:\n- " + "\n- ".join(few_shot_examples)
            if few_shot_examples
            else ""
        )

        # Format events if provided
        events_text = BasePrompts.format_event_snippet(events or [])

        # Get contextual elements
        context_elements = BasePrompts.get_contextual_elements()

        # Get brand voice context
        brand_context = BasePrompts.get_brand_voice_context()

        return (
            f"{brand_context}\n\n"
            f"Task: Write ONE tweet in the '{mode}' mode for theme '{theme}'.\n"
            f"Inspiration seed: {inspiration}.\n"
            f"Use micro‑genres like: {context_elements['micro_genres']}. "
            f"Neighborhood to nod to: {context_elements['neighborhood']}.\n"
            f"{events_text}\n\n"
            f"{fewshot}\n\n"
            f"Output: just the tweet text (no preamble). End with a soft CTA like: {context_elements['cta']}"
        )

    @staticmethod
    def get_comment_generation_prompt(base_response: str, pattern_type: str) -> str:
        """Generate prompt for template comments - extracted from _compose_comment_context()."""
        return (
            f"You reply on behalf of Fest Vibes NOLA. Style: concise, warm, no emojis.\n"
            f"Audience: Gen‑Z 18–25, diverse, budget‑aware.\n"
            f"Pattern: {pattern_type}. Base idea: {base_response}\n"
            f"Rules: be specific to NOLA music culture; invite a tiny next step (ask a question or suggest a neighborhood).\n"
            f"Keep to <= 100 characters if possible."
        )

    @staticmethod
    def get_contextual_reply_prompt(
        tweet_text: str, author_info: Dict[str, Any]
    ) -> str:
        """Generate prompt for contextual replies to specific tweets."""
        return (
            "Create a thoughtful, authentic reply to this tweet. Be supportive and engaging, "
            "true to New Orleans culture and Gen‑Z communication. \n"
            f'Tweet: "{tweet_text}"\n'
            f"Author bio: {author_info.get('description', 'No bio available')}\n"
            "Rules: no emojis; avoid brand‑speak; keep under 120 characters; ask 1 concrete question or offer 1 tiny next step."
        )

    @staticmethod
    def get_theme_inspiration(theme: str) -> str:
        """Get inspiration text for a given theme."""
        theme_options = BasePrompts.CONTENT_THEMES.get(
            theme, BasePrompts.CONTENT_THEMES["music"]
        )
        return random.choice(theme_options)

    @staticmethod
    def get_random_response_pattern(pattern_type: str = None) -> str:
        """Get a random response pattern, optionally filtered by type."""
        if pattern_type and pattern_type in ContentPrompts.RESPONSE_PATTERNS:
            return random.choice(ContentPrompts.RESPONSE_PATTERNS[pattern_type])

        # Return random pattern from any type
        all_patterns = []
        for patterns in ContentPrompts.RESPONSE_PATTERNS.values():
            all_patterns.extend(patterns)
        return random.choice(all_patterns)


class EngagementPrompts:
    """Prompt generation for engagement agent interactions."""

    # Fallback responses when template generation fails - from engagement_agent.py
    FALLBACK_RESPONSES = [
        "This is so important",
        "Love this perspective!",
        "Absolutely needed to see this",
        "This hits different",
        "So here for this energy",
    ]

    @staticmethod
    def get_contextual_comment_prompt(
        tweet_text: str, tweet_context: Dict[str, Any]
    ) -> str:
        """Generate prompt for contextual comments on tweets."""
        author_info = tweet_context.get("author", {})

        return (
            f"Create a brief, authentic comment on this tweet as Fest Vibes NOLA:\n\n"
            f'Tweet: "{tweet_text}"\n'
            f"Author: @{author_info.get('username', 'unknown')}\n"
            f"Context: {author_info.get('description', 'No bio')}\n\n"
            f"Guidelines:\n"
            f"- Keep it under 100 characters\n"
            f"- No emojis or hashtags\n"
            f"- Be supportive and genuine\n"
            f"- Reference NOLA culture if relevant\n"
            f"- Ask a question or suggest a next step if appropriate\n\n"
            f"Output only the comment text."
        )

    @staticmethod
    def get_fallback_response() -> str:
        """Get a fallback response when generation fails."""
        return random.choice(EngagementPrompts.FALLBACK_RESPONSES)

    @staticmethod
    def get_repost_comment_prompt(
        original_tweet: Dict[str, Any], decision_score: float
    ) -> str:
        """Generate prompt for quote tweet comments when reposting."""
        return (
            f"Create a brief quote tweet comment for this high-value content (score: {decision_score:.2f}):\n\n"
            f'Original tweet: "{original_tweet.get("text", "")}"\n'
            f"Author: @{original_tweet.get('author', {}).get('username', 'unknown')}\n\n"
            f"Guidelines:\n"
            f"- Add value, don't just repeat\n"
            f"- Keep under 100 characters (leaves room for quoted content)\n"
            f"- Be authentic to Fest Vibes NOLA voice\n"
            f"- No emojis\n"
            f"- Reference why this matters for NOLA nightlife if relevant\n\n"
            f"Output only the comment text."
        )


class FollowPrompts:
    """Prompt generation for follow agent user analysis (if needed for future expansion)."""

    @staticmethod
    def get_user_analysis_prompt(user_data: Dict[str, Any]) -> str:
        """Generate prompt for analyzing user relevance (placeholder for future use)."""
        return (
            f"Analyze this Twitter user for relevance to New Orleans music and nightlife:\n\n"
            f"Username: @{user_data.get('username', 'unknown')}\n"
            f"Bio: {user_data.get('description', 'No bio')}\n"
            f"Location: {user_data.get('location', 'No location')}\n"
            f"Followers: {user_data.get('public_metrics', {}).get('followers_count', 0)}\n\n"
            f"Rate relevance 0-1 based on:\n"
            f"- New Orleans connection\n"
            f"- Music/culture interest\n"
            f"- Gen-Z alignment\n"
            f"- Authentic engagement potential\n\n"
            f'Output format: {{"score": 0.XX, "reasoning": "brief explanation"}}'
        )
