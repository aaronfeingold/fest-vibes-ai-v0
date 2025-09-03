"""Base prompts and shared components for all agents.

This module contains the core brand voice, shared constants, and common prompt
building utilities used across all agent prompt systems.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class BasePrompts:
    """Shared prompt components and brand voice definitions."""

    # Core brand voice and identity - centralized from content_agent.py
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

    # New Orleans cultural lexicon - centralized from content_agent.py
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

    # Common CTA patterns used across different content types
    CTA_PATTERNS = [
        "Reply VIBES for a mini lineup.",
        "DM 'vibes' and we'll stitch your hop route.",
        "Rizzm with the Vibes. Make your own music scene.",
        "Wanna music packed weekend? Reply 'fest vibes' and we'll clock that tea",
    ]

    @staticmethod
    def get_brand_voice_context() -> str:
        """Get formatted brand voice context for prompts."""
        rules = "\n".join(f"- {r}" for r in BasePrompts.BRAND_VOICE["style_rules"])

        return (
            f"{BasePrompts.BRAND_VOICE['who_we_are']}\n"
            f"Audience: {BasePrompts.BRAND_VOICE['audience']}\n"
            f"Mission: {BasePrompts.BRAND_VOICE['what_we_do']}\n"
            f"Tone: {BasePrompts.BRAND_VOICE['tone']}\n\n"
            f"Hard rules:\n{rules}"
        )

    @staticmethod
    def format_event_snippet(events: List[Dict[str, Any]]) -> str:
        """Format event data snippet for prompt context."""
        if not events:
            return ""

        items = []
        for event in events[:3]:  # Limit to 3 events
            when = event.get("time_short") or event.get("start_time") or ""
            title = event.get("title", "live set")
            venue = event.get("venue", "")
            genre = event.get("genre", "")
            items.append(f"{when} {title} ({genre}) @ {venue}")

        return "\nEvent picks: " + "; ".join(items)

    @staticmethod
    def get_contextual_elements(
        micro_genres: List[str] = None, neighborhood: str = None
    ) -> Dict[str, str]:
        """Get contextual elements for prompt personalization."""
        import random

        # Default selections if not provided
        if not micro_genres:
            micro_genres = random.sample(BasePrompts.CITY_LEXICON["micro_genres"], k=2)
        if not neighborhood:
            neighborhood = random.choice(BasePrompts.CITY_LEXICON["neighborhoods"])

        micro_text = ", ".join(micro_genres)
        cta = random.choice(BasePrompts.CTA_PATTERNS)

        return {"micro_genres": micro_text, "neighborhood": neighborhood, "cta": cta}

    @staticmethod
    def prepare_llm_prompt(
        base_prompt: str, content_type: str, context: Optional[str] = None
    ) -> str:
        """Prepare a complete prompt for LLM generation with domain context and instructions."""

        # Get content type instructions
        type_instructions = BasePrompts.CONTENT_TYPE_INSTRUCTIONS.get(
            content_type, "Create engaging social media content."
        )

        # Build the complete prompt
        full_prompt = f"""
{BasePrompts.DOMAIN_CONTEXT}

{type_instructions}

{base_prompt}
"""

        # Add additional context if provided
        if context:
            full_prompt += f"\n\nContext: {context}"

        # Add tone guidelines
        full_prompt += f"\n\nImportant: {BasePrompts.TONE_GUIDELINES}"

        return full_prompt.strip()

    @staticmethod
    def get_domain_context() -> str:
        """Get the current domain context for external collaboration."""
        return BasePrompts.DOMAIN_CONTEXT

    @staticmethod
    def update_domain_context(new_context: str) -> None:
        """Update domain context (for non-coder collaboration)."""
        BasePrompts.DOMAIN_CONTEXT = new_context

    @staticmethod
    def get_content_type_instruction(content_type: str) -> str:
        """Get instruction for specific content type."""
        return BasePrompts.CONTENT_TYPE_INSTRUCTIONS.get(
            content_type, "Create engaging social media content."
        )

    @staticmethod
    def update_content_type_instruction(content_type: str, instruction: str) -> None:
        """Update instruction for specific content type (for non-coder collaboration)."""
        BasePrompts.CONTENT_TYPE_INSTRUCTIONS[content_type] = instruction

    @staticmethod
    def validate_content_rules(content: str) -> Dict[str, Any]:
        """Validate content against brand rules."""
        issues = []

        # Check for banned patterns
        content_lower = content.lower()
        for banned in BasePrompts.BRAND_VOICE["banned"]:
            if banned.lower() in content_lower:
                issues.append(f"Contains banned phrase: {banned}")

        # Check length
        if len(content) > 280:
            issues.append(f"Too long: {len(content)} chars (max 280)")

        # Check for multiple hashtags
        hashtag_count = content.count("#")
        if hashtag_count > 1:
            issues.append(f"Too many hashtags: {hashtag_count} (max 1)")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "character_count": len(content),
            "word_count": len(content.split()),
        }

    # Domain context for LLM client - extracted from llm_client.py
    DOMAIN_CONTEXT = (
        "You are creating content for a Twitter bot focused on New Orleans culture, "
        "music scene, and GenZ trends. The content should be authentic, engaging, "
        "and reflect the vibrant culture of New Orleans."
    )

    # Content type specific instructions - extracted from llm_client.py
    CONTENT_TYPE_INSTRUCTIONS = {
        "post": (
            "Create an original tweet (under 280 characters) that's conversational "
            "and engaging. Avoid excessive hashtags or promotional language. "
            "Focus on local culture, music events, food, or GenZ topics."
        ),
        "comment": (
            "Create a thoughtful reply that adds value to the conversation. "
            "Be supportive, authentic, and brief. Match the tone of the original post."
        ),
        "repost_comment": (
            "Create a brief comment to accompany a repost. Add insight, context, "
            "or your perspective while keeping it concise and engaging."
        ),
    }

    # General tone guidelines - extracted from llm_client.py
    TONE_GUIDELINES = (
        "Keep the tone casual, authentic, and aligned with GenZ "
        "communication style. Avoid corporate or overly promotional language."
    )

    # Content themes for inspiration - centralized from content_agent.py
    CONTENT_THEMES = {
        "music": [
            "Frenchmen brass pre‑game then bounce afters",
            "Bywater vinyl DJ night with backyard lights",
            "Freret indie then a late funk sit‑in",
            "Treme second‑line grooves rolling into a jam",
        ],
        "culture": [
            "Neighborhood‑first: each block has a different BPM",
            "Lineups over chaos: choose your night, don't chase it",
            "DIY fest energy any day of the week",
        ],
        "guides": [
            "3‑stop hop for funk heads, budget‑friendly",
            "Queer‑friendly dance route with two 18+ venues",
            "Zero‑small‑talk date plan: listening rooms + late snack",
        ],
        "genz": [
            "We built a city‑as‑festival switchboard for your crew",
            "Don't scroll for hours; tell us your micro‑genre",
            "Your vibe isn't random — map it",
        ],
        "events": [
            "Sample tonight with brass → funk → bounce",
            "Late‑night house off St. Claude after the show",
            "Zero‑cover openers, pay‑what‑you‑can finales",
        ],
    }
