#!/usr/bin/env python3
"""
Shared utilities for content generation scripts.
Provides themes, patterns, and helper functions without requiring the full bot infrastructure.
"""

import asyncio
import random
import json
import csv
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config.settings import LLMConfig
from src.utils.llm_client import LLMClient
from src.prompts.content_utility_prompts import ContentUtilityPrompts


class ContentThemes:
    """Updated content themes matching the revamped ContentAgent for Gen‑Z NOLA nightlife."""

    THEMES = {
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

    @classmethod
    def get_random_theme(cls) -> str:
        """Get a random theme."""
        return random.choice(list(cls.THEMES.keys()))

    @classmethod
    def get_random_inspiration(cls, theme: str = None) -> tuple[str, str]:
        """Get random inspiration from a theme. Returns (theme, inspiration)."""
        if theme is None:
            theme = cls.get_random_theme()
        elif theme not in cls.THEMES:
            raise ValueError(
                f"Unknown theme: {theme}. Available: {list(cls.THEMES.keys())}"
            )

        inspiration = random.choice(cls.THEMES[theme])
        return theme, inspiration


class ResponsePatterns:
    """Updated response patterns matching the revamped ContentAgent style."""

    PATTERNS = {
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

    @classmethod
    def get_random_pattern(cls, pattern_type: str = None) -> tuple[str, str]:
        """Get random response pattern. Returns (pattern_type, response)."""
        if pattern_type is None:
            pattern_type = random.choice(list(cls.PATTERNS.keys()))
        elif pattern_type not in cls.PATTERNS:
            raise ValueError(
                f"Unknown pattern: {pattern_type}. Available: {list(cls.PATTERNS.keys())}"
            )

        response = random.choice(cls.PATTERNS[pattern_type])
        return pattern_type, response


class ContentGenerator:
    """Main content generation class."""

    def __init__(self):
        """Initialize the content generator."""
        self.llm_config = LLMConfig()
        self.llm_client = LLMClient(self.llm_config)

    async def generate_post(
        self, theme: str = None, custom_context: str = None
    ) -> Dict[str, Any]:
        """Generate a single post."""
        # Get theme and inspiration
        if custom_context:
            prompt_context = ContentUtilityPrompts.format_context(
                "custom", custom_context=custom_context
            )
            used_theme = "custom"
        else:
            used_theme, inspiration = ContentThemes.get_random_inspiration(theme)
            prompt_context = ContentUtilityPrompts.format_context(
                "theme_based", theme=used_theme, inspiration=inspiration
            )

        # Create prompt
        prompt = self._create_post_prompt(prompt_context)

        # Generate content with validation
        return await self._generate_with_validation(
            prompt, "post", used_theme, prompt_context
        )

    async def generate_comment(
        self, pattern_type: str = None, custom_context: str = None
    ) -> Dict[str, Any]:
        """Generate a single comment."""
        # Get pattern
        if custom_context:
            prompt_context = ContentUtilityPrompts.format_context(
                "custom", custom_context=custom_context
            )
            used_pattern = "custom"
        else:
            used_pattern, base_response = ResponsePatterns.get_random_pattern(
                pattern_type
            )
            prompt_context = ContentUtilityPrompts.format_context(
                "pattern_based", pattern_type=used_pattern, base_response=base_response
            )

        # Create prompt
        prompt = self._create_comment_prompt(prompt_context)

        # Generate content with validation
        return await self._generate_with_validation(
            prompt, "comment", used_pattern, prompt_context
        )

    def _create_post_prompt(self, context: str) -> str:
        """Create prompt for post generation."""
        return ContentUtilityPrompts.get_brand_enhanced_prompt("post", context)

    def _create_comment_prompt(self, context: str) -> str:
        """Create prompt for comment generation."""
        return ContentUtilityPrompts.get_brand_enhanced_prompt("comment", context)

    async def _generate_with_validation(
        self, prompt: str, content_type: str, theme: str, context: str
    ) -> Dict[str, Any]:
        """Generate content with validation and retry logic."""
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                # Generate content
                llm_result = await self.llm_client.generate_content(
                    prompt,
                    content_type=content_type,
                    context=context,
                    max_tokens=300,
                    temperature=0.7 + (attempt * 0.1),
                )

                content = llm_result["content"]

                # Validate content
                validation = self.llm_client.validate_content(content, content_type)

                if validation["valid"]:
                    # Analyze sentiment
                    sentiment = await self.llm_client.analyze_sentiment(content)

                    return {
                        "content": content,
                        "valid": True,
                        "theme": theme,
                        "context": context,
                        "word_count": validation["word_count"],
                        "character_count": validation["character_count"],
                        "sentiment_score": sentiment.get("sentiment_score", 0.0),
                        "sentiment_label": sentiment.get("sentiment_label", "neutral"),
                        "model_used": llm_result.get("model", "unknown"),
                        "attempt": attempt + 1,
                        "generated_at": datetime.now().isoformat(),
                    }
                else:
                    print(
                        f"WARNING: Content validation failed (attempt {attempt + 1}): {validation['issues']}"
                    )
                    if attempt == max_attempts - 1:
                        return {
                            "content": content,
                            "valid": False,
                            "issues": validation["issues"],
                            "attempt": attempt + 1,
                        }

            except Exception as e:
                print(f"ERROR: Content generation attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise

                await asyncio.sleep(2**attempt)  # Exponential backoff

        return {"content": "", "valid": False, "attempt": max_attempts}


class ContentOutputter:
    """Helper class for outputting content in various formats."""

    @staticmethod
    def save_to_json(content_list: List[Dict[str, Any]], filename: str) -> None:
        """Save content to JSON file."""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(content_list, f, indent=2, ensure_ascii=False)

        print(f"SAVED: {len(content_list)} items to {output_path}")

    @staticmethod
    def save_to_csv(content_list: List[Dict[str, Any]], filename: str) -> None:
        """Save content to CSV file."""
        if not content_list:
            return

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = content_list[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(content_list)

        print(f"SAVED: {len(content_list)} items to {output_path}")

    @staticmethod
    def save_to_text(content_list: List[Dict[str, Any]], filename: str) -> None:
        """Save content to plain text file (content only)."""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(content_list, 1):
                if item.get("valid", True):  # Only save valid content
                    f.write(f"{i}. {item['content']}\n\n")

        valid_count = sum(1 for item in content_list if item.get("valid", True))
        print(f"SAVED: {valid_count} valid items to {output_path}")

    @staticmethod
    def print_preview(content_list: List[Dict[str, Any]], max_items: int = 5) -> None:
        """Print a preview of generated content."""
        print("\nPREVIEW: Generated content:")
        print("=" * 60)

        for i, item in enumerate(content_list[:max_items], 1):
            status = "VALID" if item.get("valid", True) else "INVALID"
            sentiment = item.get("sentiment_label", "unknown")
            char_count = item.get("character_count", len(item.get("content", "")))

            print(f"[{status}] {i}. [{sentiment.upper()}, {char_count} chars]")
            print(f"   {item.get('content', 'No content')}")

            if "theme" in item:
                print(f"   Theme: {item['theme']}")

            print()

        if len(content_list) > max_items:
            print(f"... and {len(content_list) - max_items} more items")

        valid_count = sum(1 for item in content_list if item.get("valid", True))
        print(f"\nSUMMARY: {valid_count}/{len(content_list)} valid items generated")


def get_output_filename(content_type: str, file_format: str, theme: str = None) -> str:
    """Generate output filename based on parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if theme:
        base_name = f"{content_type}_{theme}_{timestamp}"
    else:
        base_name = f"{content_type}_{timestamp}"

    return f"output/{base_name}.{file_format}"
