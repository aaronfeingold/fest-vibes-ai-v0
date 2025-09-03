"""Content utility prompts for standalone content generation scripts.

This module contains prompt configurations specifically for the content generation
utilities in the scripts/ directory, separated from the main agent prompts but
following the same architectural patterns.
"""

from typing import Dict, Any
from src.prompts.base_prompts import BasePrompts


class ContentUtilityPrompts:
    """Prompt configurations for content utility scripts."""

    # Base system prompts for different content types
    SYSTEM_PROMPTS = {
        "post": {
            "role": (
                "You are creating content for a Twitter bot focused on "
                "New Orleans culture, music scene, and GenZ trends."
            ),
            "requirements": (
                "The content should be authentic, engaging, and reflect "
                "the vibrant culture of New Orleans."
            ),
            "format": "Create an original tweet (under 280 characters) that's conversational and engaging.",
            "focus": "Focus on local culture, music events, food, or GenZ topics.",
            "restrictions": "Avoid excessive hashtags or promotional language.",
        },
        "comment": {
            "role": (
                "You are creating content for a Twitter bot focused on "
                "New Orleans culture, music scene, and GenZ trends."
            ),
            "requirements": (
                "The content should be authentic, engaging, and reflect "
                "the vibrant culture of New Orleans."
            ),
            "format": "Create a thoughtful reply that adds value to the conversation.",
            "focus": "Be supportive, authentic, and brief. Match the tone of the original post.",
            "restrictions": "Avoid excessive hashtags or promotional language.",
        },
    }

    # Style guidelines that apply to all content types
    STYLE_GUIDELINES = {
        "tone": "Keep the tone casual, authentic, and aligned with GenZ communication style.",
        "restrictions": "Avoid corporate or overly promotional language.",
        "character_limit": 280,
        "preferred_range": "160-240 characters",
    }

    # Context templates for different generation scenarios
    CONTEXT_TEMPLATES = {
        "theme_based": "Theme: {theme}. Inspiration: {inspiration}",
        "pattern_based": "Create a {pattern_type} comment variation of: {base_response}",
        "custom": "{custom_context}",
    }

    @classmethod
    def get_post_prompt(cls, context: str) -> str:
        """Generate a complete post generation prompt."""
        system = cls.SYSTEM_PROMPTS["post"]
        style = cls.STYLE_GUIDELINES

        return f"""{system['role']} {system['requirements']}

{system['format']} {system['restrictions']}
{system['focus']}

{context}

Important: {style['tone']} {style['restrictions']}"""

    @classmethod
    def get_comment_prompt(cls, context: str) -> str:
        """Generate a complete comment generation prompt."""
        system = cls.SYSTEM_PROMPTS["comment"]
        style = cls.STYLE_GUIDELINES

        return f"""{system['role']} {system['requirements']}

{system['format']}
{system['focus']}

{context}

Important: {style['tone']} {style['restrictions']}"""

    @classmethod
    def format_context(cls, context_type: str, **kwargs) -> str:
        """Format context using predefined templates."""
        template = cls.CONTEXT_TEMPLATES.get(
            context_type, cls.CONTEXT_TEMPLATES["custom"]
        )
        return template.format(**kwargs)

    @classmethod
    def get_brand_enhanced_prompt(cls, content_type: str, context: str) -> str:
        """Get prompt enhanced with full brand voice context."""
        brand_context = BasePrompts.get_brand_voice_context()

        if content_type == "post":
            base_prompt = cls.get_post_prompt(context)
        elif content_type == "comment":
            base_prompt = cls.get_comment_prompt(context)
        else:
            raise ValueError(f"Unknown content type: {content_type}")

        return f"""{brand_context}

{base_prompt}"""

    @classmethod
    def validate_prompt_config(cls) -> Dict[str, Any]:
        """Validate the prompt configuration integrity."""
        issues = []

        # Check that all required keys exist
        required_system_keys = [
            "role",
            "requirements",
            "format",
            "focus",
            "restrictions",
        ]
        for content_type, config in cls.SYSTEM_PROMPTS.items():
            for key in required_system_keys:
                if key not in config:
                    issues.append(f"Missing {key} in {content_type} system prompt")

        # Check style guidelines
        required_style_keys = ["tone", "restrictions", "character_limit"]
        for key in required_style_keys:
            if key not in cls.STYLE_GUIDELINES:
                issues.append(f"Missing {key} in style guidelines")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "content_types": list(cls.SYSTEM_PROMPTS.keys()),
            "template_types": list(cls.CONTEXT_TEMPLATES.keys()),
        }
