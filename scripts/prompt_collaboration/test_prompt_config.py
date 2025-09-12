#!/usr/bin/env python3
"""
Test script to validate the prompt abstraction configuration.
This script validates that the prompt system is working correctly.
"""

import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.prompts.content_utility_prompts import ContentUtilityPrompts
from scripts.content_utils import ContentThemes, ResponsePatterns


def test_prompt_configuration():
    """Test the prompt configuration system."""
    print("Testing Prompt Configuration System")
    print("=" * 50)

    # Test configuration validation
    validation = ContentUtilityPrompts.validate_prompt_config()
    print(f"Configuration Valid: {validation['valid']}")

    if not validation["valid"]:
        print("Issues found:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
        return False

    print(f"Content Types: {validation['content_types']}")
    print(f"Template Types: {validation['template_types']}")
    print()

    # Test context formatting
    print("Testing Context Formatting")
    print("-" * 30)

    # Test theme-based context
    theme_context = ContentUtilityPrompts.format_context(
        "theme_based",
        theme="music",
        inspiration="Frenchmen brass pre-game then bounce afters",
    )
    print("Theme-based context:")
    print(f"  {theme_context}")
    print()

    # Test pattern-based context
    pattern_context = ContentUtilityPrompts.format_context(
        "pattern_based",
        pattern_type="supportive",
        base_response="Real. NOLA's best nights start with one good pick.",
    )
    print("Pattern-based context:")
    print(f"  {pattern_context}")
    print()

    # Test custom context
    custom_context = ContentUtilityPrompts.format_context(
        "custom", custom_context="Create a post about late-night venues in the Quarter"
    )
    print("Custom context:")
    print(f"  {custom_context}")
    print()

    # Test prompt generation
    print("Testing Prompt Generation")
    print("-" * 30)

    # Test post prompt
    post_prompt = ContentUtilityPrompts.get_brand_enhanced_prompt(
        "post", "Theme: music. Inspiration: Bywater vinyl DJ night with backyard lights"
    )
    print("Post prompt preview (first 200 chars):")
    print(f"  {post_prompt[:200]}...")
    print()

    # Test comment prompt
    comment_prompt = ContentUtilityPrompts.get_brand_enhanced_prompt(
        "comment",
        "Create a supportive comment variation of: Love this. Which neighborhood are you starting in?",
    )
    print("Comment prompt preview (first 200 chars):")
    print(f"  {comment_prompt[:200]}...")
    print()

    return True


def test_integration_with_existing_classes():
    """Test integration with existing ContentThemes and ResponsePatterns."""
    print("Testing Integration with Existing Classes")
    print("=" * 50)

    # Test theme integration
    theme, inspiration = ContentThemes.get_random_inspiration("music")
    theme_context = ContentUtilityPrompts.format_context(
        "theme_based", theme=theme, inspiration=inspiration
    )
    print(f"Random theme: {theme}")
    print(f"Inspiration: {inspiration}")
    print(f"Formatted context: {theme_context}")
    print()

    # Test response pattern integration
    pattern_type, base_response = ResponsePatterns.get_random_pattern("enthusiastic")
    pattern_context = ContentUtilityPrompts.format_context(
        "pattern_based", pattern_type=pattern_type, base_response=base_response
    )
    print(f"Random pattern: {pattern_type}")
    print(f"Base response: {base_response}")
    print(f"Formatted context: {pattern_context}")
    print()

    return True


def main():
    """Run all prompt configuration tests."""
    print("Prompt Configuration Test Suite")
    print("=" * 60)
    print()

    try:
        # Run configuration tests
        config_success = test_prompt_configuration()
        if not config_success:
            print("Configuration tests failed!")
            return 1

        # Run integration tests
        integration_success = test_integration_with_existing_classes()
        if not integration_success:
            print("Integration tests failed!")
            return 1

        print("All tests passed! Prompt abstraction is working correctly.")
        print()
        print("Summary:")
        print("  - Configuration validation: PASS")
        print("  - Context formatting: PASS")
        print("  - Prompt generation: PASS")
        print("  - Class integration: PASS")

        return 0

    except Exception as e:
        print(f"Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
