#!/usr/bin/env python3
"""
Standalone script for generating Twitter posts using AI.
Extracts content generation from the bot for manual use.

Usage:
    python scripts/generate_posts.py --count 5
    python scripts/generate_posts.py --theme music --count 3 --format json
    python scripts/generate_posts.py --custom "Generate a post about local coffee shops"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add current directory for relative imports
sys.path.append(str(Path(__file__).parent))

from content_utils import (
    ContentGenerator,
    ContentThemes,
    ContentOutputter,
    get_output_filename,
)


async def generate_posts(
    count: int,
    theme: str = None,
    custom_context: str = None,
    output_format: str = "text",
    output_file: str = None,
    preview: bool = True,
) -> None:
    """Generate multiple posts and save them."""

    print(f"\033[92mSTARTING\033[0m Generating {count} posts...")
    if theme:
        print(f"\033[94mTHEME\033[0m: {theme}")
    if custom_context:
        print(f"\033[95mCUSTOM\033[0m: {custom_context}")

    print("\033[93mWAIT\033[0m This may take a while due to rate limiting...")
    print()

    generator = ContentGenerator()
    generated_posts = []

    for i in range(count):
        try:
            print(f"\033[96mGENERATING\033[0m post {i + 1}/{count}...")

            if custom_context:
                result = await generator.generate_post(custom_context=custom_context)
            else:
                result = await generator.generate_post(theme=theme)

            generated_posts.append(result)

            # Show progress
            if result["valid"]:
                print(
                    f"\033[92mSUCCESS\033[0m Post {i + 1}: {result['content'][:50]}..."
                )
            else:
                print(f"\033[91mFAILED\033[0m Post {i + 1}: Failed validation")

        except Exception as e:
            print(f"\033[91mERROR\033[0m generating post {i + 1}: {e}")
            continue

    print(f"\n\033[92mCOMPLETE\033[0m Generated {len(generated_posts)} posts!")

    # Show preview
    if preview and generated_posts:
        ContentOutputter.print_preview(generated_posts)

    # Save output
    if generated_posts:
        if output_file is None:
            output_file = get_output_filename("posts", output_format, theme)

        if output_format == "json":
            ContentOutputter.save_to_json(generated_posts, output_file)
        elif output_format == "csv":
            ContentOutputter.save_to_csv(generated_posts, output_file)
        elif output_format == "text":
            ContentOutputter.save_to_text(generated_posts, output_file)
        else:
            print(f"\033[91mERROR\033[0m Unknown output format: {output_format}")

    # Show valid content summary
    valid_posts = [p for p in generated_posts if p.get("valid", True)]
    print(
        f"\n\033[96mSUMMARY\033[0m: {len(valid_posts)}/{len(generated_posts)} valid posts generated"
    )

    if valid_posts:
        print("\n\033[94mREADY\033[0m for manual posting:")
        for i, post in enumerate(valid_posts[:3], 1):  # Show first 3
            print(f"  {i}. {post['content']}")

        if len(valid_posts) > 3:
            print(f"  ... and {len(valid_posts) - 3} more in output file")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate Twitter posts using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_posts.py --count 5
  python scripts/generate_posts.py --theme music --count 3
  python scripts/generate_posts.py --theme food --format json --count 10
  python scripts/generate_posts.py --custom "Post about street art in NOLA"
  python scripts/generate_posts.py --list-themes

Available themes: music, culture, food, genz, events
        """,
    )

    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=5,
        help="Number of posts to generate (default: 5)",
    )

    parser.add_argument(
        "--theme",
        "-t",
        choices=list(ContentThemes.THEMES.keys()),
        help="Specific theme to use for all posts",
    )

    parser.add_argument(
        "--custom", type=str, help="Custom context/prompt for generation"
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output filename (auto-generated if not specified)",
    )

    parser.add_argument("--no-preview", action="store_true", help="Skip preview output")

    parser.add_argument(
        "--list-themes", action="store_true", help="List available themes and exit"
    )

    args = parser.parse_args()

    # Handle list themes
    if args.list_themes:
        print("\033[96mAVAILABLE THEMES\033[0m:")
        for theme, inspirations in ContentThemes.THEMES.items():
            print(f"\n\033[94m{theme.upper()}\033[0m:")
            for inspiration in inspirations[:3]:  # Show first 3
                print(f"  • {inspiration}")
            if len(inspirations) > 3:
                print(f"  ... and {len(inspirations) - 3} more")
        return

    # Validation
    if args.count <= 0:
        print("\033[91mERROR\033[0m Count must be greater than 0")
        return

    if args.count > 20:
        print(
            "\033[93mWARNING\033[0m Generating more than 20 posts may take a very long time due to rate limiting"
        )
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Operation cancelled")
            return

    # Run generation
    try:
        asyncio.run(
            generate_posts(
                count=args.count,
                theme=args.theme,
                custom_context=args.custom,
                output_format=args.format,
                output_file=args.output,
                preview=not args.no_preview,
            )
        )
    except KeyboardInterrupt:
        print("\n\033[93mCANCELLED\033[0m Generation cancelled by user")
    except Exception as e:
        print(f"\033[91mFATAL ERROR\033[0m: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
