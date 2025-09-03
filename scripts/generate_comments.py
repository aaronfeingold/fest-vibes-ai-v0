#!/usr/bin/env python3
"""
Standalone script for generating Twitter comments/replies using AI.
Creates template responses you can use for manual engagement.

Usage:
    python scripts/generate_comments.py --count 5
    python scripts/generate_comments.py --pattern supportive --count 3
    python scripts/generate_comments.py --custom "Reply to a tweet about jazz music"
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add current directory for relative imports
sys.path.append(str(Path(__file__).parent))

from content_utils import (
    ContentGenerator,
    ResponsePatterns,
    ContentOutputter,
    get_output_filename,
)


async def generate_comments(
    count: int,
    pattern_type: str = None,
    custom_context: str = None,
    output_format: str = "text",
    output_file: str = None,
    preview: bool = True,
) -> None:
    """Generate multiple comments and save them."""

    print(f"\033[92mSTARTING\033[0m Generating {count} comments...")
    if pattern_type:
        print(f"\033[94mPATTERN\033[0m: {pattern_type}")
    if custom_context:
        print(f"\033[95mCUSTOM\033[0m: {custom_context}")

    print("\033[93mWAIT\033[0m This may take a while due to rate limiting...")
    print()

    generator = ContentGenerator()
    generated_comments = []

    for i in range(count):
        try:
            print(f"\033[96mGENERATING\033[0m comment {i + 1}/{count}...")

            if custom_context:
                result = await generator.generate_comment(custom_context=custom_context)
            else:
                result = await generator.generate_comment(pattern_type=pattern_type)

            generated_comments.append(result)

            # Show progress
            if result["valid"]:
                print(
                    f"\033[92mSUCCESS\033[0m Comment {i + 1}: {result['content'][:50]}..."
                )
            else:
                print(f"\033[91mFAILED\033[0m Comment {i + 1}: Failed validation")

        except Exception as e:
            print(f"\033[91mERROR\033[0m generating comment {i + 1}: {e}")
            continue

    print(f"\n\033[92mCOMPLETE\033[0m Generated {len(generated_comments)} comments!")

    # Show preview
    if preview and generated_comments:
        ContentOutputter.print_preview(generated_comments)

    # Save output
    if generated_comments:
        if output_file is None:
            output_file = get_output_filename("comments", output_format, pattern_type)

        if output_format == "json":
            ContentOutputter.save_to_json(generated_comments, output_file)
        elif output_format == "csv":
            ContentOutputter.save_to_csv(generated_comments, output_file)
        elif output_format == "text":
            ContentOutputter.save_to_text(generated_comments, output_file)
        else:
            print(f"\033[91mERROR\033[0m Unknown output format: {output_format}")

    # Show valid content summary
    valid_comments = [c for c in generated_comments if c.get("valid", True)]
    print(
        f"\n\033[96mSUMMARY\033[0m: {len(valid_comments)}/{len(generated_comments)} valid comments generated"
    )

    if valid_comments:
        print("\n\033[94mREADY\033[0m for manual engagement:")

        # Group by pattern type for better organization
        patterns = {}
        for comment in valid_comments:
            pattern = comment.get("theme", "custom")
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(comment["content"])

        for pattern, comments in patterns.items():
            print(f"\n  \033[94m{pattern.upper()}\033[0m responses:")
            for comment in comments[:2]:  # Show first 2 per pattern
                print(f"    • {comment}")

            if len(comments) > 2:
                print(f"    ... and {len(comments) - 2} more")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate Twitter comments/replies using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_comments.py --count 5
  python scripts/generate_comments.py --pattern supportive --count 3
  python scripts/generate_comments.py --pattern enthusiastic --format json
  python scripts/generate_comments.py --custom "Reply to a post about local music"
  python scripts/generate_comments.py --list-patterns

Available patterns: supportive, enthusiastic, curious, relatable
        """,
    )

    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=5,
        help="Number of comments to generate (default: 5)",
    )

    parser.add_argument(
        "--pattern",
        "-p",
        choices=list(ResponsePatterns.PATTERNS.keys()),
        help="Specific response pattern to use for all comments",
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
        "--list-patterns",
        action="store_true",
        help="List available response patterns and exit",
    )

    args = parser.parse_args()

    # Handle list patterns
    if args.list_patterns:
        print("\033[96mAVAILABLE PATTERNS\033[0m:")
        for pattern, responses in ResponsePatterns.PATTERNS.items():
            print(f"\n\033[94m{pattern.upper()}\033[0m:")
            for response in responses[:3]:  # Show first 3
                print(f"  • {response}")
            if len(responses) > 3:
                print(f"  ... and {len(responses) - 3} more")
        return

    # Validation
    if args.count <= 0:
        print("\033[91mERROR\033[0m Count must be greater than 0")
        return

    if args.count > 20:
        print(
            "\033[93mWARNING\033[0m Generating more than 20 comments may take a very long time due to rate limiting"
        )
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Operation cancelled")
            return

    # Run generation
    try:
        asyncio.run(
            generate_comments(
                count=args.count,
                pattern_type=args.pattern,
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
