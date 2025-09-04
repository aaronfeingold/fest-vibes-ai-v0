#!/usr/bin/env python3
"""
RAG-powered Twitter post generation using real event data from Neon database.
Uses semantic search and optimized scheduling to create authentic NOLA nightlife content.

Usage:
    python scripts/generate_rag_posts.py --count 5
    python scripts/generate_rag_posts.py --query "jazz brass funk tonight" --count 3
    python scripts/generate_rag_posts.py --schedule-type distance_optimized --format json
    python scripts/generate_rag_posts.py --days-ahead 7 --similarity-threshold 0.7
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import random

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import DatabaseConfig, LLMConfig, AgentConfig
from src.database.rag_manager import RAGManager, OptimizedSchedule
from src.utils.llm_client import LLMClient
from src.utils.route_calculator import RouteCalculator
from src.prompts.rag_prompts import RAGPrompts
from scripts.content_utils import ContentOutputter, get_output_filename


class RAGContentGenerator:
    """RAG-powered content generator using real event data."""

    # Default search queries for different vibes
    DEFAULT_QUERIES = [
        "jazz brass funk music tonight New Orleans",
        "bounce hip hop live music venue",
        "indie rock alternative live show NOLA",
        "electronic house dance music event",
        "acoustic singer songwriter intimate venue",
        "zydeco second line street music",
        "funk jam session late night",
        "neo soul R&B live performance",
    ]

    def __init__(self):
        """Initialize the RAG content generator."""
        self.db_config = DatabaseConfig()
        self.llm_config = LLMConfig()
        self.agents_config = AgentConfig()

        self.llm_client = LLMClient(self.llm_config)
        self.rag_manager = RAGManager(self.db_config, self.llm_client)
        self.route_calculator = RouteCalculator()

    async def connect(self):
        """Connect to the database."""
        await self.rag_manager.connect()
        print("Connected to Neon database with RAG manager")

    async def disconnect(self):
        """Disconnect from the database."""
        await self.rag_manager.disconnect()
        print("Disconnected from database")

    async def generate_rag_post(
        self,
        search_query: str = None,
        days_ahead: int = None,
        similarity_threshold: float = None,
        max_events: int = None,
        max_venues: int = None,
        schedule_type: str = None,
    ) -> Dict[str, Any]:
        """Generate a single RAG-powered post."""

        # Use defaults from config if not specified
        days_ahead = days_ahead or self.agents_config.rag_search_days_ahead
        similarity_threshold = (
            similarity_threshold or self.agents_config.rag_similarity_threshold
        )
        max_events = max_events or self.agents_config.rag_max_events_per_search
        max_venues = max_venues or self.agents_config.rag_max_venues_per_schedule
        schedule_type = schedule_type or random.choice(
            ["distance_optimized", "time_optimized", "genre_focused"]
        )

        # Use random query if none provided
        if search_query is None:
            search_query = random.choice(self.DEFAULT_QUERIES)

        try:
            print(f"Searching for events: '{search_query}'")

            # Debug: Check database contents first
            debug_info = await self.rag_manager.debug_database_contents()
            print(f"Debug - Total genres: {debug_info.get('total_genres', 0)}")
            print(f"Debug - Funk events: {debug_info.get('funk_events_count', 0)}")
            print(
                f"Debug - Events with embeddings: {debug_info.get('events_with_embeddings', 0)}"
            )
            if debug_info.get("funk_events"):
                print(f"Debug - First funk event: {debug_info['funk_events'][0]}")

            # Search for events using RAG
            events = await self.rag_manager.search_events_by_query(
                query=search_query,
                days_ahead=days_ahead,
                similarity_threshold=similarity_threshold,
                limit=max_events,
            )

            if not events or len(events) < 2:
                return {
                    "content": "",
                    "valid": False,
                    "reason": "Not enough events found for schedule generation",
                    "events_found": len(events) if events else 0,
                    "search_query": search_query,
                }

            print(f"Found {len(events)} events, building {schedule_type} schedule...")

            # Build optimized schedule
            schedule = await self.rag_manager.build_event_schedule(
                events=events, max_venues=max_venues, schedule_type=schedule_type
            )

            if not schedule or len(schedule.events) < 2:
                return {
                    "content": "",
                    "valid": False,
                    "reason": "Could not build viable schedule from events",
                    "events_found": len(events),
                    "search_query": search_query,
                }
            print(
                f"Built schedule with {len(schedule.events)} events across "
                f"{len(set(e.venue_name for e in schedule.events))} venues"
            )

            # Convert events to format for prompt generation
            event_dicts = []
            for event in schedule.events:
                event_dicts.append(
                    {
                        "performance_time": event.performance_time.isoformat(),
                        "venue_name": event.venue_name,
                        "artist_name": event.artist_name,
                        "genres": event.genres,
                        "description": event.description,
                    }
                )

            # Generate route summary
            route_summary = self.route_calculator.generate_route_summary(
                self._convert_schedule_to_route(schedule)
            )

            route_info = {
                "total_distance_miles": schedule.total_distance_miles,
                "total_travel_time_minutes": schedule.total_travel_time_minutes,
                "total_estimated_cost": 0.0,
                "route_summary": route_summary,
                "schedule_type": schedule.schedule_type,
            }

            # Choose appropriate prompt type
            prompt_type = self._choose_prompt_type(schedule)
            prompt = self._get_rag_prompt(
                prompt_type, event_dicts, route_info, schedule
            )

            print(f"Generating content using {prompt_type} prompt...")

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
                return {
                    "content": content,
                    "valid": False,
                    "reason": f"Content validation failed: {validation.get('issues')}",
                    "search_query": search_query,
                    "events_used": len(schedule.events),
                }

            # Analyze sentiment
            sentiment = await self.llm_client.analyze_sentiment(content)

            return {
                "content": content,
                "valid": True,
                "search_query": search_query,
                "schedule_type": schedule.schedule_type,
                "events_used": len(schedule.events),
                "venues_used": len(set(e.venue_name for e in schedule.events)),
                "total_distance_miles": schedule.total_distance_miles,
                "total_travel_time_minutes": schedule.total_travel_time_minutes,
                "word_count": validation["word_count"],
                "character_count": validation["character_count"],
                "sentiment_score": sentiment.get("sentiment_score", 0.0),
                "sentiment_label": sentiment.get("sentiment_label", "neutral"),
                "model_used": llm_result.get("model", "unknown"),
                "generated_at": datetime.now().isoformat(),
                "prompt_type": prompt_type,
                "route_summary": route_summary,
                "events": [
                    {
                        "artist": e.artist_name,
                        "venue": e.venue_name,
                        "time": e.performance_time.strftime("%I:%M %p"),
                        "genres": e.genres,
                    }
                    for e in schedule.events
                ],
            }

        except Exception as e:
            return {
                "content": "",
                "valid": False,
                "reason": f"Generation error: {str(e)}",
                "search_query": search_query,
            }

    def _convert_schedule_to_route(self, schedule: OptimizedSchedule):
        """Convert schedule to route format for route calculator."""
        from src.utils.route_calculator import OptimizedRoute, VenueCoordinate

        coordinates = []
        for event in schedule.events:
            if event.latitude and event.longitude:
                coordinates.append(
                    VenueCoordinate(
                        venue_id=event.venue_id,
                        venue_name=event.venue_name,
                        latitude=event.latitude,
                        longitude=event.longitude,
                    )
                )

        return OptimizedRoute(
            coordinates=coordinates,
            total_distance_miles=schedule.total_distance_miles,
            total_travel_time_minutes=schedule.total_travel_time_minutes,
            total_estimated_cost=0.0,
            route_type=schedule.schedule_type,
        )

    def _choose_prompt_type(self, schedule: OptimizedSchedule) -> str:
        """Choose the best prompt type based on schedule characteristics."""
        venue_count = len(set(e.venue_name for e in schedule.events))
        genre_diversity = len(
            set(genre for event in schedule.events for genre in event.genres)
        )

        # Choose based on schedule characteristics
        if schedule.schedule_type == "genre_focused":
            return "genre_focus"
        elif venue_count >= 4:
            return "route_optimization"
        elif schedule.total_travel_time_minutes <= 30:
            return "neighborhood_focus"
        elif genre_diversity >= 4:
            return "preview_demo"
        else:
            return "schedule_generation"

    def _get_rag_prompt(
        self,
        prompt_type: str,
        event_dicts: List[Dict],
        route_info: Dict,
        schedule: OptimizedSchedule,
    ) -> str:
        """Get the appropriate RAG prompt."""
        if prompt_type == "schedule_generation":
            return RAGPrompts.get_schedule_generation_prompt(event_dicts, route_info)
        elif prompt_type == "genre_focus":
            primary_genre = self._get_primary_genre(schedule)
            return RAGPrompts.get_genre_focus_prompt(
                event_dicts, primary_genre, route_info
            )
        elif prompt_type == "route_optimization":
            return RAGPrompts.get_route_optimization_prompt(
                event_dicts, route_info, "distance"
            )
        elif prompt_type == "neighborhood_focus":
            primary_neighborhood = self._get_primary_neighborhood(schedule)
            return RAGPrompts.get_neighborhood_focus_prompt(
                event_dicts, primary_neighborhood, route_info
            )
        elif prompt_type == "preview_demo":
            return RAGPrompts.get_preview_demo_prompt(event_dicts, route_info)
        else:
            # Default to schedule generation
            return RAGPrompts.get_schedule_generation_prompt(event_dicts, route_info)

    def _get_primary_genre(self, schedule: OptimizedSchedule) -> str:
        """Get the most common genre from the schedule."""
        genre_counts = {}
        for event in schedule.events:
            for genre in event.genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        if genre_counts:
            return max(genre_counts.keys(), key=genre_counts.get)
        return "live music"

    def _get_primary_neighborhood(self, schedule: OptimizedSchedule) -> str:
        """Extract primary neighborhood from venue names/locations."""
        # Simple heuristic - could be enhanced with actual venue location data
        neighborhood_keywords = {
            "Frenchmen": "Frenchmen",
            "French Quarter": "French Quarter",
            "Bywater": "Bywater",
            "Marigny": "Marigny",
            "Uptown": "Uptown",
            "Treme": "Treme",
            "Mid-City": "Mid-City",
        }

        venue_names = " ".join([e.venue_name for e in schedule.events]).lower()

        for keyword, neighborhood in neighborhood_keywords.items():
            if keyword.lower() in venue_names:
                return neighborhood

        return "NOLA"


async def generate_multiple_rag_posts(
    count: int,
    search_query: str = None,
    days_ahead: int = None,
    similarity_threshold: float = None,
    max_events: int = None,
    max_venues: int = None,
    schedule_type: str = None,
    output_format: str = "text",
    output_file: str = None,
    preview: bool = True,
) -> None:
    """Generate multiple RAG-powered posts."""

    print(f"Generating {count} RAG-powered posts using Neon database...")
    if search_query:
        print(f"Search query: {search_query}")
    if schedule_type:
        print(f"Schedule type: {schedule_type}")

    print("This may take a while due to database queries and rate limiting...\n")

    generator = RAGContentGenerator()
    generated_posts = []

    try:
        await generator.connect()

        for i in range(count):
            try:
                print(f"Generating RAG post {i + 1}/{count}...")

                result = await generator.generate_rag_post(
                    search_query=search_query,
                    days_ahead=days_ahead,
                    similarity_threshold=similarity_threshold,
                    max_events=max_events,
                    max_venues=max_venues,
                    schedule_type=schedule_type,
                )

                generated_posts.append(result)

                # Show progress
                if result["valid"]:
                    print(f"Post {i + 1}: {result['content'][:60]}...")
                    print(
                        f"   {result['events_used']} events, {result['venues_used']} venues, {result['schedule_type']}"
                    )
                else:
                    print(f"Post {i + 1} failed: {result['reason']}")

                print()

            except Exception as e:
                print(f"Error generating post {i + 1}: {e}")
                continue

    finally:
        await generator.disconnect()

    print(f"Generated {len(generated_posts)} posts!\n")

    # Show preview
    if preview and generated_posts:
        ContentOutputter.print_preview(generated_posts)

    # Save output
    if generated_posts:
        if output_file is None:
            theme = schedule_type or "rag"
            output_file = get_output_filename("rag_posts", output_format, theme)

        if output_format == "json":
            ContentOutputter.save_to_json(generated_posts, output_file)
        elif output_format == "csv":
            ContentOutputter.save_to_csv(generated_posts, output_file)
        elif output_format == "text":
            ContentOutputter.save_to_text(generated_posts, output_file)
        else:
            print(f"Unknown output format: {output_format}")

    # Show summary
    valid_posts = [p for p in generated_posts if p.get("valid", True)]
    print(
        f"\nFinal summary: {len(valid_posts)}/{len(generated_posts)} valid posts generated"
    )

    if valid_posts:
        print("\nReady for manual posting:")
        for i, post in enumerate(valid_posts[:3], 1):
            print(f"  {i}. {post['content']}")
            if "events_used" in post:
                print(
                    f"     [{post['events_used']} events, {post.get('schedule_type', 'unknown')} schedule]"
                )

        if len(valid_posts) > 3:
            print(f"  ... and {len(valid_posts) - 3} more in output file")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate RAG-powered Twitter posts using real event data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_rag_posts.py --count 5
  python scripts/generate_rag_posts.py --query "jazz brass funk tonight" --count 3
  python scripts/generate_rag_posts.py --schedule-type distance_optimized --format json
  python scripts/generate_rag_posts.py --days-ahead 7 --similarity-threshold 0.8

Available schedule types: distance_optimized, time_optimized, genre_focused
        """,
    )

    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=3,
        help="Number of posts to generate (default: 3)",
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Custom search query for events (e.g. 'jazz brass funk tonight')",
    )

    parser.add_argument(
        "--schedule-type",
        "-s",
        choices=["distance_optimized", "time_optimized", "genre_focused"],
        help="Type of schedule optimization to use",
    )

    parser.add_argument(
        "--days-ahead",
        "-d",
        type=int,
        help="Number of days ahead to search for events (default: from config)",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Similarity threshold for event matching (0.0-1.0, default: from config)",
    )

    parser.add_argument(
        "--max-events",
        type=int,
        help="Maximum events to consider per search (default: from config)",
    )

    parser.add_argument(
        "--max-venues",
        type=int,
        help="Maximum venues to include in schedule (default: from config)",
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

    args = parser.parse_args()

    # Validation
    if args.count <= 0:
        print("Count must be greater than 0")
        return

    if args.count > 10:
        print(
            "WARNING: Generating more than 10 RAG posts may take a very long time "
            "due to database queries and rate limiting"
        )
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Operation cancelled")
            return

    if args.similarity_threshold is not None and not (
        0.0 <= args.similarity_threshold <= 1.0
    ):
        print("Similarity threshold must be between 0.0 and 1.0")
        return

    # Run generation
    try:
        asyncio.run(
            generate_multiple_rag_posts(
                count=args.count,
                search_query=args.query,
                days_ahead=args.days_ahead,
                similarity_threshold=args.similarity_threshold,
                max_events=args.max_events,
                max_venues=args.max_venues,
                schedule_type=args.schedule_type,
                output_format=args.format,
                output_file=args.output,
                preview=not args.no_preview,
            )
        )
    except KeyboardInterrupt:
        print("\nGeneration cancelled by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
