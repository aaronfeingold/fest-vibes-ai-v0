#!/usr/bin/env python3
"""
Test script for RAG integration with sample event data.
This script tests the RAG-powered content generation without requiring actual database connections.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add current directory for relative imports
sys.path.append(str(Path(__file__).parent.parent))

from src.database.rag_manager import EventSearchResult, OptimizedSchedule, VenueDistance
from src.utils.route_calculator import RouteCalculator, VenueCoordinate, OptimizedRoute
from src.prompts.rag_prompts import RAGPrompts


class MockRAGTester:
    """Test RAG functionality with mock data."""

    def __init__(self):
        self.route_calculator = RouteCalculator()

    def create_sample_events(self) -> List[EventSearchResult]:
        """Create sample event data for testing."""
        base_time = datetime.now() + timedelta(hours=2)

        events = [
            EventSearchResult(
                id=1,
                title="Treme Brass Band at Preservation Hall",
                artist_name="Treme Brass Band",
                venue_name="Preservation Hall",
                venue_id=1,
                performance_time=base_time,
                end_time=base_time + timedelta(hours=1),
                description="Traditional New Orleans brass band performance",
                genres=["brass", "traditional jazz"],
                latitude=29.9584,
                longitude=-90.0644,
                similarity_score=0.95
            ),
            EventSearchResult(
                id=2,
                title="The Revivalists at Tipitina's",
                artist_name="The Revivalists",
                venue_name="Tipitina's",
                venue_id=2,
                performance_time=base_time + timedelta(hours=2),
                end_time=base_time + timedelta(hours=3),
                description="Local indie rock with New Orleans flavor",
                genres=["indie rock", "alternative"],
                latitude=29.9311,
                longitude=-90.1122,
                similarity_score=0.87
            ),
            EventSearchResult(
                id=3,
                title="Big Freedia at Hi-Ho Lounge",
                artist_name="Big Freedia",
                venue_name="Hi-Ho Lounge",
                venue_id=3,
                performance_time=base_time + timedelta(hours=4),
                end_time=base_time + timedelta(hours=5),
                description="Queen of Bounce bringing the energy",
                genres=["bounce", "hip hop"],
                latitude=29.9692,
                longitude=-90.0328,
                similarity_score=0.92
            ),
            EventSearchResult(
                id=4,
                title="Galactic at The Fillmore",
                artist_name="Galactic",
                venue_name="The Fillmore",
                venue_id=4,
                performance_time=base_time + timedelta(hours=1.5),
                end_time=base_time + timedelta(hours=2.5),
                description="Funk fusion with electronic elements",
                genres=["funk", "electronic"],
                latitude=29.9465,
                longitude=-90.0782,
                similarity_score=0.89
            )
        ]

        return events

    def create_sample_venues(self, events: List[EventSearchResult]) -> List[VenueCoordinate]:
        """Convert events to venue coordinates for route calculation."""
        venues = []
        for event in events:
            venues.append(VenueCoordinate(
                venue_id=event.venue_id,
                name=event.venue_name,
                address=f"Address for {event.venue_name}",
                latitude=event.latitude,
                longitude=event.longitude
            ))
        return venues

    def create_mock_schedule(self, events: List[EventSearchResult]) -> OptimizedSchedule:
        """Create a mock optimized schedule for testing."""
        # Calculate distances between venues
        venue_transitions = []
        total_distance = 0.0
        total_time = 0

        for i in range(len(events) - 1):
            from_event = events[i]
            to_event = events[i + 1]

            # Calculate distance using Haversine formula
            distance = self.route_calculator.calculate_distance(
                from_event.latitude, from_event.longitude,
                to_event.latitude, to_event.longitude
            )

            walking_time = int(distance * 20)  # ~20 minutes per mile

            transition = VenueDistance(
                from_venue_id=from_event.venue_id,
                to_venue_id=to_event.venue_id,
                from_venue_name=from_event.venue_name,
                to_venue_name=to_event.venue_name,
                distance_miles=distance,
                walking_time_minutes=walking_time
            )

            venue_transitions.append(transition)
            total_distance += distance
            total_time += walking_time

        return OptimizedSchedule(
            events=events,
            total_distance_miles=total_distance,
            total_travel_time_minutes=total_time,
            venue_transitions=venue_transitions,
            schedule_type="distance_optimized"
        )

    async def test_route_calculation(self):
        """Test route calculation functionality."""
        print("🗺️  Testing Route Calculation...")

        events = self.create_sample_events()
        venues = self.create_sample_venues(events)
        event_times = [event.performance_time for event in events]

        # Test route optimization
        optimized_route = self.route_calculator.calculate_complete_route(
            venues=venues,
            event_times=event_times,
            optimization_type="distance",
            budget_preference="moderate"
        )

        print(f"   ✅ Route calculated: {len(optimized_route.segments)} segments")
        print(f"   📏 Total distance: {optimized_route.total_distance_miles:.2f} miles")
        print(f"   ⏱️  Total travel time: {optimized_route.total_travel_time_minutes} minutes")
        print(f"   💰 Total cost: ${optimized_route.total_estimated_cost:.2f}")

        # Test route summary generation
        summary = self.route_calculator.generate_route_summary(optimized_route)
        print(f"   📝 Route summary: {summary}")

        # Test turn-by-turn instructions
        instructions = self.route_calculator.get_route_turn_by_turn(optimized_route)
        print(f"   📍 Instructions: {len(instructions)} steps")
        for i, instruction in enumerate(instructions[:3], 1):  # Show first 3
            print(f"      {i}. {instruction}")

        return optimized_route

    async def test_prompt_generation(self):
        """Test RAG prompt generation with sample data."""
        print("\n📝 Testing Prompt Generation...")

        events = self.create_sample_events()
        schedule = self.create_mock_schedule(events)

        # Convert events to dict format for prompts
        event_dicts = []
        for event in events:
            event_dicts.append({
                "performance_time": event.performance_time.isoformat(),
                "venue_name": event.venue_name,
                "artist_name": event.artist_name,
                "genres": event.genres,
                "description": event.description
            })

        route_info = {
            "total_distance_miles": schedule.total_distance_miles,
            "total_travel_time_minutes": schedule.total_travel_time_minutes,
            "total_estimated_cost": 15.50,
            "route_summary": "walkable French Quarter route",
            "schedule_type": schedule.schedule_type
        }

        # Test different prompt types
        prompt_tests = [
            ("schedule_generation", RAGPrompts.get_schedule_generation_prompt),
            ("genre_focus", lambda ed, ri: RAGPrompts.get_genre_focus_prompt(ed, "brass", ri)),
            ("route_optimization", lambda ed, ri: RAGPrompts.get_route_optimization_prompt(ed, ri, "distance")),
            ("neighborhood_focus", lambda ed, ri: RAGPrompts.get_neighborhood_focus_prompt(ed, "French Quarter", ri)),
            ("time_optimization", RAGPrompts.get_time_optimization_prompt),
            ("preview_demo", RAGPrompts.get_preview_demo_prompt)
        ]

        for prompt_name, prompt_func in prompt_tests:
            try:
                prompt = prompt_func(event_dicts, route_info)
                print(f"   ✅ {prompt_name}: {len(prompt)} characters")

                # Show a snippet of the prompt
                snippet = prompt[:200].replace('\n', ' ')
                print(f"      Preview: {snippet}...")

            except Exception as e:
                print(f"   ❌ {prompt_name}: Failed - {e}")

    async def test_content_generation_flow(self):
        """Test the complete content generation flow."""
        print("\n🔄 Testing Complete Content Generation Flow...")

        # 1. Create sample events
        events = self.create_sample_events()
        print(f"   📅 Created {len(events)} sample events")

        # 2. Build optimized schedule
        schedule = self.create_mock_schedule(events)
        print(f"   📋 Built schedule: {schedule.schedule_type}")
        print(f"      Events: {len(schedule.events)}")
        print(f"      Distance: {schedule.total_distance_miles:.2f} miles")
        print(f"      Travel time: {schedule.total_travel_time_minutes} minutes")

        # 3. Convert to route format
        venues = self.create_sample_venues(events)
        event_times = [event.performance_time for event in events]
        route = self.route_calculator.calculate_complete_route(venues, event_times)

        # 4. Generate route summary
        route_summary = self.route_calculator.generate_route_summary(route)
        print(f"   🗺️  Route summary: {route_summary}")

        # 5. Prepare data for prompt
        event_dicts = [{
            "performance_time": event.performance_time.isoformat(),
            "venue_name": event.venue_name,
            "artist_name": event.artist_name,
            "genres": event.genres,
            "description": event.description
        } for event in events]

        route_info = {
            "total_distance_miles": route.total_distance_miles,
            "total_travel_time_minutes": route.total_travel_time_minutes,
            "total_estimated_cost": route.total_estimated_cost,
            "route_summary": route_summary,
            "schedule_type": schedule.schedule_type
        }

        # 6. Generate different types of prompts
        schedule_prompt = RAGPrompts.get_schedule_generation_prompt(event_dicts, route_info)

        print(f"   📝 Generated schedule prompt: {len(schedule_prompt)} characters")
        print("\n   🎯 Sample prompt output:")
        print("   " + "="*60)

        # Show the actual prompt that would be sent to LLM
        lines = schedule_prompt.split('\n')
        for line in lines[:15]:  # Show first 15 lines
            print(f"   {line}")
        if len(lines) > 15:
            print(f"   ... and {len(lines) - 15} more lines")

        print("   " + "="*60)

        # 7. Test contextual comment generation
        original_tweet = "Looking for some good live music tonight in NOLA!"
        comment_prompt = RAGPrompts.get_contextual_comment_prompt(
            event_dicts, original_tweet, route_info
        )
        print(f"\n   💬 Generated comment prompt: {len(comment_prompt)} characters")

    async def run_all_tests(self):
        """Run all RAG integration tests."""
        print("🚀 Starting RAG Integration Tests")
        print("="*50)

        try:
            await self.test_route_calculation()
            await self.test_prompt_generation()
            await self.test_content_generation_flow()

            print("\n" + "="*50)
            print("✅ All RAG integration tests completed successfully!")
            print("\n🎉 Ready for integration with:")
            print("   • Real PostgreSQL event database")
            print("   • LLM content generation")
            print("   • Twitter posting pipeline")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main test function."""
    tester = MockRAGTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
