"""RAG-specific prompts for schedule generation and event interpretation."""

from typing import Dict, List, Any
from datetime import datetime


class RAGPrompts:
    """Collection of prompts for RAG-enhanced content generation."""

    # Base context for all RAG-generated content
    BASE_CONTEXT = """
    You are writing for Fest Vibes NOLA, a Gen-Z nightlife compass that helps 18-25 year olds
    create their own decentralized music fest experience in New Orleans. Your task is to take
    real event data and present it as an enticing, curated schedule that showcases our app's
    core functionality: intelligent event discovery and route optimization.

    Brand Voice:
    - Conversational, first/second-person
    - No emojis, minimal hashtags
    - Specific to NOLA culture (brass, bounce, funk, zydeco, Frenchmen, Bywater, etc.)
    - Never imply under-21 drinking; focus on music and venues
    - Be curious, warm, a little mischievous, absolutely local
    - Tweets must be ≤280 chars, prefer 160-240
    """

    @staticmethod
    def _format_time_string(performance_time: str) -> str:
        """Extract time string from ISO datetime or return TBA."""
        return performance_time.split("T")[1][:5] if performance_time else "TBA"

    @staticmethod
    def _format_event_list(
        events: List[Dict[str, Any]], include_genres: bool = True
    ) -> str:
        """Format events into a readable list with consistent styling."""
        event_descriptions = []
        for event in events:
            time_str = RAGPrompts._format_time_string(event.get("performance_time", ""))
            venue = event.get("venue_name", "Unknown Venue")
            artist = event.get("artist_name", "Live Music")

            if include_genres:
                genres = (
                    ", ".join(event.get("genres", [])[:2])
                    if event.get("genres")
                    else "music"
                )
                event_descriptions.append(f"{time_str} {artist} at {venue} ({genres})")
            else:
                event_descriptions.append(f"{time_str} {artist} @ {venue}")

        return "\n".join(event_descriptions)

    @staticmethod
    def _format_venue_arrow_list(events: List[Dict[str, Any]]) -> str:
        """Format venues as arrow-separated list for route display."""
        venues = []
        for event in events:
            venue = event.get("venue_name", "Unknown Venue")
            venues.append(venue)
        return " → ".join(venues)

    @staticmethod
    def _extract_route_info(route_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized route information for consistent display."""
        return {
            "distance": route_info.get("total_distance_miles", 0),
            "travel_time": route_info.get("total_travel_time_minutes", 0),
            "cost": route_info.get("total_estimated_cost", 0),
            "summary": route_info.get("route_summary", ""),
            "transport_modes": route_info.get("recommended_transport_modes", []),
        }

    @staticmethod
    def _get_optimization_message(
        events: List[Dict[str, Any]],
        route_info: Dict[str, Any],
        focus: str = "distance",
    ) -> str:
        """Generate optimization message based on focus type."""
        event_count = len(events)
        route_data = RAGPrompts._extract_route_info(route_info)

        if focus == "distance":
            return f"{event_count} venues, {route_data['distance']:.1f} miles total"
        elif focus == "time":
            return f"{route_data['travel_time']} minutes between all venues"
        elif focus == "cost":
            return f"${route_data['cost']:.2f} total, budget-optimized route"
        else:
            return f"{event_count}-stop route, optimized for efficiency"

    @staticmethod
    def _format_timed_schedule(
        events: List[Dict[str, Any]], route_info: Dict[str, Any]
    ) -> str:
        """Format events with detailed timing gaps and travel calculations."""
        # Calculate time gaps between events
        time_gaps = []
        for i in range(len(events) - 1):
            current_time = datetime.fromisoformat(events[i].get("performance_time", ""))
            next_time = datetime.fromisoformat(
                events[i + 1].get("performance_time", "")
            )
            gap_minutes = int((next_time - current_time).total_seconds() / 60)
            time_gaps.append(gap_minutes)

        # Format event schedule with gaps
        schedule_parts = []
        for i, event in enumerate(events):
            time_str = RAGPrompts._format_time_string(event.get("performance_time", ""))
            venue = event.get("venue_name", "Unknown Venue")
            artist = event.get("artist_name", "Live Music")
            schedule_parts.append(f"{time_str} {artist} @ {venue}")

            if i < len(time_gaps):
                gap = time_gaps[i]
                travel_time = (
                    route_info.get("segments", [{}])[i].get("travel_time_minutes", 0)
                    if route_info.get("segments")
                    else 0
                )
                buffer_time = gap - travel_time
                if buffer_time > 0:
                    schedule_parts.append(
                        f"  ({buffer_time}min to enjoy + {travel_time}min travel)"
                    )

        return "\n".join(schedule_parts)

    @staticmethod
    def _format_event_summary(events: List[Dict[str, Any]]) -> str:
        """Create a brief summary of events with count and genres."""
        event_summary = f"{len(events)} events"
        if events:
            genres = set()
            for event in events:
                genres.update(event.get("genres", []))
            if genres:
                genre_list = ", ".join(list(genres)[:2])
                event_summary += f" ({genre_list})"
        return event_summary

    @staticmethod
    def _format_preview_details(
        events: List[Dict[str, Any]], route_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and format all details needed for preview demo prompts."""
        route_data = RAGPrompts._extract_route_info(route_info)
        venue_count = len(events)

        # Extract time span
        start_time = (
            RAGPrompts._format_time_string(events[0].get("performance_time", ""))
            if events
            else "TBA"
        )
        end_time = (
            RAGPrompts._format_time_string(events[-1].get("performance_time", ""))
            if events
            else "TBA"
        )

        # Extract genres and venues
        genres = set()
        venues = []
        for event in events:
            genres.update(event.get("genres", []))
            venues.append(event.get("venue_name", "Unknown"))

        genre_text = ", ".join(list(genres)[:3]) if genres else "music"
        venue_list = " → ".join(venues)

        return {
            "venue_count": venue_count,
            "venue_list": venue_list,
            "start_time": start_time,
            "end_time": end_time,
            "distance": route_data["distance"],
            "travel_time": route_data["travel_time"],
            "genre_text": genre_text,
        }

    @staticmethod
    def get_schedule_generation_prompt(
        events: List[Dict[str, Any]],
        route_info: Dict[str, Any],
        schedule_type: str = "demo",
    ) -> str:
        """Generate prompt for creating a schedule-based tweet from real event data."""

        events_text = RAGPrompts._format_event_list(events, include_genres=True)
        route_data = RAGPrompts._extract_route_info(route_info)

        prompt = f"""
        {RAGPrompts.BASE_CONTEXT}

        Task: Create a tweet that showcases our app's schedule-building capability using this REAL event data:

        TONIGHT'S EVENTS:
        {events_text}

        ROUTE OPTIMIZATION:
        - Total distance: {route_data['distance']:.1f} miles
        - Travel time: {route_data['travel_time']} minutes
        - Estimated cost: ${route_data['cost']:.2f}
        - Route type: {route_data['summary']}

        Create a tweet that:
        1. Presents this as a curated "mini fest" schedule
        2. Highlights the route efficiency (walkable, quick hops, etc.)
        3. Includes a soft CTA for engagement
        4. Shows our app's smart planning in action

        Examples of the style we want:
        - "Tonight's brass crawl: 7:30 Preservation Hall → 9:45 d.b.a. → 12:15 Spotted Cat. \
1.2 miles total, all walkable. Want us to ping you for transitions?"
        - "Saturday bounce lineup: 8pm Saturn Bar, 10:30 Gasa Gasa, midnight The Saint. \
Each venue under 15 mins apart. Reply ROUTE for walking directions."

        Output only the tweet text, no explanation.
        """

        return prompt

    @staticmethod
    def get_genre_focus_prompt(
        events: List[Dict[str, Any]], primary_genre: str, route_info: Dict[str, Any]
    ) -> str:
        """Generate prompt for genre-focused schedule tweets."""

        events_text = RAGPrompts._format_event_list(events, include_genres=False)
        route_data = RAGPrompts._extract_route_info(route_info)

        prompt = f"""
        {RAGPrompts.BASE_CONTEXT}

        Task: Create a genre-focused tweet showcasing our {primary_genre} event curation:

        {primary_genre.upper()} EVENTS TONIGHT:
        {events_text}

        ROUTE: {route_data['summary']}

        Create a tweet that:
        1. Leads with the genre as the organizing theme
        2. Shows the curated progression through NOLA's {primary_genre} scene
        3. Mentions route efficiency
        4. Includes a CTA related to genre preferences

        Style examples:
        - "Funk night done right: 8pm Tip's House → 10:30 Gasa Gasa → 12:45 Hi-Ho. \
2.1 miles of pure groove. What's your go-to funk progression?"
        - "Brass line starts at 7: Preservation Hall → French Market → Spotted Cat. \
Walking the tradition, venue by venue. Reply BRASS for more lineups."

        Output only the tweet text.
        """

        return prompt

    @staticmethod
    def get_route_optimization_prompt(
        events: List[Dict[str, Any]],
        route_info: Dict[str, Any],
        optimization_focus: str = "distance",
    ) -> str:
        """Generate prompt emphasizing our route optimization capabilities."""

        venues_text = RAGPrompts._format_venue_arrow_list(events)
        optimization_message = RAGPrompts._get_optimization_message(
            events, route_info, optimization_focus
        )

        prompt = f"""
        {RAGPrompts.BASE_CONTEXT}

        Task: Create a tweet that highlights our smart route optimization:

        OPTIMIZED ROUTE: {venues_text}
        EFFICIENCY: {optimization_message}

        Create a tweet that:
        1. Shows the route as an achievement of smart planning
        2. Emphasizes the optimization (distance/time/cost efficiency)
        3. Positions this as a preview of our app's capabilities
        4. Includes a CTA about route planning

        Style examples:
        - "We mapped the perfect 3-venue to end you night uptown: starting on Poland → Marigny → CBD. \
4.9 miles, 47 minutes total travel. This is what experiencing New Orleans looks like."
        - "Optimized your Saturday: 4 venues loop, 2.3 miles \
No Dice → Maison → Negril → Kajun's. Want us to build your route?"

        Output only the tweet text.
        """

        return prompt

    @staticmethod
    def get_neighborhood_focus_prompt(
        events: List[Dict[str, Any]],
        primary_neighborhood: str,
        route_info: Dict[str, Any],
    ) -> str:
        """Generate prompt for neighborhood-focused event curation."""

        # Create custom event format for neighborhood focus (includes first genre)
        event_descriptions = []
        for event in events:
            time_str = RAGPrompts._format_time_string(event.get("performance_time", ""))
            venue = event.get("venue_name", "Unknown Venue")
            artist = event.get("artist_name", "Live Music")
            genres = event.get("genres", [])
            genre_str = f"({genres[0]})" if genres else ""
            event_descriptions.append(f"{time_str} {artist} @ {venue} {genre_str}")

        events_text = "\n".join(event_descriptions)
        route_data = RAGPrompts._extract_route_info(route_info)

        prompt = f"""
        {RAGPrompts.BASE_CONTEXT}

        Task: Create a neighborhood-focused tweet showcasing {primary_neighborhood}'s music scene:

        {primary_neighborhood.upper()} TONIGHT:
        {events_text}

        ROUTE: {route_data['distance']:.1f} miles within {primary_neighborhood}

        Create a tweet that:
        1. Celebrates the neighborhood's unique music character
        2. Shows how we curate hyper-local experiences
        3. Emphasizes the contained, walkable nature of the route
        4. Includes a CTA about neighborhood discovery

        Style examples:
        - "Frenchmen crawl: 7pm Spotted Cat → 9:30 Snug Harbor → 11:45 d.b.a. \
Never leaving the strip, maximum music density. What's good?"
        - "Bywater nights: three venues, 0.8 miles, pure local flavor. \
The Country Club → Saturn Bar → Hi-Ho. This is how you do a micro-scene deep dive."

        Output only the tweet text.
        """

        return prompt

    @staticmethod
    def get_time_optimization_prompt(
        events: List[Dict[str, Any]], route_info: Dict[str, Any]
    ) -> str:
        """Generate prompt emphasizing perfect timing and transitions."""

        schedule_text = RAGPrompts._format_timed_schedule(events, route_info)

        prompt = f"""
        {RAGPrompts.BASE_CONTEXT}

        Task: Create a tweet that showcases perfect timing optimization:

        TIMED SCHEDULE:
        {schedule_text}

        Create a tweet that:
        1. Emphasizes the precise timing and smooth transitions
        2. Shows how we calculate optimal gaps between venues
        3. Demonstrates the app's temporal intelligence
        4. Includes a CTA about timing optimization

        Style examples:
        - "Perfectly timed tonight: catch the 8pm set, 25 minutes to savor, \
10-minute walk to the 9pm show. We calculated every transition."
        - "No rushing, no waiting: 7:30 show + 40min window + 15min walk = \
9pm arrival at the next venue. Just sauntering around New Orleans."

        Output only the tweet text.
        """

        return prompt

    @staticmethod
    def get_contextual_comment_prompt(
        events: List[Dict[str, Any]],
        original_tweet_context: str,
        route_info: Dict[str, Any],
    ) -> str:
        """Generate prompt for commenting on other tweets with our event data."""

        event_summary = RAGPrompts._format_event_summary(events)
        route_data = RAGPrompts._extract_route_info(route_info)

        prompt = f"""
        {RAGPrompts.BASE_CONTEXT}

        Task: Create a helpful reply that offers relevant event suggestions based on someone's tweet.

        ORIGINAL TWEET: "{original_tweet_context}"

        OUR EVENT DATA: {event_summary}, {route_data['distance']:.1f}mi route, \
{route_data['summary']}

        Create a reply that:
        1. Acknowledges their interest/question
        2. Offers specific, relevant event suggestions from our data
        3. Briefly mentions route planning capability
        4. Stays conversational and helpful, not promotional

        Keep under 100 characters if possible. Examples:
        - "We found 3 brass shows within 1.2 miles tonight. Want the lineup?"
        - "There's a funk progression through Bywater starting at 8. \
Mapped and timed."

        Output only the reply text.
        """

        return prompt

    @staticmethod
    def get_preview_demo_prompt(
        events: List[Dict[str, Any]], route_info: Dict[str, Any]
    ) -> str:
        """Generate prompt for 'preview' tweets that demo the app functionality."""

        schedule_details = RAGPrompts._format_preview_details(events, route_info)

        prompt = f"""
        {RAGPrompts.BASE_CONTEXT}

        Task: Create a "preview" tweet that shows what our app does by presenting a real schedule:

        GENERATED SCHEDULE:
        - {schedule_details['venue_count']} venues: {schedule_details['venue_list']}
        - Time span: {schedule_details['start_time']} to {schedule_details['end_time']}
        - Route: {schedule_details['distance']:.1f} miles, {schedule_details['travel_time']} minutes travel
        - Genres: {schedule_details['genre_text']}

        Create a tweet that:
        1. Frames this as "here's what we built for tonight"
        2. Shows the comprehensive planning (venues, timing, routing)
        3. Positions this as a preview of app functionality
        4. Includes a CTA about trying the full experience

        Style examples:
        - "Preview: Tonight's generated fest route. 4 venues, 90 minutes of music, \
1.8 miles optimized. This is what planning looks like when AI does the work."
        - "Built you a {schedule_details['genre_text']} crawl: \
{schedule_details['start_time']}-{schedule_details['end_time']}, \
{schedule_details['distance']:.1f}mi route, zero backtracking. \
Ready to see what else we can map?"

        Output only the tweet text.
        """

        return prompt
