"""RAG (Retrieval Augmented Generation) manager for semantic event search and schedule optimization."""

import math
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import asyncpg
from loguru import logger

from src.config.settings import DatabaseConfig
from src.utils.llm_client import LLMClient


@dataclass
class EventSearchResult:
    """Result from semantic event search."""

    id: int
    title: str
    artist_name: str
    venue_name: str
    venue_id: int
    performance_time: datetime
    end_time: Optional[datetime]
    description: str
    genres: List[str]
    latitude: Optional[float]
    longitude: Optional[float]
    similarity_score: float


@dataclass
class VenueDistance:
    """Distance information between venues."""

    from_venue_id: int
    to_venue_id: int
    from_venue_name: str
    to_venue_name: str
    distance_miles: float
    walking_time_minutes: int


@dataclass
class OptimizedSchedule:
    """An optimized event schedule with routing information."""

    events: List[EventSearchResult]
    total_distance_miles: float
    total_travel_time_minutes: int
    venue_transitions: List[VenueDistance]
    schedule_type: str  # "genre_focused", "time_optimized", "distance_optimized"


class RAGManager:
    """RAG manager for semantic event search and schedule optimization."""

    def __init__(self, config: DatabaseConfig, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm_client = llm_client
        self.pool: Optional[asyncpg.Pool] = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection pool to Events PostgreSQL database."""
        try:
            # Get connection kwargs for events database (separate from bot operations)
            connection_kwargs = self.config.get_events_postgres_connection_kwargs()

            self.pool = await asyncpg.create_pool(
                self.config.events_postgres_uri,
                min_size=1,
                max_size=self.config.events_postgres_max_pool_size,
                **connection_kwargs,
            )

            # Test connection and ensure pgvector extension is enabled
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")

                # Check if pgvector is available
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                if not result:
                    raise RuntimeError("pgvector extension is not installed")

            self._connected = True
            logger.info("RAG Manager connected to PostgreSQL with pgvector")

        except Exception as e:
            logger.error(f"Failed to connect RAG Manager to PostgreSQL: {e}")
            raise

    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            self._connected = False
            logger.info("RAG Manager disconnected from PostgreSQL")

    async def search_events_by_query(
        self,
        query: str,
        days_ahead: int = 14,
        similarity_threshold: float = 0.5,
        limit: int = 20,
    ) -> List[EventSearchResult]:
        """Semantic search for events using natural language query."""
        try:
            if not self.llm_client:
                logger.warning("No LLM client available for query embedding")
                return []

            # Generate embedding for the search query
            query_embedding = await self.llm_client.generate_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate embedding for query")
                return []

            # Format embedding as string for pgvector compatibility
            # pgvector expects vector format like '[1.0,2.0,3.0]'
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            logger.debug(f"Formatted embedding string: {embedding_str[:100]}...")

            # Calculate time range
            now = datetime.utcnow()
            end_time = now + timedelta(days=days_ahead)

            async with self.pool.acquire() as conn:
                # Complex query joining events, venues, artists, and genres
                search_query = """
                    WITH event_search AS (
                        SELECT
                            e.id,
                            e.artist_name,
                            e.venue_name,
                            e.performance_time,
                            e.end_time,
                            e.description,
                            e.venue_id,
                            v.latitude,
                            v.longitude,
                            v.name as venue_full_name,
                            a.name as artist_full_name,
                            -- Calculate similarity using event, venue, artist, and genre embeddings
                            -- For genres, we'll use the maximum similarity across all genres for this event
                            GREATEST(
                                COALESCE(1 - (e.event_text_embedding <=> $1), 0),
                                COALESCE(1 - (v.venue_info_embedding <=> $1), 0),
                                COALESCE(1 - (a.description_embedding <=> $1), 0),
                                COALESCE(MAX(1 - (g.genre_embedding <=> $1)), 0)
                            ) as similarity_score,
                            -- Aggregate genres
                            ARRAY_AGG(DISTINCT g.name) as genres
                        FROM events e
                        LEFT JOIN venues v ON e.venue_id = v.id
                        LEFT JOIN artists a ON e.artist_id = a.id
                        LEFT JOIN event_genres eg ON e.id = eg.event_id
                        LEFT JOIN genres g ON eg.genre_id = g.id
                        WHERE e.performance_time BETWEEN $2 AND $3
                        AND (
                            (e.event_text_embedding IS NOT NULL AND 1 - (e.event_text_embedding <=> $1) >= $4) OR
                            (v.venue_info_embedding IS NOT NULL AND 1 - (v.venue_info_embedding <=> $1) >= $4) OR
                            (a.description_embedding IS NOT NULL AND 1 - (a.description_embedding <=> $1) >= $4) OR
                            (g.genre_embedding IS NOT NULL AND 1 - (g.genre_embedding <=> $1) >= $4)
                        )
                        GROUP BY e.id, e.artist_name, e.venue_name, e.performance_time,
                                e.end_time, e.description, e.venue_id, v.latitude, v.longitude,
                                v.name, a.name, e.event_text_embedding, v.venue_info_embedding,
                                a.description_embedding
                    )
                    SELECT * FROM event_search
                    ORDER BY similarity_score DESC, performance_time ASC
                    LIMIT $5
                """

                rows = await conn.fetch(
                    search_query,
                    embedding_str,
                    now,
                    end_time,
                    similarity_threshold,
                    limit,
                )

                results = []
                for row in rows:
                    results.append(
                        EventSearchResult(
                            id=row["id"],
                            title=f"{row['artist_name']} at {row['venue_name']}",
                            artist_name=row["artist_full_name"] or row["artist_name"],
                            venue_name=row["venue_full_name"] or row["venue_name"],
                            venue_id=row["venue_id"],
                            performance_time=row["performance_time"],
                            end_time=row["end_time"],
                            description=row["description"] or "",
                            genres=row["genres"] or [],
                            latitude=row["latitude"],
                            longitude=row["longitude"],
                            similarity_score=row["similarity_score"],
                        )
                    )

                logger.debug(f"Found {len(results)} events for query: {query}")

                # Debug: Check why no results if empty
                if not results:
                    debug_query = """
                        SELECT
                            COUNT(*) as total_events,
                            COUNT(CASE WHEN event_text_embedding IS NOT NULL THEN 1 END) as events_with_embeddings,
                            COUNT(CASE WHEN performance_time BETWEEN $1 AND $2 THEN 1 END) as events_in_timeframe
                        FROM events e
                    """
                    debug_row = await conn.fetchrow(debug_query, now, end_time)
                    logger.warning(
                        f"No semantic results for query '{query}'. Debug info: "
                        f"total_events={debug_row['total_events']}, "
                        f"events_with_embeddings={debug_row['events_with_embeddings']}, "
                        f"events_in_timeframe={debug_row['events_in_timeframe']}, "
                        f"similarity_threshold={similarity_threshold}"
                    )

                    # Fallback: Try genre-based search if semantic search fails
                    logger.info(f"Attempting fallback genre search for query: {query}")
                    fallback_results = await self._search_by_genre_fallback(
                        query, now, end_time, limit
                    )
                    if fallback_results:
                        logger.info(
                            f"Fallback genre search found {len(fallback_results)} events"
                        )
                        return fallback_results

                return results

        except Exception as e:
            logger.error(f"Failed to search events by query: {e}")
            return []

    async def debug_database_contents(self) -> Dict[str, Any]:
        """Debug method to check what's actually in the database."""
        try:
            async with self.pool.acquire() as conn:
                # Check genres
                genres = await conn.fetch("SELECT name FROM genres ORDER BY name")

                # Check events with funk genre
                funk_events = await conn.fetch(
                    """
                    SELECT e.id, e.artist_name, e.venue_name, e.performance_time, g.name as genre
                    FROM events e
                    LEFT JOIN event_genres eg ON e.id = eg.event_id
                    LEFT JOIN genres g ON eg.genre_id = g.id
                    WHERE g.name ILIKE '%funk%'
                    LIMIT 10
                """
                )

                # Check events with embeddings
                events_with_embeddings = await conn.fetchval(
                    "SELECT COUNT(*) FROM events WHERE event_text_embedding IS NOT NULL"
                )

                return {
                    "total_genres": len(genres),
                    "all_genres": [g["name"] for g in genres],
                    "funk_events_count": len(funk_events),
                    "funk_events": [dict(e) for e in funk_events],
                    "events_with_embeddings": events_with_embeddings,
                }

        except Exception as e:
            logger.error(f"Failed to debug database contents: {e}")
            return {}

    async def _search_by_genre_fallback(
        self, query: str, start_time: datetime, end_time: datetime, limit: int
    ) -> List[EventSearchResult]:
        """Fallback search using genre matching when semantic search fails."""
        try:
            async with self.pool.acquire() as conn:
                # Simple genre-based search for common queries
                search_pattern = f"%{query.lower()}%"

                genre_search_query = """
                    SELECT DISTINCT
                        e.id,
                        e.artist_name,
                        e.venue_name,
                        e.performance_time,
                        e.end_time,
                        e.description,
                        e.venue_id,
                        v.latitude,
                        v.longitude,
                        v.name as venue_full_name,
                        a.name as artist_full_name,
                        0.8 as similarity_score,  -- Default high score for direct genre matches
                        ARRAY_AGG(DISTINCT g.name) as genres
                    FROM events e
                    LEFT JOIN venues v ON e.venue_id = v.id
                    LEFT JOIN artists a ON e.artist_id = a.id
                    LEFT JOIN event_genres eg ON e.id = eg.event_id
                    LEFT JOIN genres g ON eg.genre_id = g.id
                    WHERE e.performance_time BETWEEN $1 AND $2
                    AND (
                        g.name ILIKE $4
                        OR e.description ILIKE $4
                        OR e.artist_name ILIKE $4
                    )
                    GROUP BY e.id, e.artist_name, e.venue_name, e.performance_time,
                            e.end_time, e.description, e.venue_id, v.latitude, v.longitude,
                            v.name, a.name
                    ORDER BY e.performance_time ASC
                    LIMIT $3
                """

                rows = await conn.fetch(
                    genre_search_query, start_time, end_time, limit, search_pattern
                )

                results = []
                for row in rows:
                    results.append(
                        EventSearchResult(
                            id=row["id"],
                            title=f"{row['artist_name']} at {row['venue_name']}",
                            artist_name=row["artist_name"],
                            venue_name=row["venue_name"],
                            venue_id=row["venue_id"],
                            performance_time=row["performance_time"],
                            end_time=row["end_time"],
                            description=row["description"] or "",
                            genres=row["genres"] or [],
                            latitude=row["latitude"],
                            longitude=row["longitude"],
                            similarity_score=row["similarity_score"],
                        )
                    )

                return results

        except Exception as e:
            logger.error(f"Fallback genre search failed: {e}")
            return []

    async def get_events_by_timeframe(
        self,
        start_time: datetime,
        end_time: datetime,
        genres: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[EventSearchResult]:
        """Get events within specific timeframe, optionally filtered by genres."""
        try:
            async with self.pool.acquire() as conn:
                if genres:
                    query = """
                        SELECT
                            e.id,
                            e.artist_name,
                            e.venue_name,
                            e.performance_time,
                            e.end_time,
                            e.description,
                            e.venue_id,
                            v.latitude,
                            v.longitude,
                            v.name as venue_full_name,
                            a.name as artist_full_name,
                            ARRAY_AGG(DISTINCT g.name) as genres
                        FROM events e
                        LEFT JOIN venues v ON e.venue_id = v.id
                        LEFT JOIN artists a ON e.artist_id = a.id
                        LEFT JOIN event_genres eg ON e.id = eg.event_id
                        LEFT JOIN genres g ON eg.genre_id = g.id
                        WHERE e.performance_time BETWEEN $1 AND $2
                        AND g.name = ANY($3)
                        GROUP BY e.id, e.artist_name, e.venue_name, e.performance_time,
                                e.end_time, e.description, e.venue_id, v.latitude, v.longitude,
                                v.name, a.name
                        ORDER BY e.performance_time ASC
                        LIMIT $4
                    """
                    rows = await conn.fetch(query, start_time, end_time, genres, limit)
                else:
                    query = """
                        SELECT
                            e.id,
                            e.artist_name,
                            e.venue_name,
                            e.performance_time,
                            e.end_time,
                            e.description,
                            e.venue_id,
                            v.latitude,
                            v.longitude,
                            v.name as venue_full_name,
                            a.name as artist_full_name,
                            ARRAY_AGG(DISTINCT g.name) as genres
                        FROM events e
                        LEFT JOIN venues v ON e.venue_id = v.id
                        LEFT JOIN artists a ON e.artist_id = a.id
                        LEFT JOIN event_genres eg ON e.id = eg.event_id
                        LEFT JOIN genres g ON eg.genre_id = g.id
                        WHERE e.performance_time BETWEEN $1 AND $2
                        GROUP BY e.id, e.artist_name, e.venue_name, e.performance_time,
                                e.end_time, e.description, e.venue_id, v.latitude, v.longitude,
                                v.name, a.name
                        ORDER BY e.performance_time ASC
                        LIMIT $3
                    """
                    rows = await conn.fetch(query, start_time, end_time, limit)

                results = []
                for row in rows:
                    results.append(
                        EventSearchResult(
                            id=row["id"],
                            title=f"{row['artist_name']} at {row['venue_name']}",
                            artist_name=row["artist_full_name"] or row["artist_name"],
                            venue_name=row["venue_full_name"] or row["venue_name"],
                            venue_id=row["venue_id"],
                            performance_time=row["performance_time"],
                            end_time=row["end_time"],
                            description=row["description"] or "",
                            genres=row["genres"] or [],
                            latitude=row["latitude"],
                            longitude=row["longitude"],
                            similarity_score=1.0,  # Default for non-semantic search
                        )
                    )

                return results

        except Exception as e:
            logger.error(f"Failed to get events by timeframe: {e}")
            return []

    async def calculate_venue_distances(
        self, venue_ids: List[int]
    ) -> List[VenueDistance]:
        """Calculate distances between all pairs of venues."""
        try:
            if len(venue_ids) < 2:
                return []

            async with self.pool.acquire() as conn:
                # Get venue coordinates
                query = """
                    SELECT id, name, latitude, longitude
                    FROM venues
                    WHERE id = ANY($1) AND latitude IS NOT NULL AND longitude IS NOT NULL
                """
                rows = await conn.fetch(query, venue_ids)

                venues = {row["id"]: row for row in rows}
                distances = []

                # Calculate distances between all venue pairs
                for i, from_id in enumerate(venue_ids[:-1]):
                    for to_id in venue_ids[i + 1 :]:
                        if from_id in venues and to_id in venues:
                            from_venue = venues[from_id]
                            to_venue = venues[to_id]

                            distance = self._calculate_haversine_distance(
                                from_venue["latitude"],
                                from_venue["longitude"],
                                to_venue["latitude"],
                                to_venue["longitude"],
                            )

                            walking_time = int(
                                distance * 20
                            )  # ~20 minutes per mile walking

                            distances.append(
                                VenueDistance(
                                    from_venue_id=from_id,
                                    to_venue_id=to_id,
                                    from_venue_name=from_venue["name"],
                                    to_venue_name=to_venue["name"],
                                    distance_miles=distance,
                                    walking_time_minutes=walking_time,
                                )
                            )

                return distances

        except Exception as e:
            logger.error(f"Failed to calculate venue distances: {e}")
            return []

    async def build_event_schedule(
        self,
        events: List[EventSearchResult],
        max_venues: int = 4,
        schedule_type: str = "distance_optimized",
    ) -> Optional[OptimizedSchedule]:
        """Build an optimized event schedule from available events."""
        try:
            if not events or len(events) < 2:
                return None

            # Sort events by time
            events.sort(key=lambda e: e.performance_time)

            if schedule_type == "distance_optimized":
                return await self._optimize_by_distance(events, max_venues)
            elif schedule_type == "time_optimized":
                return await self._optimize_by_time(events, max_venues)
            elif schedule_type == "genre_focused":
                return await self._optimize_by_genre(events, max_venues)
            else:
                # Default to distance optimization
                return await self._optimize_by_distance(events, max_venues)

        except Exception as e:
            logger.error(f"Failed to build event schedule: {e}")
            return None

    async def _optimize_by_distance(
        self, events: List[EventSearchResult], max_venues: int
    ) -> Optional[OptimizedSchedule]:
        """Optimize schedule to minimize travel distance."""
        if len(events) <= max_venues:
            selected_events = events
        else:
            # Simple greedy selection - start with first event, then pick nearest subsequent events
            selected_events = [events[0]]
            remaining_events = events[1:]

            while len(selected_events) < max_venues and remaining_events:
                last_venue_id = selected_events[-1].venue_id

                # Find nearest event that starts after the last selected event
                best_next = None
                min_distance = float("inf")

                for event in remaining_events:
                    if (
                        event.performance_time > selected_events[-1].performance_time
                        and event.latitude
                        and event.longitude
                        and selected_events[-1].latitude
                        and selected_events[-1].longitude
                    ):

                        distance = self._calculate_haversine_distance(
                            selected_events[-1].latitude,
                            selected_events[-1].longitude,
                            event.latitude,
                            event.longitude,
                        )

                        if distance < min_distance:
                            min_distance = distance
                            best_next = event

                if best_next:
                    selected_events.append(best_next)
                    remaining_events.remove(best_next)
                else:
                    break

        return await self._create_schedule_result(selected_events, "distance_optimized")

    async def _optimize_by_time(
        self, events: List[EventSearchResult], max_venues: int
    ) -> Optional[OptimizedSchedule]:
        """Optimize schedule for smooth time transitions."""
        # Select events with good time spacing (1.5-3 hours apart)
        selected_events = [events[0]]

        for event in events[1:]:
            if len(selected_events) >= max_venues:
                break

            last_event = selected_events[-1]
            time_diff = (
                event.performance_time - last_event.performance_time
            ).total_seconds() / 3600

            # Look for events 1.5-3 hours after the last one
            if 1.5 <= time_diff <= 3.0:
                selected_events.append(event)

        return await self._create_schedule_result(selected_events, "time_optimized")

    async def _optimize_by_genre(
        self, events: List[EventSearchResult], max_venues: int
    ) -> Optional[OptimizedSchedule]:
        """Optimize schedule for genre coherence/progression."""
        # Group events by primary genre
        genre_groups = {}
        for event in events:
            if event.genres:
                primary_genre = event.genres[0]
                if primary_genre not in genre_groups:
                    genre_groups[primary_genre] = []
                genre_groups[primary_genre].append(event)

        # Select events to create a genre journey
        selected_events = []

        # Prioritize genres with multiple events
        sorted_genres = sorted(
            genre_groups.items(), key=lambda x: len(x[1]), reverse=True
        )

        for genre, genre_events in sorted_genres:
            if len(selected_events) >= max_venues:
                break

            # Add 1-2 events from this genre, time-sorted
            genre_events.sort(key=lambda e: e.performance_time)

            for event in genre_events:
                if len(selected_events) >= max_venues:
                    break
                if (
                    not selected_events
                    or event.performance_time > selected_events[-1].performance_time
                ):
                    selected_events.append(event)

        return await self._create_schedule_result(selected_events, "genre_focused")

    async def _create_schedule_result(
        self, events: List[EventSearchResult], schedule_type: str
    ) -> OptimizedSchedule:
        """Create schedule result with distance calculations."""
        if len(events) < 2:
            return OptimizedSchedule(
                events=events,
                total_distance_miles=0.0,
                total_travel_time_minutes=0,
                venue_transitions=[],
                schedule_type=schedule_type,
            )

        venue_ids = [e.venue_id for e in events]
        all_distances = await self.calculate_venue_distances(venue_ids)

        # Calculate sequential transitions
        transitions = []
        total_distance = 0.0
        total_time = 0

        for i in range(len(events) - 1):
            from_venue = events[i].venue_id
            to_venue = events[i + 1].venue_id

            # Find distance between these venues
            distance_info = next(
                (
                    d
                    for d in all_distances
                    if (d.from_venue_id == from_venue and d.to_venue_id == to_venue)
                    or (d.from_venue_id == to_venue and d.to_venue_id == from_venue)
                ),
                None,
            )

            if distance_info:
                transitions.append(distance_info)
                total_distance += distance_info.distance_miles
                total_time += distance_info.walking_time_minutes

        return OptimizedSchedule(
            events=events,
            total_distance_miles=total_distance,
            total_travel_time_minutes=total_time,
            venue_transitions=transitions,
            schedule_type=schedule_type,
        )

    @staticmethod
    def _calculate_haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate the great circle distance between two points on Earth in miles."""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        # Radius of earth in miles
        r = 3956

        return c * r
