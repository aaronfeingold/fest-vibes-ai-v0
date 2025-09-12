"""Route calculation utilities for venue-to-venue navigation and optimization."""

import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from src.database.rag_manager import ScheduleType
from loguru import logger


class TransportMode(Enum):
    """Transportation modes for route calculation."""

    WALKING = "walking"
    TRANSIT = "transit"
    RIDESHARE = "rideshare"
    BIKING = "biking"


@dataclass
class RouteSegment:
    """A single segment of a route between two venues."""

    from_venue_id: int
    to_venue_id: int
    from_venue_name: str
    to_venue_name: str
    from_address: str
    to_address: str
    distance_miles: float
    travel_time_minutes: int
    transport_mode: TransportMode
    instructions: str
    estimated_cost: Optional[float] = None  # For rideshare


@dataclass
class VenueCoordinate:
    """Venue coordinate information."""

    venue_id: int
    name: str
    address: str
    latitude: float
    longitude: float


@dataclass
class OptimizedRoute:
    """A complete optimized route through multiple venues."""

    venue_ids: List[int]
    coordinates: List[VenueCoordinate]
    segments: List[RouteSegment]
    total_distance_miles: float
    total_travel_time_minutes: int
    total_estimated_cost: float
    recommended_transport_modes: List[TransportMode]
    route_efficiency_score: float  # 0-1, higher is better
    route_type: ScheduleType


class RouteCalculator:
    """Calculate optimal routes between venues for event hopping."""

    # New Orleans specific constants
    AVERAGE_WALKING_SPEED_MPH = 3.0  # Conservative for city walking
    AVERAGE_BIKING_SPEED_MPH = 8.0
    AVERAGE_TRANSIT_SPEED_MPH = 12.0  # Including wait time
    AVERAGE_RIDESHARE_SPEED_MPH = 15.0

    # Cost estimates (in USD)
    RIDESHARE_BASE_COST = 3.50
    RIDESHARE_PER_MILE = 1.80
    TRANSIT_FARE = 1.25
    BIKING_COST_PER_MILE = 0.0  # Assuming personal bike

    # Distance thresholds for mode recommendations
    WALKING_MAX_COMFORTABLE_MILES = 0.8
    BIKING_MAX_COMFORTABLE_MILES = 3.0
    TRANSIT_MIN_EFFICIENT_MILES = 0.5

    def __init__(self):
        self.logger = logger

    def calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate the great circle distance between two points in miles."""
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

    def estimate_travel_time(
        self, distance_miles: float, transport_mode: TransportMode
    ) -> int:
        """Estimate travel time in minutes for given distance and transport mode."""
        speed_map = {
            TransportMode.WALKING: self.AVERAGE_WALKING_SPEED_MPH,
            TransportMode.BIKING: self.AVERAGE_BIKING_SPEED_MPH,
            TransportMode.TRANSIT: self.AVERAGE_TRANSIT_SPEED_MPH,
            TransportMode.RIDESHARE: self.AVERAGE_RIDESHARE_SPEED_MPH,
        }

        speed = speed_map.get(transport_mode, self.AVERAGE_WALKING_SPEED_MPH)
        travel_time = (distance_miles / speed) * 60  # Convert to minutes

        # Add mode-specific overhead
        if transport_mode == TransportMode.TRANSIT:
            travel_time += 10  # Average wait time
        elif transport_mode == TransportMode.RIDESHARE:
            travel_time += 5  # Pickup time

        return int(travel_time)

    def estimate_cost(
        self, distance_miles: float, transport_mode: TransportMode
    ) -> float:
        """Estimate cost for given distance and transport mode."""
        if transport_mode == TransportMode.WALKING:
            return 0.0
        elif transport_mode == TransportMode.BIKING:
            return distance_miles * self.BIKING_COST_PER_MILE
        elif transport_mode == TransportMode.TRANSIT:
            return self.TRANSIT_FARE
        elif transport_mode == TransportMode.RIDESHARE:
            return self.RIDESHARE_BASE_COST + (distance_miles * self.RIDESHARE_PER_MILE)
        else:
            return 0.0

    def recommend_transport_mode(
        self,
        distance_miles: float,
        time_available_minutes: Optional[int] = None,
        budget_preference: str = "moderate",  # "budget", "moderate", "convenience"
    ) -> TransportMode:
        """Recommend the best transport mode for a given distance and constraints."""

        # Budget preference: prioritize walking/transit
        if budget_preference == "budget":
            if distance_miles <= self.WALKING_MAX_COMFORTABLE_MILES:
                return TransportMode.WALKING
            elif distance_miles >= self.TRANSIT_MIN_EFFICIENT_MILES:
                return TransportMode.TRANSIT
            else:
                return TransportMode.WALKING

        # Convenience preference: prioritize speed
        elif budget_preference == "convenience":
            if distance_miles <= 0.3:  # Very short distance
                return TransportMode.WALKING
            elif distance_miles <= 2.0:
                return TransportMode.RIDESHARE
            else:
                return TransportMode.RIDESHARE

        # Moderate preference: balance of cost, time, and convenience
        else:
            if distance_miles <= self.WALKING_MAX_COMFORTABLE_MILES:
                return TransportMode.WALKING
            elif distance_miles <= self.BIKING_MAX_COMFORTABLE_MILES:
                return TransportMode.BIKING
            else:
                return TransportMode.RIDESHARE

    def generate_route_instructions(self, segment: RouteSegment) -> str:
        """Generate human-readable route instructions."""
        if segment.transport_mode == TransportMode.WALKING:
            if segment.distance_miles <= 0.2:
                return f"Quick {int(segment.distance_miles * 5280)}ft walk to {segment.to_venue_name}"
            else:
                return f"{segment.travel_time_minutes}min walk ({segment.distance_miles:.1f}mi) to {segment.to_venue_name}"

        elif segment.transport_mode == TransportMode.BIKING:
            return (
                f"{segment.travel_time_minutes}min bike ride to {segment.to_venue_name}"
            )

        elif segment.transport_mode == TransportMode.TRANSIT:
            return f"Take transit to {segment.to_venue_name} (~{segment.travel_time_minutes}min)"

        elif segment.transport_mode == TransportMode.RIDESHARE:
            cost_str = (
                f"${segment.estimated_cost:.2f}" if segment.estimated_cost else ""
            )
            return f"Rideshare to {segment.to_venue_name} (~{segment.travel_time_minutes}min, {cost_str})"

        else:
            return f"Head to {segment.to_venue_name}"

    def calculate_route_segment(
        self,
        from_venue: VenueCoordinate,
        to_venue: VenueCoordinate,
        transport_mode: Optional[TransportMode] = None,
        budget_preference: str = "moderate",
    ) -> RouteSegment:
        """Calculate a single route segment between two venues."""

        distance = self.calculate_distance(
            from_venue.latitude,
            from_venue.longitude,
            to_venue.latitude,
            to_venue.longitude,
        )

        # Determine transport mode if not specified
        if transport_mode is None:
            transport_mode = self.recommend_transport_mode(
                distance, budget_preference=budget_preference
            )

        travel_time = self.estimate_travel_time(distance, transport_mode)
        cost = self.estimate_cost(distance, transport_mode)

        segment = RouteSegment(
            from_venue_id=from_venue.venue_id,
            to_venue_id=to_venue.venue_id,
            from_venue_name=from_venue.name,
            to_venue_name=to_venue.name,
            from_address=from_venue.address,
            to_address=to_venue.address,
            distance_miles=distance,
            travel_time_minutes=travel_time,
            transport_mode=transport_mode,
            instructions="",
            estimated_cost=cost,
        )

        segment.instructions = self.generate_route_instructions(segment)
        return segment

    def optimize_venue_order(
        self,
        venues: List[VenueCoordinate],
        event_times: List[datetime],
        optimization_type: str = "distance",  # "distance", "time", "cost"
    ) -> List[int]:
        """
        Optimize the order of visiting venues using a simple greedy algorithm.
        Returns list of venue_ids in optimized order.
        """
        if len(venues) <= 2:
            return [v.venue_id for v in venues]

        # Create venue lookup
        venue_lookup = {v.venue_id: v for v in venues}
        time_lookup = {venues[i].venue_id: event_times[i] for i in range(len(venues))}

        # Start with the earliest event
        sorted_by_time = sorted(venues, key=lambda v: time_lookup[v.venue_id])
        optimized_order = [sorted_by_time[0].venue_id]
        remaining_venues = sorted_by_time[1:]

        # Greedy selection: pick next venue that optimizes the chosen metric
        while remaining_venues:
            current_venue = venue_lookup[optimized_order[-1]]
            best_next = None
            best_score = float("inf")

            for venue in remaining_venues:
                # Only consider venues with events after current time
                current_time = time_lookup[current_venue.venue_id]
                venue_time = time_lookup[venue.venue_id]

                if venue_time <= current_time:
                    continue

                distance = self.calculate_distance(
                    current_venue.latitude,
                    current_venue.longitude,
                    venue.latitude,
                    venue.longitude,
                )

                if optimization_type == "distance":
                    score = distance
                elif optimization_type == "time":
                    transport_mode = self.recommend_transport_mode(distance)
                    score = self.estimate_travel_time(distance, transport_mode)
                elif optimization_type == "cost":
                    transport_mode = self.recommend_transport_mode(distance)
                    score = self.estimate_cost(distance, transport_mode)
                else:
                    score = distance

                if score < best_score:
                    best_score = score
                    best_next = venue

            if best_next:
                optimized_order.append(best_next.venue_id)
                remaining_venues.remove(best_next)
            else:
                # Add remaining venues in time order if no valid next venue
                remaining_venues.sort(key=lambda v: time_lookup[v.venue_id])
                optimized_order.extend([v.venue_id for v in remaining_venues])
                break

        return optimized_order

    def calculate_complete_route(
        self,
        venues: List[VenueCoordinate],
        event_times: Optional[List[datetime]] = None,
        optimization_type: str = "distance",
        budget_preference: str = "moderate",
    ) -> OptimizedRoute:
        """Calculate a complete optimized route through multiple venues."""

        if len(venues) < 2:
            raise ValueError("Need at least 2 venues for route calculation")

        # Use current time for all events if not specified
        if event_times is None:
            base_time = datetime.now()
            event_times = [
                base_time + timedelta(hours=i * 2) for i in range(len(venues))
            ]

        # Optimize venue order
        optimized_venue_ids = self.optimize_venue_order(
            venues, event_times, optimization_type
        )

        # Create venue lookup for ordered calculation
        venue_lookup = {v.venue_id: v for v in venues}
        ordered_venues = [venue_lookup[vid] for vid in optimized_venue_ids]

        # Calculate route segments
        segments = []
        total_distance = 0.0
        total_time = 0
        total_cost = 0.0
        transport_modes = []

        for i in range(len(ordered_venues) - 1):
            segment = self.calculate_route_segment(
                ordered_venues[i],
                ordered_venues[i + 1],
                budget_preference=budget_preference,
            )

            segments.append(segment)
            total_distance += segment.distance_miles
            total_time += segment.travel_time_minutes
            total_cost += segment.estimated_cost or 0.0
            transport_modes.append(segment.transport_mode)

        # Calculate efficiency score (lower distance/time = higher efficiency)
        efficiency_score = 1.0 / (1.0 + total_distance + (total_time / 60.0))

        return OptimizedRoute(
            venue_ids=optimized_venue_ids,
            segments=segments,
            total_distance_miles=total_distance,
            total_travel_time_minutes=total_time,
            total_estimated_cost=total_cost,
            recommended_transport_modes=list(set(transport_modes)),
            route_efficiency_score=efficiency_score,
        )

    def generate_route_summary(self, route: OptimizedRoute) -> str:
        """Generate a human-readable summary of the route for social media."""

        if not route.segments:
            return "No route available"
        logger.info(f"Generating route summary for {route.route_type} route")
        # Create concise summary
        venue_count = len(route.venue_ids)
        distance_str = f"{route.total_distance_miles:.1f}mi"

        # Determine primary transport mode
        mode_counts = {}
        for segment in route.segments:
            mode = segment.transport_mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        primary_mode = max(mode_counts.items(), key=lambda x: x[1])[0]

        # Generate descriptive text
        if primary_mode == TransportMode.WALKING:
            mobility_desc = "walkable"
        elif primary_mode == TransportMode.BIKING:
            mobility_desc = "bike-friendly"
        elif primary_mode == TransportMode.TRANSIT:
            mobility_desc = "transit-connected"
        else:
            mobility_desc = "rideshare route"

        # Cost information
        if route.total_estimated_cost <= 5.0:
            cost_desc = "budget-friendly"
        elif route.total_estimated_cost <= 15.0:
            cost_desc = "moderate cost"
        else:
            cost_desc = f"${route.total_estimated_cost:.0f}"

        # Time information
        if route.total_travel_time_minutes <= 30:
            time_desc = "quick hops"
        elif route.total_travel_time_minutes <= 60:
            time_desc = "smooth transitions"
        else:
            time_desc = f"{route.total_travel_time_minutes}min total travel"

        return f"{venue_count}-venue {distance_str} {mobility_desc} route, {cost_desc}, {time_desc}"

    def get_route_turn_by_turn(self, route: OptimizedRoute) -> List[str]:
        """Get turn-by-turn instructions for the route."""
        instructions = []

        for i, segment in enumerate(route.segments):
            step_num = i + 1
            instructions.append(f"{step_num}. {segment.instructions}")

        return instructions
