import random
import math

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def total_distance(route):
    """Compute total distance of a given route."""
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1)) + distance(route[-1], route[0])

def hill_climb(locations, max_iterations=1000):
    """Hill Climbing algorithm to find the shortest route."""
    # Start with a random initial route
    current_route = locations[:]
    random.shuffle(current_route)
    current_distance = total_distance(current_route)

    for _ in range(max_iterations):
        # Generate a small change (swap two random locations)
        new_route = current_route[:]
        i, j = random.sample(range(len(locations)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]

        new_distance = total_distance(new_route)

        # Accept the new route if it's shorter
        if new_distance < current_distance:
            current_route, current_distance = new_route, new_distance

    return current_route, current_distance

# Example usage
locations = [(0, 0), (2, 3), (5, 4), (7, 1), (6, 7), (8, 3)]
best_route, best_distance = hill_climb(locations)

print("Optimized Route:", best_route)
print("Total Distance:", best_distance)
