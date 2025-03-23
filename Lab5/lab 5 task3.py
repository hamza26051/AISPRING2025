import random
import math

# Distance calculation
def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Total route distance
def total_distance(route):
    """Compute total distance of a given route."""
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1)) + distance(route[-1], route[0])

# Generate initial population
def generate_population(cities, population_size):
    """Create a population of random routes."""
    return [random.sample(cities, len(cities)) for _ in range(population_size)]

# Selection (tournament selection)
def select_parents(population, num_parents):
    """Select best routes based on fitness (shorter distance is better)."""
    population.sort(key=total_distance)
    return population[:num_parents]

# Crossover (Ordered Crossover - OX)
def crossover(parent1, parent2):
    """Generate offspring using ordered crossover (OX)."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start:end] = parent1[start:end]

    remaining = [city for city in parent2 if city not in child]
    for i in range(size):
        if child[i] is None:
            child[i] = remaining.pop(0)

    return child

# Mutation (Swap two cities)
def mutate(route, mutation_rate=0.2):
    """Randomly swap two cities in the route with a given mutation probability."""
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Genetic Algorithm function
def genetic_algorithm(cities, population_size=100, generations=500, mutation_rate=0.2):
    """Solve TSP using a genetic algorithm."""
    population = generate_population(cities, population_size)

    for _ in range(generations):
        parents = select_parents(population, num_parents=population_size // 2)
        offspring = [mutate(crossover(random.choice(parents), random.choice(parents)), mutation_rate)
                     for _ in range(population_size - len(parents))]
        population = parents + offspring  # New generation

    best_route = min(population, key=total_distance)
    return best_route, total_distance(best_route)

# Example Usage
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]  # 10 random cities
best_route, best_distance = genetic_algorithm(cities)

print("Best Route:", best_route)
print("Total Distance:", best_distance)
