import numpy as np

states = ['Sunny', 'Cloudy', 'Rainy']
state_to_index = {state: i for i, state in enumerate(states)}

# Transition matrix
transition_matrix = np.array([
    [0.6, 0.3, 0.1],  # from Sunny
    [0.3, 0.4, 0.3],  # from Cloudy
    [0.2, 0.4, 0.4],  # from Rainy
])

def simulate_weather(start_state, days=10):
    current_state = state_to_index[start_state]
    weather_sequence = [states[current_state]]

    for _ in range(days - 1):
        current_state = np.random.choice([0, 1, 2], p=transition_matrix[current_state])
        weather_sequence.append(states[current_state])

    return weather_sequence

# Simulation parameters
simulations = 10000
rainy_counts = []

for _ in range(simulations):
    forecast = simulate_weather('Sunny', 10)
    rainy_days = forecast.count('Rainy')
    rainy_counts.append(rainy_days >= 3)

# Calculate the estimated probability
probability = sum(rainy_counts) / simulations

# Show result
print("Estimated Probability of at least 3 Rainy Days in 10 days:", round(probability, 4))
