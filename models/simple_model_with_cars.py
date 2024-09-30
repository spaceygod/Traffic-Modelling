import numpy as np
import random
from utils.plotting import plot_3D_surface, plot_scatter_with_correlation
from utils.functional import add_properties_to_edges
from utils.simulate_with_cars import simulate_and_compare
from Images.cars import *

# Parameters for BPR function
alpha = 0.15
beta = 4
sigma = 2
l_car = 4.5 # Length of a car in meters
d_spacing = 55 # Minimum safe spacing between cars in meters

# Define the road network (length in meters, speed limit in km/h, number of lanes)
edges = {
    "City 1 → A": {"length": 30000, "speed_limit": 100, "lanes": 2},
    "City 1 → B": {"length": 50000, "speed_limit": 100, "lanes": 2},
    "A → C": {"length": 40000, "speed_limit": 100, "lanes": 2},
    "B → C": {"length": 60000, "speed_limit": 100, "lanes": 2},
    "C → D": {"length": 20000, "speed_limit": 100, "lanes": 2},
    "C → E": {"length": 30000, "speed_limit": 100, "lanes": 2},
    "C → City 2": {"length": 20000, "speed_limit": 100, "lanes": 1},
    "D → City 2": {"length": 10000, "speed_limit": 100, "lanes": 2},
    "E → City 2": {"length": 5000, "speed_limit": 100, "lanes": 2},
}

# Node positions for plotting
node_positions = {
    "City 1": (0, 0),
    "A": (2, 2),
    "B": (2, -2),
    "C": (4, 0),
    "D": (6, 2),
    "E": (6, -2),
    "City 2": (8, 0)
}

# Add properties to edges
edges = add_properties_to_edges(edges, l_car, d_spacing)

# Simulation settings
num_minutes = 240
warmup_steps = 120
car_distribution_mean = 85
car_distribution_std = 1.5

# Sample the number of cars arriving at City 1 each minute
np.random.seed(42)
cars_arriving_each_minute = np.random.normal(car_distribution_mean, car_distribution_std, num_minutes)
cars_arriving_each_minute = np.round(cars_arriving_each_minute).astype(int)

# Initialize data for cars
cars = []
car_icons = ['bowser_car.png', 'hot_wheels.png', 'mario_car.png', 'orange_bus.png', 
             'royal_carriage.png', 'super_car.png', 'white_car.png', 'dot_im.png'] # List with all car icons
car_images = [plt.imread(icon) for icon in car_icons] # All icons are pre-loaded and are put into this list
car_id = 0
for minute, num_cars in enumerate(cars_arriving_each_minute):
    for _ in range(num_cars):
        car = {"id": car_id, "start_time": minute, "location": "", "arrived_at_node": 0, "left_at_node": 0, "was_on_route": "", "image": random.choice(car_images)}  # Initialize car data
        cars.append(car)
        car_id += 1

# Run the simulation and plot the results
all_car_reach_times, most_congested_edge = simulate_and_compare(cars, edges, node_positions, num_minutes, warmup_steps, deltas=[1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0], alpha=0.15, beta=4, sigma=2)
plot_3D_surface(all_car_reach_times)
plot_scatter_with_correlation(all_car_reach_times)