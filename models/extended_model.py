import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
from utils.functional_extended import add_properties_to_nodes
from utils.functional_extended import add_properties_to_edges

# Parameters for BPR function
alpha = 0.15
beta = 4
sigma = 2
l_car = 4.5 # Length of a car in meters
d_spacing = 55 # Minimum safe spacing between cars in meters

# Simulation settings
num_minutes = 240
warmup_steps = 120
total_cars_spawned_each_minute = 100
car_distribution_std = 1.5
heuristic_constant = 1

# Node positions for plotting
nodes = {
    "City 1": {"coordinates": (0, 0), "population": 1000},
    "A": {"coordinates": (2, 2), "population": 0},
    "B": {"coordinates": (2, -2), "population": 0},
    "C": {"coordinates": (4, 0), "population": 0},
    "D": {"coordinates": (6, 2), "population": 0},
    "E": {"coordinates": (6, -2), "population": 0},
    "City 2": {"coordinates": (8, 0), "population": 600}
}

# Create a dictionary containing the Euclidean distance between each pair of nodes
distance_matrix = {}
for node_A, properties_A in nodes.items():
    for node_B, properties_B in nodes.items():
        if node_A == node_B:
            distance_matrix[node_A + " → " + node_B] = 0
        elif node_B + " → " + node_A in distance_matrix:
            distance_matrix[node_A + " → " + node_B] = distance_matrix[node_B + " → " + node_A]
        else:
            distance_matrix[node_A + " → " + node_B] = math.sqrt((properties_A["coordinates"][0] - properties_B["coordinates"][0])**2 + (properties_A["coordinates"][1] - properties_B["coordinates"][1])**2)

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

# Add properties to nodes
add_properties_to_nodes(nodes, edges)

# Add properties to edges
edges = add_properties_to_edges(edges, l_car, d_spacing, num_minutes)

# Create a dictionary containing what fraction of cars will travel from each node A to each node B
travel_matrix = {}

# Calculating the total population
total_population = 0
for node, properties in nodes.items():
    total_population += properties["population"]

# Filling in the travel matrix
for origin, origin_properties in nodes.items():
    for destination, destination_properties in nodes.items():
        if origin == destination:
            continue
        else:
            travel_matrix[origin + " → " + destination] = (origin_properties["population"] / total_population) * (destination_properties["population"] / (total_population - origin_properties["population"]))

print(travel_matrix)

# Sample the number of cars spawning at each origin and going to each destination for each minute
np.random.seed(42)
cars_spawned_each_minute = {} # dictionary with structure 'origin → destination : [#cars spawned at t=0 in origin going to destination, #cars spawned at t=1 in ..., ...]'
for origin_destination, fraction_of_cars in travel_matrix.items():
    cars_spawned_each_minute[origin_destination] = np.round(np.random.normal(total_cars_spawned_each_minute * fraction_of_cars, car_distribution_std, num_minutes)).astype(int)

# Initialize data for cars
cars = []
car_id = 0
for minute in range(num_minutes):
    for origin_destination, cars_spawned_over_time in cars_spawned_each_minute.items():
        for _ in range(cars_spawned_over_time[minute]):
            car = {
                "id": car_id, 
                "origin": origin_destination.split(" → ")[0], 
                "destination": origin_destination.split(" → ")[1], 
                "optimal path": None, 
                "optimal travel time": None, 
                "time spawned": minute, 
                "time arrived": None, 
                "active": False, 
                "location": None, 
                "time entered last edge": None, 
                "finished edge": False,
                "next edge": None
                } 
            cars.append(car)
            car_id += 1