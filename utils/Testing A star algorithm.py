import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
from utils.functional_extended import add_properties_to_nodes
from utils.functional_extended import add_properties_to_edges
from utils.simulate_extended import simulate_A_star
from utils.functional import travel_time_bpr
from utils.functional_extended import iterate_A_star, determine_optimal_route

# Parameters for BPR function
alpha = 0.15
beta = 4
sigma = 2
l_car = 4.5 # Length of a car in meters
d_spacing = 55 # Minimum safe spacing between cars in meters
num_minutes = 240

heuristic_constant = 1

# Node positions for plotting
nodes = {
    "City 1": {"coordinates": (0, 0), "population": 1000},
    "A": {"coordinates": (2, 2), "population": 0},
    "B": {"coordinates": (2, -2), "population": 0},
    "C": {"coordinates": (4, 0), "population": 0},
    "D": {"coordinates": (6, 2), "population": 0},
    "E": {"coordinates": (6, -2), "population": 0},
    "City 2": {"coordinates": (8, 0), "population": 1000}
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
add_properties_to_edges(edges, l_car, d_spacing, num_minutes)

# Initialize data for cars
cars = []

car1 = {
                "id": 0, 
                "origin": "City 1", 
                "destination": "City 2", 
                "optimal path": None, 
                "optimal travel time": None, 
                "time spawned": 0, 
                "time arrived": None, 
                "active": False, 
                "location": None, 
                "time entered last edge": None, 
                "finished edge": False,
                "next edge": None
                }

cars.append(car1)

optimal_path_car1, travel_time = determine_optimal_route(car1, nodes, edges, 0, heuristic_constant, distance_matrix)
# print(optimal_path_car1)