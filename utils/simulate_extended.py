import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import copy
from utils.functional import travel_time_bpr
from utils.functional_extended import iterate_A_star, determine_optimal_route

# Each iteration of the simulation the following things are done:
# 1. The number of cars on each edge is determined and the travel time of each edge is updated
# 2. Cars that have reached their destination are removed and have their arrival time added
# 3. New cars are spawned at their origin and their optimal path is calculated
# 4. All cars in the system try to move according to their predetermined path:
#    A car is either travelling on an edge, or waiting at the end of the edge (or waiting at its origin) until the next edge on its path becomes free
#   - If a car is on its origin, check if the first edge on its path is free. If so, enter the edge. If not, wait on the origin.
#   - If a car had not finished its edge (if it was still travelling), check if it has finished its edge now. 
#   - If a car has finished its edge (if it is waiting at the end), check if the next edge in its path is free. If so, enter the edge. If not, continue waiting on the edge.
#
# The simulation returns the travel time of each car. Cars that did not reach their destination and cars spawned during the warmup steps are ignored.

def simulate_A_star(nodes, edges, cars, alpha, beta, sigma, num_minutes, distance_matrix, heuristic_constant):
    # Cars and edges after the simulation
    new_cars = copy.deepcopy(cars) 
    new_edges = copy.deepcopy(edges)

    # Iteration of the simulation
    for time in range(num_minutes):

        ## Removing all cars that have reached their destination
        for car in new_cars:
            if car["location"] != None:
                if car["location"].split(" → ")[-1] == car["destination"] and car["finished edge"] == True: # If a car is on the edge to its destination and it has finished that edge ...
                    car["time arrived"] = time
                    car["active"] = False
                    car["location"] = None
        
        ## Spawning new cars at their origin and calculating their optimal path
        for car in new_cars:
            if car["time spawned"] == time:
                car["optimal path"], car["optimal travel time"] = determine_optimal_route(car, nodes, new_edges, time, heuristic_constant, distance_matrix)
                
                if car["optimal path"] is None:
                    car["active"] = False  # No route available
                else:
                    car["active"] = True
                    car["location"] = car["origin"]
                    car["next edge"] = car["optimal path"][0] + " → " + car["optimal path"][1] if len(car["optimal path"]) > 1 else None
        
        ## Moving all active cars according to their predetermined optimal path
        for car in new_cars:
            # Checking if the car is in the system (active)
            if car["active"]:

                # If a car is on its origin, check if the first edge on its path is free. If so, enter the edge. If not, wait on the origin
                if car["location"] == car["origin"] and car["next edge"] is not None:
                    cars_on_next_edge = new_edges[car["next edge"]]["cars on edge"][time]
                    capacity_of_next_edge = new_edges[car["next edge"]]["capacity"]

                    if cars_on_next_edge < capacity_of_next_edge:
                        car["location"] = car["next edge"]
                        car["time entered last edge"] = time

                        # If this first edge immediately leads to the cars destination, the 'next edge' property should be None
                        if car["optimal path"][1] == car["destination"]:
                            car["next edge"] = None
                        else:
                            car["next edge"] = car["optimal path"][1] + " → " + car["optimal path"][2]

                        # Add one to the number of cars on the edge where this car is located
                        new_edges[car["location"]]["cars on edge"][time] += 1
                    else:
                        continue
                
                # If a car had not finished its edge (if it was still travelling), check if it has finished its edge now
                if not car["finished edge"]:
                    if time - car["time entered last edge"] >= new_edges[car["location"]]["travel time"][time]:
                        car["finished edge"] = True
                    else:
                        # Add one to the number of cars on the edge where this car is located
                        new_edges[car["location"]]["cars on edge"][time] += 1
                
                # If a car has finished its edge (if it is waiting at the end), check if the next edge in its path is free. If so, enter the edge. If not, continue waiting on the edge
                if car["finished edge"] == True and car["next edge"] is not None:
                    cars_on_next_edge = new_edges[car["next edge"]]["cars on edge"][time]
                    capacity_of_next_edge = new_edges[car["next edge"]]["capacity"]
                    
                    # Check if the next edge is free
                    if cars_on_next_edge < capacity_of_next_edge:
                        car["location"] = car["next edge"]
                        car["time entered last edge"] = time
                        car["finished edge"] = False

                        # Some code to determine the next edge from the optimal path
                        node_just_passed = car["location"].split(" → ")[0]
                        next_node = car["optimal path"][car["optimal path"].index(node_just_passed) + 1]
                        if next_node == car["destination"]:
                            car["next edge"] = None
                        else:
                            car["next edge"] = next_node + " → " + car["optimal path"][car["optimal path"].index(node_just_passed) + 2]

                        # Add one to the number of cars on the edge where this car is located
                        new_edges[car["location"]]["cars on edge"][time] += 1
                    else:
                        # Add one to the number of cars on the edge where this car is located
                        new_edges[car["location"]]["cars on edge"][time] += 1
                
            else:
                continue
    
        ## Based on the number of cars on each edge, calculate the travel time of each edge
        for edge, properties in new_edges.items():
            properties["travel time"][time] = travel_time_bpr(properties["tt_0"], properties["cars on edge"][time], properties["capacity"], alpha, beta, sigma)

    return new_cars, new_edges

# Simulate the modified A* algorithm
def simulate_A_mod(nodes, edges, cars, alpha, beta, sigma, num_minutes, distance_matrix, heuristic_constant):
    # Cars and edges after the simulation
    new_cars = copy.deepcopy(cars) 
    new_edges = copy.deepcopy(edges)

    # Iteration of the simulation
    for time in range(num_minutes):
                       