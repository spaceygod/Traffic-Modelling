import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
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

def simulate_A_star(nodes, edges, cars, alpha, beta, sigma, num_minutes, warmup_steps, distance_matrix, heuristic_constant):
    travel_times_of_cars = {} # Dictionary of the form {"A → B": [travel times of cars travelling from A to B], ...}

    for time in range(num_minutes):
        ## Based on the number of cars on each edge, calculate the travel time of each edge
        for edge, properties in edges.items():
            properties["travel time"][time] = travel_time_bpr(properties["tt_0"], properties["cars on edge"][time], properties["capacity"], alpha, beta, sigma)

        ## Removing all cars that have reached their destination
        for car in cars:
            if car["location"] != None:
                if car["location"].split(" → ")[-1] == car["destination"] and car["finished edge"] == True: # If a car is on the edge to its destination and it has finished that edge ...
                    car["time arrived"] = time
                    car["active"] = False
                    car["location"] = None
        
        ## Spawning new cars at their origin and calculating their optimal path
        for car in cars:
            if car["time spawned"] == time:
                car["active"] = True
                car["location"] = car["origin"] # Normally a car's location is an edge, but this is the exception
                car["optimal path"], car["optimal travel time"] = determine_optimal_route(car, nodes, edges, time, heuristic_constant, distance_matrix)
                car["next edge"] = car["optimal path"][0] + " → " + car["optimal path"][1]
        
        ## Moving all active cars according to their predetermined optimal path
        for car in cars:
            # Checking if the car is in the system (active)
            if car["active"]:

                # If a car is on its origin, check if the first edge on its path is free. If so, enter the edge. If not, wait on the origin
                if car["location"] == car["origin"]:
                    cars_on_next_edge = edges[car["next edge"]]["cars on edge"][time]
                    capacity_of_next_edge = edges[car["next edge"]]["capacity"]
                    if cars_on_next_edge < capacity_of_next_edge:
                        car["location"] = car["next edge"]
                        car["next edge"] = car["optimal path"][1] + " → " + car["optimal path"][2]
                        car["time entered last edge"] = time

                        # Add one to the number of cars on the edge where this car is located
                        edges[car["location"]]["cars on edge"][time] += 1
                    else:
                        continue
                
                # If a car had not finished its edge (if it was still travelling), check if it has finished its edge now
                if car["finished edge"] == False:
                    if time - car["time entered last edge"] >= edges[car["location"]]["travel time"]:
                        car["finished edge"] = True
                    else:
                        # Add one to the number of cars on the edge where this car is located
                        edges[car["location"]]["cars on edge"][time] += 1
                
                # If a car has finished its edge (if it is waiting at the end), check if the next edge in its path is free. If so, enter the edge. If not, continue waiting on the edge
                if car["finished edge"] == True:
                    cars_on_next_edge = edges[car["next edge"]]["cars on edge"][time]
                    capacity_of_next_edge = edges[car["next edge"]]["capacity"]
                    if cars_on_next_edge < capacity_of_next_edge:
                        car["location"] = car["next edge"]
                        car["time entered last edge"] = time
                        car["finished edge"] = False

                        # Some code to determine the next edge from the optimal path
                        index_of_end_node_of_finished_edge = car["optimal path"].index(car["next edge"].split(" → ")[0])
                        car["next edge"] = car["optimal path"][index_of_end_node_of_finished_edge + 1] + " → " + car["optimal path"][index_of_end_node_of_finished_edge + 2]

                        # Add one to the number of cars on the edge where this car is located
                        edges[car["location"]]["cars on edge"][time] += 1
                    else:
                        # Add one to the number of cars on the edge where this car is located
                        edges[car["location"]]["cars on edge"][time] += 1
                
            else:
                continue
    
    return cars, edges

            