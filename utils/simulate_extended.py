import sys
import os
from tqdm import tqdm
# import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.functional import travel_time_bpr
from utils.visualization_extended import initialize_plot, update_plot
from utils.functional_extended import determine_optimal_route, find_next_node, convert_nodes, save_simulation_results
from utils.modified_A_star import run_A_mod, update_future_edges

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

def simulate_A_star(nodes, edges, cars, alpha, beta, sigma, num_minutes, distance_matrix, heuristic_constant, bg_image=None, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    # Cars and edges after the simulation; a copy is made so that the original cars and edges are not changed and can be used for the normal A* simulation
    new_cars = copy.deepcopy(cars) 
    new_edges = copy.deepcopy(edges)

    # Ask the user if it wants the traffic simulation to be animated
    animate = input("Do you want to animate the traffic simulation in real-time? Code wil run slower. (y/n): ").lower() == "y"

    # Converting the nodes database from the format used in the simulation to the format used in the visualization
    nodes_visualization = convert_nodes(nodes)

    if animate:
        # Initializing the visualization
        fig, ax, edge_texts, timestep_text, edge_lines = initialize_plot(edges, nodes_visualization, bg_image, lat_min, lat_max, lon_min, lon_max)

    def update(time):
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

        # Determining the vehicle_counts dictionary used in the visualization
        vehicle_counts = {edge: [new_edges[edge]["cars on edge"][time]] for edge in edges}

        if animate:
            # Update the visualization
            update_plot(time, edges, vehicle_counts, edge_texts, timestep_text, num_minutes, edge_lines)

    if animate:
        # Animate the plot over time
        anim = FuncAnimation(fig, update, frames=range(num_minutes), repeat=False, interval=100)

        try: 
            # Save as GIF
            anim.save("A_star_simulation.gif", writer="pillow", fps=20)
            print("Animation saved as A_star_simulation.gif")
        except Exception as e:
            print("Error saving animation as gif:", e)

        plt.show()
        plt.close(fig)
    else:
        # Run the simulation without animation using tqdm for a progress bar
        for t in tqdm(range(num_minutes), desc=f"Simulating A*"):
            update(t)

    # Call to save the simulation results after the loop ends
    save_simulation_results(new_cars, nodes, new_edges, distance_matrix, heuristic_constant, filename="A_star_simulation_results.csv")

    return new_cars, new_edges

# Simulate the modified A* algorithm
def simulate_A_mod(nodes, edges, cars, alpha, beta, sigma, num_minutes, distance_matrix, heuristic_constant):
    # Cars and edges after the simulation
    new_cars = copy.deepcopy(cars) 
    new_edges = copy.deepcopy(edges)

    # An extra copy of edges that the modified A* algorithm will use to predict travel times in the future
    future_edges = copy.deepcopy(edges)

    # Iteration of the simulation
    for time in tqdm(range(num_minutes), desc=f"Simulating A* Mod"):
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
                car["trajectory"] = run_A_mod(nodes, future_edges, car, heuristic_constant, distance_matrix, num_minutes)

                # Updating the future edge occupation and travel times
                update_future_edges(future_edges, car["trajectory"], alpha, beta, sigma)
                
                if car["trajectory"] is None:
                    car["active"] = False  # No route available
                else:
                    car["active"] = True
                    car["location"] = car["origin"]
                    car["next edge"] = car["trajectory"][0][0] + " → " + car["trajectory"][1][0] if len(car["trajectory"]) > 1 else None

        ## Moving all active cars according to their predetermined trajectory
        for car in new_cars:
            # Checking if the car is in the system (active)
            if car["active"]:

                # If a car is on its origin, check if the first edge on its trajectory is free. If so, enter the edge. If not, wait on the origin
                if car["location"] == car["origin"] and car["next edge"] is not None:
                    cars_on_next_edge = new_edges[car["next edge"]]["cars on edge"][time]
                    capacity_of_next_edge = new_edges[car["next edge"]]["capacity"]

                    if cars_on_next_edge < capacity_of_next_edge:
                        car["location"] = car["next edge"]
                        car["time entered last edge"] = time

                        # If this first edge immediately leads to the cars destination, the 'next edge' property should be None
                        if car["trajectory"][1][0] == car["destination"]:
                            car["next edge"] = None
                        else:
                            car["next edge"] = car["trajectory"][1][0] + " → " + car["trajectory"][2][0]

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
                        next_node = find_next_node(car["trajectory"], node_just_passed)
                        
                        if next_node == car["destination"]:
                            car["next edge"] = None
                        else:
                            car["next edge"] = next_node + " → " + find_next_node(car["trajectory"], next_node)

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

    # Call to save the simulation results after the loop ends
    save_simulation_results(new_cars, nodes, new_edges, distance_matrix, heuristic_constant, filename="A_star_mod_simulation_results.csv")

    return new_cars, new_edges, future_edges
                