import numpy as np

# Add neighboring nodes to each node
def add_properties_to_nodes(nodes, edges):
    for node, node_properties in nodes.items():
        neighboring_nodes = []
        for edge, edge_properties in edges.items():
            if edge.split(" → ")[0] == node:
                neighboring_nodes.append(edge.split(" → ")[1])
        nodes[node]["neighboring nodes"] = neighboring_nodes

# Adding the tt_0, capacity, number of cars on the edge at each time, and (expected) travel time at each time as properties of each edge
def add_properties_to_edges(edges, l_car, d_spacing):
    for edge, properties in edges.items():
        length = properties["length"]
        speed_limit = properties["speed_limit"]
        tt_0 = (length / speed_limit) * (60 / 1000)  # minutes
        properties["tt_0"] = tt_0
        properties["capacity"] = int(properties["lanes"] * properties["length"] / (l_car + d_spacing))
        properties["cars_on_edge"] = [] # list of the amount of cars occupying the edge at each time
        properties["travel_time"] = [] # list of the (expected) travel time on the edge at each time

# Using A*-algorithm to determine the optimal path for a car
## A* applied on a single node
def A_star(current_route, current_queue, nodes, edges, time, heuristic_constant, distance_matrix, destination):
    current_path = current_route["path"]
    current_travel_time = current_route["travel time"]
    last_node_of_path = current_path[-1].split(" → ")[-1]
    updated_queue = current_queue

    neighboring_nodes = nodes[last_node_of_path]["neighboring nodes"] # list of the neighboring nodes of the last node of the route that is currently checked
    for neighbor in neighboring_nodes:
        new_path = current_path.append(last_node_of_path + " → " + neighbor)
        new_travel_time = current_travel_time + edges[last_node_of_path + " → " + neighbor]["travel time"][time]
        new_heuristic = heuristic_constant * distance_matrix[neighbor + " → " + destination]
        new_total_cost = new_travel_time + new_heuristic
        new_route = {"path": new_path, "travel time": new_travel_time, "heuristic": new_heuristic, "total cost": new_total_cost}
        updated_queue.append(new_route)
    
    # Sorting routes on total cost, deleting the routes going to the same node that have higher total cost
    updated_queue = sorted(updated_queue, key=lambda x: x["total cost"])
    for neighbor in neighboring_nodes:

    

        


def determine_path(car, nodes, edges, time, heuristic_constant, distance_matrix):
    origin = car["origin"]
    destination = car["destination"]
    queue = [] # list of routes checked by the algorithm. A route is a dictionary {"path": path, "travel time": travel time, "heuristic": heuristic, "total cost": travel time + heuristic} where path is a list of the traversed edges (e.g. [origin→A, A→B, ...]) and heuristic is the heuristic constant * the Euclidean distance from the last node of the path to the destination

