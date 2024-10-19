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
def A_star(current_node, queue, nodes, edges, time):
    neighboring_nodes = []


def determine_path(car, nodes, edges, time, heuristic_constant):
    origin = car["origin"]
    destination = car["destination"]
    queue = {} # queue of paths checked by the algorithm in the form {path : cost} where cost is the total travel time of the path + the heuristic (i.e. some constant * the Euclidean distance)

