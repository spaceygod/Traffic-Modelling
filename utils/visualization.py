import matplotlib.pyplot as plt

# Plotting function to visualize nodes, edges, and car counts
def initialize_plot(edges, node_positions, bg_image=None, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    if bg_image is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-1, 9)
        ax.set_ylim(-3, 3)
        fig.patch.set_facecolor('black')  # Black background for the figure
        ax.set_facecolor('black')         # Black background for the axes

        ax.set_title("Real-Time Simulation of Car Movements", color='white')  # Title in white

        # Plot nodes (cities)
        for node, pos in node_positions.items():
            ax.plot(pos[0], pos[1], 'o', markersize=20, label=node)
            ax.text(pos[0], pos[1], node, fontsize=12, ha='center', va='center', color='white')

        # Store edge and node text for dynamic updating
        edge_texts = {}
        for edge in edges:
            start_node, end_node = edge.split(" → ")
            start_pos, end_pos = node_positions[start_node], node_positions[end_node]
            mid_pos = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='cyan', linestyle='--', linewidth=2)
            edge_texts[edge] = ax.text(mid_pos[0], mid_pos[1], '', fontsize=12, color='purple')
    else:
        fig, ax = plt.subplots(figsize=(12, 16))

        # Set the plot limits to match the extent of the image
        ax.set_xlim([lon_min, lon_max])
        ax.set_ylim([lat_min, lat_max])

        # Load and display the background image (scale according to the map)
        ax.imshow(bg_image, extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto')  # Fit image to the latitude/longitude extent

        ax.set_title("Real-Time Simulation of Car Movements", color='white')  # Title in white

        # Plot nodes (cities)
        for node, pos in node_positions.items():
            ax.plot(pos[0], pos[1], 'o', markersize=20, label=node)
            ax.text(pos[0], pos[1], node, fontsize=12, ha='center', va='center', color='white')

        ax.set_facecolor('black')         # Black background for the axes
        ax.set_title("Real-Time Simulation of Car Movements", color='black')  # Title in white

        # Store edge and node text for dynamic updating
        edge_texts = {}
        for edge in edges:
            start_node, end_node = edge.split(" → ")
            start_pos, end_pos = node_positions[start_node], node_positions[end_node]
            mid_pos = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)

            # Increase line width for better visibility
            ax.plot([start_pos[1], end_pos[1]], [start_pos[0], end_pos[0]], color='cyan', linestyle='--', linewidth=2)
            # Increase font size for edge texts
            edge_texts[edge] = ax.text(mid_pos[1], mid_pos[0], '', fontsize=12, color='purple')

    # Add a text annotation to display the current timestep
    timestep_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='blue')

    return fig, ax, edge_texts, timestep_text

# Update function for the animation
def update_plot(t, edges, vehicle_counts, edge_texts, timestep_text, num_minutes):
    # Update edge texts with the number of cars and capacity
    for edge, text in edge_texts.items():
        num_cars = int(vehicle_counts[edge][0])
        capacity = edges[edge]["current_capacity"]
        text.set_text(f"{num_cars}/{capacity}")

    # Update the timestep text
    timestep_text.set_text(f"Timestep: {t}/{num_minutes}")