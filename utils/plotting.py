import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import pearsonr

# Plotting 3D Surface of Travel Times with interpolation for smoothness
def plot_3D_surface(all_car_reach_times, bins=25, grid_res=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare lists for surface plotting
    travel_times = []
    multipliers = []
    frequencies = []

    for delta, car_reach_times in all_car_reach_times:
        hist, xedges = np.histogram(car_reach_times, bins=bins)
        xpos = (xedges[:-1] + xedges[1:]) / 2
        ypos = np.full_like(xpos, delta)
        travel_times.append(xpos)
        multipliers.append(ypos)
        frequencies.append(hist)

    travel_times = np.concatenate(travel_times)
    multipliers = np.concatenate(multipliers)
    frequencies = np.concatenate(frequencies)

    travel_time_grid = np.linspace(min(travel_times), max(travel_times), grid_res)
    multiplier_grid = np.linspace(min(multipliers), max(multipliers), grid_res)
    grid_x, grid_y = np.meshgrid(travel_time_grid, multiplier_grid)

    grid_z = griddata((travel_times, multipliers), frequencies, (grid_x, grid_y), method='cubic')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Travel Time (minutes)")
    ax.set_ylabel("Capacity Multiplier")
    ax.set_zlabel("Number of Cars")
    plt.title("Effect of Capacity Increase on Travel Times")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

# Plot scatter with correlation analysis
def plot_scatter_with_correlation(all_car_reach_times, end_location='City 2'):
    deltas = []
    avg_reach_times = []
    cars_reached_end = []

    for delta, car_reach_times in all_car_reach_times:
        deltas.append(delta)
        avg_reach_times.append(np.mean(car_reach_times))
        cars_reached_end.append(len(car_reach_times))

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(deltas, avg_reach_times, color='b')
    plt.title("Delta vs. Average Car Reach Time")
    plt.xlabel("Capacity Multiplier (Delta)")
    plt.ylabel("Average Car Reach Time (minutes)")
    correlation_coef1, _ = pearsonr(deltas, avg_reach_times)
    plt.text(1.1, max(avg_reach_times) - 1, f"Correlation Coefficient: {correlation_coef1:.2f}", fontsize=12, color='blue')

    plt.subplot(1, 2, 2)
    plt.scatter(deltas, cars_reached_end, color='g')
    plt.title(f"Delta vs. Number of Cars Reached {end_location}")
    plt.xlabel("Capacity Multiplier (Delta)")
    plt.ylabel(f"Number of Cars Reached {end_location}")
    correlation_coef2, _ = pearsonr(deltas, cars_reached_end)
    plt.text(1.1, max(cars_reached_end) - 10, f"Correlation Coefficient: {correlation_coef2:.2f}", fontsize=12, color='green')

    plt.tight_layout()
    plt.show()
