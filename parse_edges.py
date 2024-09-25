def parse_highway_data(file_path):
    edges = {}

    with open(file_path, 'r') as file:
        current_highway = None

        for line in file:
            line = line.strip()

            # Replace all - with ->
            line = line.replace("-", "→")

            # Skip empty lines
            if not line:
                continue

            # Detect highway headers like "A1:", "A2:", etc.
            if line.endswith(':'):
                current_highway = line.strip(':')
            else:
                # Split the road description by ":"
                segments = line.split(':')
                if len(segments) != 3:
                    continue  # Skip any malformed lines

                # Extract the road details
                cities, lanes_info, length_str = segments

                # Strip unnecessary spaces
                cities = cities.strip()
                lanes_info = lanes_info.strip()
                length_str = length_str.strip()

                # Convert the length to meters (in kilometers by default)
                length_km = float(length_str)
                length_meters = int(length_km * 1000)

                # Extract the number of lanes
                lanes = lanes_info.split('x')[1].split()[0]  # e.g., "2x3 lanes" -> 3 lanes

                # Add the road to the edges dictionary
                edges[cities] = {
                    "length": length_meters,
                    "speed_limit": 100,  # Set speed limit to 100 for all roads
                    "lanes": int(lanes)
                }

    return edges

# Path to the uploaded file
file_path = './real_highway_data.txt'
edges = parse_highway_data(file_path)
print(edges)