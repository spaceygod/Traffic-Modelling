def parse_highway_data(file_path):
    import logging

    # Configure logging to display warnings
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    edges = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            current_highway = None

            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                # Replace all - with ->
                line = line.replace(" - ", " → ")

                # Skip empty lines
                if not line:
                    continue

                # Detect highway headers like "A1:", "A2:", etc.
                if line.endswith(':'):
                    current_highway = line.strip(':')
                    continue  # Proceed to the next line after setting the highway
                else:
                    # Split the road description by ":"
                    segments = line.split(':')
                    if len(segments) != 3:
                        logging.warning(f"Line {line_number}: Skipping malformed line (expected 3 segments separated by ':'). Line content: '{line}'")
                        continue  # Skip any malformed lines

                    # Extract the road details
                    cities, lanes_info, length_str = segments

                    # Strip unnecessary spaces
                    cities = cities.strip()
                    lanes_info = lanes_info.strip()
                    length_str = length_str.strip()

                    # Convert the length to meters (assuming the length is in kilometers)
                    try:
                        length_km = float(length_str)
                        length_meters = int(length_km * 1000)
                    except ValueError:
                        logging.warning(f"Line {line_number}: Invalid length value '{length_str}'. Skipping this line.")
                        continue  # Skip lines with invalid length

                    # Extract the number of lanes
                    try:
                        lanes_part = lanes_info.split('x')[1]  # e.g., "2x3 lanes" -> "3 lanes"
                        lanes = lanes_part.split()[0]  # "3 lanes" -> "3"
                        lanes = int(lanes)
                    except (IndexError, ValueError):
                        logging.warning(f"Line {line_number}: Unable to parse lanes information '{lanes_info}'. Skipping this line.")
                        continue  # Skip lines with invalid lanes information

                    # Split cities and validate
                    city_split = cities.split("→")
                    if len(city_split) != 2:
                        logging.warning(f"Line {line_number}: Expected exactly two cities separated by '→'. Found {len(city_split)} parts. Line content: '{cities}'")
                        continue  # Skip lines that don't have exactly two cities

                    city_from, city_to = [city.strip() for city in city_split]

                    # Define edge keys
                    edge_key = f"{city_from} → {city_to}"
                    reverse_edge_key = f"{city_to} → {city_from}"

                    # Add the road to the edges dictionary (original direction)
                    edges[edge_key] = {
                        "highway": current_highway,
                        "length": length_meters,
                        "speed_limit": 100,  # Set speed limit to 100 for all roads
                        "lanes": lanes
                    }

                    # Add the reverse direction
                    edges[reverse_edge_key] = {
                        "highway": current_highway,
                        "length": length_meters,
                        "speed_limit": 100,
                        "lanes": lanes
                    }

    except FileNotFoundError:
        logging.error(f"The file '{file_path}' does not exist.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return edges


# Path to the uploaded file
file_path = './real_data/real_highway_data.txt'
edges = parse_highway_data(file_path)