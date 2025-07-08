import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import os

# Data provided by the user (simulating test_data.csv content)
#csv_data = """profile id,image id,temp,tint,exposure,contrast,highlights,shadows,whites,blacks
#hQ8L0kP3sA,jU2F5tG9oC,28567,-112,0.92,-78,45,-19,63,-81
#eN7W1qR2xD,kM3G6vH0bE,41234,87,-3.15,12,-67,88,10,-55
#bY9C4fO7lZ,pA1D8eQ5wX,19876,-45,4.01,92,23,-7,34,-2
#tR1A6nB8yF,sC9H3jL2kD,35678,130,2.56,-10,77,-98,50,66
#fV3E0mU2qI,xZ4Y7iN1oP,22345,-8,-1.23,55,-30,15,-90,40
#kX5R9aW1bG,cC2T7uV6fM,48901,101,-0.78,70,8,-60,25,75
#dZ2S7bC0pJ,aB9W4xK3hD,31209,22,3.89,-42,100,5,-10,95
#gJ6N0pM4rT,lO5P8qR7sU,39045,-140,-2.99,85,-50,20,80,-35
#wA0V5sQ9uC,jK6L3mN2oP,25789,66,1.05,-20,60,-40,15,-70
#zX7C1vB2nM,tY9U4iO8eS,44567,-33,0.01,-5,30,-10,70,-15
#"""

# Define slider columns and their original ranges, and data types
slider_info = {
    #'Temperature': {'range': (2000, 50000), 'dtype': int},
    #'Tint': {'range': (-150, 150), 'dtype': int},
    #'Exposure': {'range': (-5.0, 5.0), 'dtype': float},
    'Contrast': {'range': (-100, 100), 'dtype': int},
    'Highlights': {'range': (-100, 100), 'dtype': int},
    'Shadows': {'range': (-100, 100), 'dtype': int},
    'Whites': {'range': (-100, 100), 'dtype': int},
    'Blacks': {'range': (-100, 100), 'dtype': int}
}

# Define the target normalized range for plotting
NORMALIZED_MIN = 0
NORMALIZED_MAX = 100


def normalize_value(value, original_min, original_max, target_min=NORMALIZED_MIN, target_max=NORMALIZED_MAX):
    """
    Normalizes a value from its original range to a target range.
    """
    if original_max == original_min:  # Handle cases where range is zero to prevent division by zero
        return target_min

    # Linear interpolation formula
    normalized_val = target_min + (value - original_min) * \
                     (target_max - target_min) / (original_max - original_min)
    return normalized_val


# Define the subfolder name
output_folder = "slider_based_shapes_data_exif_neil_1"

try:
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Ensured output folder '{output_folder}' exists.")

    # Read the CSV data from the string buffer (simulating file reading)
    # If you have a physical file named 'test_data.csv', use: df = pd.read_csv('test_data.csv')
    #df = pd.read_csv(io.StringIO(csv_data))
    df = pd.read_csv('sliders_exif_neil.csv')
    df = df.fillna(0)

    # Convert slider columns to appropriate data types
    for col, info in slider_info.items():
        if col in df.columns:
            df[col] = df[col].astype(info['dtype'])
        else:
            print(f"Warning: Column '{col}' not found in the CSV data.")

    # Generate an image for each row
    for index, row in df.iterrows():
        profile_id = row['folder_id']
        # CORRECTED: Access 'image id' with a space as per CSV header
        image_id = row['img_name']

        # Create a new figure and axes for each row
        fig, ax = plt.subplots(figsize=(10, 6))

        # Lists to store normalized values and corresponding y-positions for plotting the line
        normalized_values_to_plot = []
        y_positions_to_plot = []

        y_counter = 0  # Counter for y-position of each slider

        # Iterate through each slider column to gather data for plotting
        for col, info in slider_info.items():
            if col in row:  # Check if the column exists in the current row
                original_value = row[col]
                original_min, original_max = info['range']

                # Normalize the current slider value
                normalized_value = normalize_value(original_value, original_min, original_max)

                # Store the normalized value and its y-position
                normalized_values_to_plot.append(normalized_value)
                y_positions_to_plot.append(y_counter)

                y_counter += 1  # Increment y-position for the next slider

        # Plot the normalized values as points joined by a blue line
        ax.plot(normalized_values_to_plot, y_positions_to_plot, '-o',
                color='blue', markersize=8, linewidth=2, zorder=2)

        # --- Remove all visual clutter as per requirements ---
        ax.set_title('')  # Remove plot title
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.grid(False)  # Remove grid lines

        # Adjust y-axis limits to fit all plotted sliders comfortably
        # Ensure y_counter is at least 1 for proper range even with few sliders
        ax.set_ylim(-0.5, max(0.5, y_counter - 0.5))

        # Set x-axis limits to the normalized range with some padding
        ax.set_xlim(NORMALIZED_MIN - (NORMALIZED_MAX - NORMALIZED_MIN) * 0.1,
                    NORMALIZED_MAX + (NORMALIZED_MAX - NORMALIZED_MIN) * 0.05)

        # Construct the full path for the filename within the specified subfolder
        image_filename = os.path.join(output_folder, f'{profile_id}_{image_id}.png')

        # Save the figure with tight layout to remove extra whitespace
        plt.savefig(image_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)  # Close the figure to free up memory

    print(f"Images generated successfully and saved in the '{output_folder}' folder.")

except FileNotFoundError:
    print("Error: test_data.csv not found. Please make sure the file is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")