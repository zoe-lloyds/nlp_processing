import pandas as pd
import re

# Function to clean each row
def clean_row(row):
    message = row['message']
    
    # Find all the timestamps in the message (e.g., HH:MM)
    timestamps = re.findall(r'\d{2}:\d{2}', message)
    
    if len(timestamps) > 1:
        # If there are multiple timestamps, split the message
        split_message = re.split(r'\d{2}:\d{2}', message, maxsplit=1)
        row['received'] = f"{row['received'].split()[0]} {timestamps[0]}"  # Keep the first timestamp
        row['message'] = f"{split_message[0].strip()}.{split_message[1].strip()}"
    
    return row

import pandas as pd
import re

# Function to clean each row
def clean_row(row):
    message = row['message']
    
    # Define the regex pattern to match the message part after the sender with the timestamp (e.g., "Williams, Megan (Description) HH:MM:")
    pattern = r"([A-Za-z, ]+\([A-Za-z &-]+\) \d{2}:\d{2}:)"
    
    # Find all occurrences of this pattern in the message
    matches = re.findall(pattern, message)
    
    if matches:
        # We assume the first match is the main message timestamp, so we split the message after the first occurrence
        split_message = re.split(pattern, message, maxsplit=1)
        row['received'] = f"{row['received'].split()[0]} {split_message[1][-6:-1]}"  # Extracting the first timestamp
        row['message'] = f"{split_message[2].strip()}"
    
    return row

# Apply the cleaning function row-wise to your dataframe
df_cleaned = df.apply(clean_row, axis=1)

import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned DataFrame", dataframe=df_cleaned)

