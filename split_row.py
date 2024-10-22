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

import pandas as pd
import re

# Function to process each row and split into multiple rows if needed
def split_messages(row):
    # Define the regex pattern to match the message part after the sender with the timestamp
    pattern = r"([A-Za-z, ]+\([A-Za-z &-]+\) \d{2}:\d{2}:)"
    
    # Find all matches of the pattern
    matches = re.finditer(pattern, row['message'])
    
    new_rows = []
    last_pos = 0
    
    # Iterate through the matches and split the message
    for match in matches:
        # Get the position of the current match
        start_pos = match.start()
        
        if last_pos == 0:
            # First message, update the current row
            row['message'] = row['message'][last_pos:start_pos].strip()
            row['received'] = f"{row['received'].split()[0]} {match.group()[-6:-1]}"  # Extract first timestamp
            new_rows.append(row.copy())
        else:
            # For subsequent matches, create new rows
            new_row = row.copy()
            new_row['message'] = row['message'][last_pos:start_pos].strip()
            new_row['received'] = f"{row['received'].split()[0]} {match.group()[-6:-1]}"  # Extract the timestamp
            new_rows.append(new_row)
        
        last_pos = start_pos
    
    # Handle the last message after the last match
    if last_pos < len(row['message']):
        new_row = row.copy()
        new_row['message'] = row['message'][last_pos:].strip()
        new_rows.append(new_row)
    
    return new_rows

# Apply the split function to each row and expand the rows
def process_dataframe(df):
    new_rows = []
    for _, row in df.iterrows():
        new_rows.extend(split_messages(row))
    
    # Create a new DataFrame from the list of new rows
    return pd.DataFrame(new_rows)

# Example dataframe
df_cleaned = process_dataframe(df)

import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned DataFrame", dataframe=df_cleaned)


