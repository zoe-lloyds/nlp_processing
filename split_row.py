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

# Example dataframe
data = {
    'from': ["Williams, Megan (Fraud and Financial Crime - Commercial Banking Business Risk)"],
    'to': ["Williams, Megan (Fraud and Financial Crime -Commercial Banking Business Risk);Choudhary, Umair (Cardnet, GTB, Commercial Banking)"],
    'received': ["06/01/2020 14:16:57"],
    'message': ["Williams, Megan (Fraud and Financial Crime - Commercial Banking Business Risk) 12:06: no Hello, how are you? I need to set Rupesh up with access to Cast (old and new) and Isnap - can you help me with that at all? And Cobra! ind Choudhary, Umair (Cardnet, GTB, Commercial Banking) 14:04: Hi Megan i think i have admin access ill give it a bash now"]
}

df = pd.DataFrame(data)

# Apply the cleaning function row-wise
df_cleaned = df.apply(clean_row, axis=1)

import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned DataFrame", dataframe=df_cleaned)
