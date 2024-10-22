import pandas as pd
import re

# Sample DataFrame
data = {'from': ['Williams, Megan (Fraud and Financial Crime - Commercial Banking Business Risk)'],
        'to': ['Williams, Megan (Fraud and Financial Crime -Commercial Banking Business Risk);Choudhary, Umair (Cardnet, GTB, Commercial Banking)'],
        'timestamp': ['06/01/2020 14:16:57'],
        'message': ['Williams, Megan (Fraud and Financial Crime - Commercial Banking Business Risk) 12:06: no. Hello, how are you? I need to set Rupesh up with access to Cast (old and new) and Isnap - can you help me with that at all? And Cobra! ind Choudhary, Umair (Cardnet, GTB, Commercial Banking) 14:04: Hi Megan i think i have admin access ill give it a bash now']}
df = pd.DataFrame(data)

# Function to clean and split messages
def clean_and_split_messages(row):
    message = row['message']
    
    # Check if the row already contains a single message
    if len(re.findall(r'\w+,\s\w+\s\(.+?\)\s\d{2}:\d{2}:', message)) <= 1:
        # No need to split, return the row as is
        return pd.DataFrame([row])

    # Regular expression to find messages by name and timestamp pattern
    message_parts = re.split(r'(\w+,\s\w+\s\(.+?\)\s\d{2}:\d{2}:)', message)

    # Clean the messages and prepare new rows
    new_rows = []
    current_sender = None
    for part in message_parts:
        # Identify if the part is a sender
        if re.match(r'\w+,\s\w+\s\(.+?\)\s\d{2}:\d{2}:', part):
            current_sender = part.strip()  # This will hold the sender and time
        else:
            if current_sender:
                # Clean and append the message
                clean_message = part.strip().replace('\n', ' ').replace('  ', ' ')
                timestamp = current_sender.split()[-1]
                sender = current_sender.rsplit(' ', 1)[0]
                
                new_rows.append({'from': row['from'], 'to': row['to'], 'timestamp': timestamp, 'message': clean_message})
                current_sender = None  # Reset sender for the next message

    return pd.DataFrame(new_rows)

# Apply function to the DataFrame
cleaned_messages = df.apply(clean_and_split_messages, axis=1)

# Concatenate all new rows into one DataFrame
final_df = pd.concat(cleaned_messages.tolist(), ignore_index=True)

import pandas as pd
import re

# Sample DataFrame
data = {'from': ['Williams, Megan (Fraud and Financial Crime - Commercial Banking Business Risk)'],
        'to': ['Williams, Megan (Fraud and Financial Crime -Commercial Banking Business Risk);Choudhary, Umair (Cardnet, GTB, Commercial Banking)'],
        'timestamp': ['06/01/2020 14:16:57'],
        'message': ['Williams, Megan (Fraud and Financial Crime - Commercial Banking Business Risk) 12:06: no. Hello, how are you? I need to set Rupesh up with access to Cast (old and new) and Isnap - can you help me with that at all? And Cobra! ind Choudhary, Umair (Cardnet, GTB, Commercial Banking) 14:04: Hi Megan i think i have admin access ill give it a bash now']}
df = pd.DataFrame(data)

# Function to clean and split messages
def clean_and_split_messages(row):
    message = row['message']
    
    # Regular expression to find name and timestamp patterns (e.g., "Name, Surname (Department) HH:MM:")
    message_parts = re.split(r'(\w+,\s\w+\s\(.+?\)\s\d{2}:\d{2}:)', message)
    
    # If there's only one part, assume it's already clean
    if len(message_parts) == 1:
        return pd.DataFrame([row])  # Return row unchanged if no splitting is needed

    # Clean the messages and prepare new rows
    new_rows = []
    current_sender = None
    for part in message_parts:
        # Identify if the part is a sender with timestamp
        if re.match(r'\w+,\s\w+\s\(.+?\)\s\d{2}:\d{2}:', part):
            current_sender = part.strip()  # This will hold the sender and time
        else:
            if current_sender:
                # Clean the message, remove any trailing whitespace or newlines
                clean_message = part.strip().replace('\n', ' ').replace('  ', ' ')
                if clean_message:  # Only add rows with non-empty messages
                    new_rows.append({
                        'from': row['from'],
                        'to': row['to'],
                        'timestamp': current_sender.split()[-1],  # Extract timestamp
                        'message': clean_message  # Only the clean message
                    })
                current_sender = None  # Reset sender for the next message

    return pd.DataFrame(new_rows)

# Apply function to the DataFrame
cleaned_messages = df.apply(clean_and_split_messages, axis=1)

# Concatenate all new rows into one DataFrame
final_df = pd.concat(cleaned_messages.tolist(), ignore_index=True)


