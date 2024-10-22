import pandas as pd
import re

# Sample data
data = {
    'from': ['Williams, Megan (Fraud and Financial Crime - Commercial Banking Business Risk)'],
    'to': ['Williams, Megan (Fraud and Financial Crime -Commercial Banking Business Risk);Choudhary, Umair (Cardnet, GTB, Commercial Banking)'],
    'received': ['06/01/2020 14:16:57'],
    'message': [
        'Williams, Megan (Fraud and Financial Crime - Commercial Banking Business Risk) 12:06:\n'
        'no\n'
        'Hello, how are you? I need to set Rupesh up with access to Cast (old and new) and Isnap - can you help me with that at all?\n'
        'And Cobra!\n'
        'ind\n'
        'Choudhary, Umair (Cardnet, GTB, Commercial Banking) 14:04:\n'
        'Hi Megan\n'
        'i think i have admin access\n'
        'ill give it a bash now'
    ]
}

df = pd.DataFrame(data)

# Function to clean the messages
def extract_messages(row):
    # Regex to find messages with timestamps
    messages = re.split(r'(?=\w+, \w+ \(\w+.*?\d{2}:\d{2}:\d{2}\))', row['message'])
    cleaned_messages = []
    
    for message in messages:
        # Clean message and strip unnecessary whitespaces
        message = message.strip()
        if message:
            sender, msg = message.split(' ', 1)  # Split at first space
            cleaned_messages.append((sender, msg.strip()))
    
    return cleaned_messages

# Apply the function to extract messages
message_data = df.apply(extract_messages, axis=1)

# Expand the results into a new DataFrame
cleaned_rows = []
for index, messages in enumerate(message_data):
    for sender, msg in messages:
        cleaned_rows.append({
            'from': sender,
            'to': df.at[index, 'to'],
            'received': df.at[index, 'received'],
            'message': msg
        })

cleaned_df = pd.DataFrame(cleaned_rows)

# Display the cleaned DataFrame
print(cleaned_df)
