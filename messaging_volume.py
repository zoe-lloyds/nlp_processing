import pandas as pd
from datetime import timedelta

# Sample data (replace this with your actual data)
data = {
    'from': ['Alice', 'Alice', 'Alice', 'Alice', 'Alice', 'Tom', 'Tom', 'Alice', 'Alice', 'Alice'],
    'to': ['Tom', 'Tom', 'Tom', 'Tom', 'Tom', 'Alice', 'Alice', 'Tom', 'Tom', 'Tom'],
    'timestamp': [
        '2024-07-24 09:00', '2024-07-24 09:05', '2024-07-24 09:10',
        '2024-07-24 09:20', '2024-07-24 09:25', '2024-07-24 09:30',
        '2024-07-24 09:40', '2024-07-24 09:45', '2024-07-24 09:50',
        '2024-07-24 10:00'
    ],
    'message': ['Hi', 'How are you?', 'Meeting at 10?', 'See you there', 'Okay', 'Hello', 'Hi Tom', 'Let\'s discuss', 'Sure', 'On it']
}
df = pd.DataFrame(data)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

def calculate_messaging_statistics(df, person, max_interval_minutes=5):
    # Filter messages by the person
    person_messages = df[df['from'] == person].copy()
    if person_messages.empty:
        return f"No messages found for {person}"

    # Sort messages by timestamp
    person_messages.sort_values(by='timestamp', inplace=True)

    # Calculate the time difference between consecutive messages
    person_messages['time_diff'] = person_messages['timestamp'].diff()

    # Define the maximum interval to consider messages as part of the same active period
    max_interval = timedelta(minutes=max_interval_minutes)

    # Identify new active periods
    person_messages['new_period'] = person_messages['time_diff'] > max_interval

    # Calculate total active time by summing time differences within active periods
    person_messages['active_time'] = person_messages['time_diff'].where(person_messages['new_period'] == False)
    total_active_time = person_messages['active_time'].sum()

    # Count messages per day
    messages_per_day = person_messages.groupby(person_messages['timestamp'].dt.date).size()

    # Calculate the proportion of time spent messaging during a working day
    total_working_minutes = 7 * 60  # 7 hours working day (excluding 1-hour lunch break)
    active_minutes = total_active_time.total_seconds() / 60
    proportion_of_working_day_messaging = active_minutes / total_working_minutes

    return total_active_time, messages_per_day, proportion_of_working_day_messaging

# Example usage
person = 'Alice'
total_active_time, messages_per_day, proportion_of_working_day_messaging = calculate_messaging_statistics(df, person)

print(f"Total active time for {person}: {total_active_time}")
print(f"Messages per day for {person}:\n{messages_per_day}")
print(f"Proportion of the working day spent messaging: {proportion_of_working_day_messaging:.2%}")


import pandas as pd
from datetime import datetime

# Sample data
data = {
    'from': ['Alice', 'Tom', 'Jerry', 'Alice', 'Tom', 'Alice', 'Jerry', 'Alice', 'Tom', 'Alice'],
    'to': ['Tom;Jerry', 'Alice;Jerry', 'Alice;Tom', 'Tom;Jerry', 'Alice;Jerry', 'Tom;Jerry', 'Alice;Tom', 'Tom;Jerry', 'Alice;Jerry', 'Tom;Jerry'],
    'timestamp': [
        '2024-07-24 10:00', '2024-07-24 10:05', '2024-07-24 10:10',
        '2024-07-24 10:15', '2024-07-24 10:20', '2024-07-24 11:00',
        '2024-07-24 11:15', '2024-07-24 12:00', '2024-07-24 12:05',
        '2024-07-24 13:00'
    ],
    'message': ['Hello everyone!', 'Hi Alice!', 'Hello Tom!', 'How are you all?', 'Good thanks!', 'Let’s meet', 'Sure!', 'At 2 PM?', 'Works for me', 'See you then']
}
df = pd.DataFrame(data)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

def calculate_time_spent_and_messages(df, person):
    # Filter messages by the person
    person_messages = df[df['from'] == person]

    if person_messages.empty:
        return f"No messages found for {person}"

    # Calculate the total time span
    first_message = person_messages['timestamp'].min()
    last_message = person_messages['timestamp'].max()
    total_time_spent = last_message - first_message

    # Calculate number of messages per day
    person_messages['date'] = person_messages['timestamp'].dt.date
    messages_per_day = person_messages.groupby('date').size()

    # Print results
    print(f"Total time spent messaging by {person}: {total_time_spent}")
    print(f"Number of messages per day by {person}:\n{messages_per_day}")
    return total_time_spent, messages_per_day

# Example usage
person = 'Alice'
calculate_time_spent_and_messages(df, person)
