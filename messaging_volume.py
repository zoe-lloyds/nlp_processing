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

def calculate_active_time_and_messages_per_day(df, person, max_interval_minutes=5):
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

    return total_active_time, messages_per_day

# Example usage
person = 'Alice'
total_active_time, messages_per_day = calculate_active_time_and_messages_per_day(df, person)

print(f"Total active time for {person}: {total_active_time}")
print(f"Messages per day for {person}:\n{messages_per_day}")



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
