import pandas as pd
from datetime import timedelta

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

def calculate_active_time_and_messages_per_day(df, person, max_interval_minutes=5, working_hours_per_day=7):
    # Filter messages by the person
    person_messages = df[df['From'] == person].copy()
    if person_messages.empty:
        return f"No messages found for {person}"

    # Sort messages by timestamp
    person_messages.sort_values(by='Timestamp', inplace=True)

    # Calculate the time difference between consecutive messages
    person_messages['TimeDiff'] = person_messages['Timestamp'].diff()

    # Define the maximum interval to consider messages as part of the same active period
    max_interval = timedelta(minutes=max_interval_minutes)

    # Identify new active periods
    person_messages['NewPeriod'] = person_messages['TimeDiff'] > max_interval

    # Calculate active time per day by summing time differences within active periods
    person_messages['ActiveTime'] = person_messages['TimeDiff'].where(person_messages['NewPeriod'] == False)
    
    # Fill NaT values with 0 active time for new periods
    person_messages['ActiveTime'] = person_messages['ActiveTime'].fillna(timedelta(0))

    # Group by date and calculate total active time and message count
    summary = person_messages.groupby(person_messages['Timestamp'].dt.date).agg(
        TotalActiveTime=pd.NamedAgg(column='ActiveTime', aggfunc='sum'),
        MessageCount=pd.NamedAgg(column='Message', aggfunc='size')
    )

    # Calculate the most frequently messaged recipient each day
    most_messaged = person_messages.groupby(person_messages['Timestamp'].dt.date)['To'].agg(lambda x: x.value_counts().idxmax())
    summary['MostMessagedPerson'] = most_messaged

    # Calculate the proportion of the working day spent messaging
    summary['WorkingDayHours'] = working_hours_per_day
    summary['ProportionOfDayMessaging'] = summary['TotalActiveTime'] / timedelta(hours=working_hours_per_day)
    
    return summary.reset_index().rename(columns={'index': 'Date'})

# Example usage
person = 'Alice'
summary_df = calculate_active_time_and_messages_per_day(df, person)

print(summary_df)


### New method
import pandas as pd
from datetime import timedelta

# Sample data (replace this with your actual data)
data = {
    'From': ['Alice', 'Alice', 'Alice', 'Alice', 'Alice', 'Tom', 'Tom', 'Alice', 'Alice', 'Alice'],
    'To': ['Tom', 'Tom', 'Tom', 'Tom', 'Tom', 'Alice', 'Alice', 'Tom', 'Tom', 'Tom'],
    'Timestamp': [
        '2024-07-24 09:00', '2024-07-24 09:05', '2024-07-24 09:10',
        '2024-07-24 09:20', '2024-07-24 09:25', '2024-07-24 09:30',
        '2024-07-24 09:40', '2024-07-24 09:45', '2024-07-24 09:50',
        '2024-07-24 10:00'
    ],
    'Message': ['Hi', 'How are you?', 'Meeting at 10?', 'See you there', 'Okay', 'Hello', 'Hi Tom', 'Let\'s discuss', 'Sure', 'On it']
}
df = pd.DataFrame(data)

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

def calculate_messages_and_active_time(df, person, working_hours_per_day=7):
    # Filter messages by the person
    person_messages = df[df['From'] == person].copy()
    if person_messages.empty:
        return f"No messages found for {person}"

    # Sort messages by timestamp
    person_messages.sort_values(by='Timestamp', inplace=True)

    # Calculate the time difference between consecutive messages
    person_messages['TimeDiff'] = person_messages['Timestamp'].diff()

    # Replace NaT (first message) with 0
    person_messages['TimeDiff'] = person_messages['TimeDiff'].fillna(timedelta(0))

    # Group by date and calculate total active time and message count
    summary = person_messages.groupby(person_messages['Timestamp'].dt.date).agg(
        TotalActiveTime=pd.NamedAgg(column='TimeDiff', aggfunc='sum'),
        MessageCount=pd.NamedAgg(column='Message', aggfunc='size')
    )

    # Calculate the proportion of the working day spent messaging
    summary['WorkingDayHours'] = working_hours_per_day
    summary['ProportionOfDayMessaging'] = summary['TotalActiveTime'] / timedelta(hours=working_hours_per_day)
    
    return summary.reset_index().rename(columns={'index': 'Date'})

# Example usage
person = 'Alice'
summary_df = calculate_messages_and_active_time(df, person)

print(summary_df)


