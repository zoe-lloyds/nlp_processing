# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Filter messages sent by the male subject
male_messages = df[df['From'] == 'Male Subject']

# Total messages sent
total_messages = male_messages.shape[0]

# Messages per day
messages_per_day = male_messages.resample('D', on='Timestamp').size()

# Display the results
print(f'Total messages sent: {total_messages}')
print(messages_per_day.describe())

# Plot messages per day
messages_per_day.plot(kind='bar', title='Messages per Day')
# Define work hours
work_start = 9
work_end = 17

# Extract hour from Timestamp
male_messages['Hour'] = male_messages['Timestamp'].dt.hour

# Messages during work hours
work_hours_messages = male_messages[(male_messages['Hour'] >= work_start) & (male_messages['Hour'] < work_end)]

# Compare work and non-work hours
work_hours_count = work_hours_messages.shape[0]
non_work_hours_count = total_messages - work_hours_count

print(f'Messages during work hours: {work_hours_count}')
print(f'Messages during non-work hours: {non_work_hours_count}')

# Plot comparison
import matplotlib.pyplot as plt

labels = ['Work Hours', 'Non-Work Hours']
sizes = [work_hours_count, non_work_hours_count]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Messages during Work vs Non-Work Hours')
plt.show()

# Top recipients
top_recipients = male_messages['To'].value_counts().head()

print('Top recipients:')
print(top_recipients)

# Message length analysis
male_messages['MessageLength'] = male_messages['text'].apply(len)
message_length_stats = male_messages['MessageLength'].describe()

print('Message length statistics:')
print(message_length_stats)

# Plot message length distribution
male_messages['MessageLength'].plot(kind='hist', bins=20, title='Message Length Distribution')

summary_report = {
    'Total Messages Sent': total_messages,
    'Average Messages Per Day': messages_per_day.mean(),
    'Messages During Work Hours': work_hours_count,
    'Messages During Non-Work Hours': non_work_hours_count,
    'Number of Inappropriate Messages': inappropriate_count,
    'Top Recipients': top_recipients.to_dict(),
    'Message Length Statistics': message_length_stats.to_dict()
}

print('Summary Report:')
for key, value in summary_report.items():
    print(f'{key}: {value}')
