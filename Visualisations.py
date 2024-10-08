import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the results are sorted by date
summary_df = summary_df.sort_values(by='Timestamp')
# Convert TotalActiveTime to hours
summary_df['TotalActiveTime'] = summary_df['TotalActiveTime'].apply(lambda x: x.total_seconds() / 3600)

# Bar Chart of Daily Message Count
plt.figure(figsize=(10, 6))
sns.barplot(x=summary_df['Timestamp'].astype(str), y=summary_df['MessageCount'])
plt.xlabel('Date')
plt.ylabel('Message Count')
plt.title(f'Daily Message Count for {person}')
plt.xticks(rotation=45)
plt.show()

# Line Chart of Total Active Time per Day
plt.figure(figsize=(10, 6))
plt.plot(summary_df['Timestamp'].astype(str), summary_df['TotalActiveTime'].dt.total_seconds() / 3600, marker='o')
plt.xlabel('Date')
plt.ylabel('Total Active Time (hours)')
plt.title(f'Total Active Time per Day for {person}')
plt.xticks(rotation=45)
plt.show()

# Pie Chart of Proportion of Day Messaging
plt.figure(figsize=(8, 8))
daily_proportion = summary_df['ProportionOfDayMessaging'].mean()
labels = ['Messaging', 'Other Activities']
sizes = [daily_proportion, 1 - daily_proportion]
colors = ['skyblue', 'lightgray']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title(f'Proportion of Workday Spent Messaging for {person}')
plt.show()

# Heatmap of Active Periods
# For the heatmap, create an hourly distribution of messages
heatmap_data = pd.DataFrame(hourly_distribution.values.reshape(-1, 1), columns=['MessageCount'])

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data.T, cmap='Blues', annot=True, cbar=False, fmt='g', xticklabels=hourly_distribution.index, yticklabels=['Messages'])
plt.xlabel('Hour of the Day')
plt.ylabel('Activity')
plt.title(f'Hourly Distribution of Messages for {person}')
plt.show()
plt.title(f'Hourly Distribution of Messages for {person}')
plt.show()
