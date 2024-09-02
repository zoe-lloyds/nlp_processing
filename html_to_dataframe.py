import os
import pandas as pd
from bs4 import BeautifulSoup

# Define the path to the folder containing HTML files
folder_path = 'path_to_your_folder'

# Initialize an empty list to store data
data = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.html'):
        file_path = os.path.join(folder_path, filename)
        
        # Read the content of the HTML file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract the data you need (example: extract all <p> tags content)
        # This depends on your specific HTML structure
        for message in soup.find_all('div', class_='message'):
            user = message.find('span', class_='user').get_text(strip=True)
            time = message.find('span', class_='time').get_text(strip=True)
            text = message.find('div', class_='text').get_text(strip=True)
            
            # Append the extracted data as a tuple to the list
            data.append((user, time, text))

# Convert the list into a DataFrame
df = pd.DataFrame(data, columns=['User', 'Time', 'Message'])

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)
