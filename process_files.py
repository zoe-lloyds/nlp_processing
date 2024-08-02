import pandas as pd

def process_excel_files(file_paths, person_of_interest):
    """
    Reads Excel files, drops duplicate rows, and filters rows by person of interest.

    Parameters:
        file_paths (list of str): List of paths to Excel files.
        person_of_interest (str): Name of the person of interest to filter rows.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    combined_df = pd.DataFrame()

    # Read each file and concatenate them into a single DataFrame
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Drop duplicate rows
    combined_df.drop_duplicates(inplace=True)

    # Filter rows where the person of interest is in 'from' or 'to' columns
    filtered_df = combined_df[(combined_df['from'].str.contains(person_of_interest, na=False)) | 
                              (combined_df['to'].str.contains(person_of_interest, na=False))]

    return filtered_df

# Example usage
file_paths = ['file1.xlsx', 'file2.xlsx']  # Replace with your actual file paths
person_of_interest = 'Alice'

processed_df = process_excel_files(file_paths, person_of_interest)

# Print the processed DataFrame
print(processed_df)
