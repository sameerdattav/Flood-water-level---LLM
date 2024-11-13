  road_columns = [col for col in df.columns if col.startswith('Road_')]
    print("Available roads range from:", road_columns[0], "to", road_columns[-1])

# Displaying only the start and end of the timestamps
    print("Available timestamps range from:", df['Timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'), 
      "to", df['Timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'))
    print("\nType 'exit' to stop.")