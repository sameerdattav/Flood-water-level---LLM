from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import json
import torch
import re


# Load the CSV file containing water level data
df = pd.read_csv('road_water_levels_large.csv', parse_dates=['Timestamp'])

# Load a pre-trained model and tokenizer from Hugging Face
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_road_number(query):
    """Extract road number from query text"""
    road_match = re.search(r'road\s*(\d+)', query, re.IGNORECASE)
    return f"Road_{road_match.group(1)}" if road_match else None

def determine_query_type(query):
    """Determine the type of query based on keywords"""
    query = query.lower()
    if 'average' in query or 'avg' in query or 'mean' in query:
        return "retrieve_average_water_level"
    elif 'maximum' in query or 'max' in query or 'highest' in query:
        if 'between' in query or 'from' in query:
            return "retrieve_max_water_level_in_range"
        return "retrieve_max_water_level"
    elif 'minimum' in query or 'min' in query or 'lowest' in query:
        return "retrieve_min_water_level"
    elif 'latest' in query or 'recent' in query or 'current' in query:
        return "retrieve_latest_water_level"
    elif 'between' in query or 'from' in query:
        return "retrieve_all_water_levels_in_range"
    return "retrieve_water_level"

def generate_structured_query(query):
    """Generate structured query directly without using the model"""
    road_id = extract_road_number(query)
    if not road_id:
        return json.dumps({"error": "No road number found in query"})

    action = determine_query_type(query)
    
    # Basic query structure
    structured_query = {
        "action": action,
        "road_id": road_id
    }

    # Extract timestamps if present
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?'
    timestamps = re.findall(timestamp_pattern, query)
    
    if timestamps:
        if len(timestamps) >= 2 and ('between' in query.lower() or 'from' in query.lower()):
            structured_query["start_timestamp"] = timestamps[0]
            structured_query["end_timestamp"] = timestamps[1]
        else:
            structured_query["timestamp"] = timestamps[0]

    return json.dumps(structured_query)

def format_water_level(value):
    """Format water level values consistently"""
    return f"{float(value):.2f}" if pd.notnull(value) else "N/A"

def find_timestamp_for_value(df, road_id, value):
    """Find the timestamp(s) when a specific water level occurred"""
    matches = df[df[road_id] == value]['Timestamp']
    return matches.iloc[0] if not matches.empty else None

def execute_query(structured_query):
    """Execute the structured query on the CSV data"""
    try:
        query_data = json.loads(structured_query)
        
        if "error" in query_data:
            return query_data["error"]

        road_id = query_data.get("road_id")
        available_roads = [col for col in df.columns if col.startswith('Road_')]
        
        if not road_id or road_id not in available_roads:
            return (f"Error: Invalid or missing road ID. Available roads are: "
                   f"{', '.join(available_roads)}")

        action = query_data.get("action", "")

        if action == "retrieve_max_water_level":
            max_level = df[road_id].max()
            max_timestamp = find_timestamp_for_value(df, road_id, max_level)
            return f"The highest water level on {road_id} was {format_water_level(max_level)} meters on {max_timestamp}."

        elif action == "retrieve_average_water_level":
            avg_level = df[road_id].mean()
            return (f"The average water level on {road_id} was {format_water_level(avg_level)} meters "
                   f"(calculated from {df['Timestamp'].min()} to {df['Timestamp'].max()}).")

        elif action == "retrieve_min_water_level":
            min_level = df[road_id].min()
            min_timestamp = find_timestamp_for_value(df, road_id, min_level)
            return f"The minimum water level on {road_id} was {format_water_level(min_level)} meters on {min_timestamp}."

        elif action == "retrieve_latest_water_level":
            latest_row = df[['Timestamp', road_id]].dropna().iloc[-1]
            return f"The latest water level on {road_id} at {latest_row['Timestamp']} was {format_water_level(latest_row[road_id])} meters."

        elif action == "retrieve_water_level":
            timestamp = query_data.get("timestamp")
            if not timestamp:
                # If no timestamp provided, return the latest reading
                latest_row = df[['Timestamp', road_id]].dropna().iloc[-1]
                return f"The latest water level on {road_id} at {latest_row['Timestamp']} was {format_water_level(latest_row[road_id])} meters."
            
            timestamp = pd.to_datetime(timestamp)
            row = df[df['Timestamp'] == timestamp]
            if not row.empty:
                water_level = format_water_level(row[road_id].values[0])
                return f"Water level on {road_id} at {timestamp} was {water_level} meters."
            return f"No data available for {road_id} at {timestamp}."

        elif action == "retrieve_all_water_levels_in_range":
            start_timestamp = pd.to_datetime(query_data.get("start_timestamp"))
            end_timestamp = pd.to_datetime(query_data.get("end_timestamp"))
            range_data = df[(df['Timestamp'] >= start_timestamp) & 
                          (df['Timestamp'] <= end_timestamp)]
            if range_data.empty:
                return f"No data available for {road_id} in the specified range."
            
            # Include summary statistics with the range data
            data_output = range_data[['Timestamp', road_id]].to_string(index=False)
            summary = (f"\n\nSummary for {road_id} from {start_timestamp} to {end_timestamp}:"
                      f"\nMinimum: {format_water_level(range_data[road_id].min())} meters"
                      f"\nMaximum: {format_water_level(range_data[road_id].max())} meters"
                      f"\nAverage: {format_water_level(range_data[road_id].mean())} meters"
                      f"\nTotal readings: {len(range_data)}")
            return data_output + summary

        elif action == "retrieve_max_water_level_in_range":
            start_timestamp = pd.to_datetime(query_data.get("start_timestamp"))
            end_timestamp = pd.to_datetime(query_data.get("end_timestamp"))
            range_data = df[(df['Timestamp'] >= start_timestamp) & 
                          (df['Timestamp'] <= end_timestamp)]
            if range_data.empty:
                return f"No data available for {road_id} in the specified range."
            
            max_level = range_data[road_id].max()
            max_timestamp = find_timestamp_for_value(range_data, road_id, max_level)
            return (f"The maximum water level on {road_id} between {start_timestamp} and {end_timestamp} "
                   f"was {format_water_level(max_level)} meters on {max_timestamp}.")

        return "I couldn't understand your query. Please try rephrasing it."

    except Exception as e:
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    print("\n=== Water Levels Query System ===")
    print("\nYou can ask questions like:")
    print("- What is the highest water level on road 101?")
    print("- What was the water level on road 102 at 2024-10-15 08:00:00?")
    print("- What is the average water level on road 103?")
    print("- What's the minimum water level on road 104?")
    print("- What's the latest water level reading for road 105?")
    print("- Show me all water levels on road 106 from 2024-10-01 00:00:00 to 2024-10-05 23:59:59")
    print("- What was the maximum water level on road 107 between 2024-10-10 08:00:00 and 2024-10-12 18:00:00?")
    # print("\nAvailable roads:", ", ".join([col for col in df.columns if col.startswith('Road_')]))
    road_columns = [col for col in df.columns if col.startswith('Road_')]
    print("Available roads range from:", road_columns[0], "to", road_columns[-1])

# Displaying only the start and end of the timestamps
    print("Available timestamps range from:", df['Timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'), 
      "to", df['Timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'))
    print("\nType 'exit' to stop.")
   
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'exit':
                print("Thank you for using the Water Levels Query System. Goodbye!")
                break
            
            if not query:
                print("Please enter a valid query.")
                continue
            
            structured_query = generate_structured_query(query)
            result = execute_query(structured_query)
            print("\nResult:", result)
            
        except KeyboardInterrupt:
            print("\nProgram terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")