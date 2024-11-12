import pandas as pd
import json
import re
from datetime import datetime

# Load the CSV file containing water level data
df = pd.read_csv('road_water_levels.csv', parse_dates=['Timestamp'])

# Function to generate structured output from the natural language queryw
def generate_structured_query(query):
    structured_query = {}
    
    # Use regex to extract road number, allowing for "Road_" prefix
    road_match = re.search(r'(?:road|Road_)\s*(\d+)', query, re.IGNORECASE)
    if road_match:
        road_id = f"Road_{road_match.group(1)}"
    else:
        road_id = None

    if "highest water level" in query.lower() and road_id:
        structured_query["action"] = "retrieve_max_water_level"
        structured_query["road_id"] = road_id
    elif "water level" in query.lower() and "at" in query.lower() and road_id:
        structured_query["action"] = "retrieve_water_level"
        structured_query["road_id"] = road_id
        timestamp_match = re.search(r'at\s+([\d-: ]+)', query)
        structured_query["timestamp"] = timestamp_match.group(1) if timestamp_match else None
    elif "all the water levels" in query.lower() and road_id:
        structured_query["action"] = "retrieve_all_water_levels"
        structured_query["road_id"] = road_id
    else:
        structured_query["action"] = "unknown"
    
    return json.dumps(structured_query)

# Function to execute the structured query on the CSV data
def execute_query(structured_query):
    try:
        # Parse the structured query as JSON
        query_data = json.loads(structured_query)
        road_id = query_data.get("road_id")

        if not road_id or road_id not in df.columns:
            return f"Error: Invalid or missing road ID. Available roads are: {', '.join([col for col in df.columns if col.startswith('Road_')])}"

        if query_data.get("action") == "retrieve_max_water_level":
            max_level = df[road_id].max()
            return f"The highest water level on {road_id} was {max_level} meters."
        
        elif query_data.get("action") == "retrieve_water_level":
            timestamp = query_data.get("timestamp")
            if not timestamp:
                return "Error: No timestamp provided for the water level query."
            try:
                timestamp = pd.to_datetime(timestamp)
                row = df[df['Timestamp'] == timestamp]
                if not row.empty:
                    water_level = row[road_id].values[0]
                    return f"Water level on {road_id} at {timestamp} was {water_level} meters."
                else:
                    return f"No data available for {road_id} at the specified time: {timestamp}. Available timestamps are: {', '.join(df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist())}"
            except ValueError:
                return f"Error: Invalid timestamp format. Please use YYYY-MM-DD HH:MM:SS."
        
        elif query_data.get("action") == "retrieve_all_water_levels":
            levels = df[['Timestamp', road_id]]
            return levels.to_string(index=False)
        
        else:
            return "I'm sorry, I couldn't understand your query. Please try rephrasing it."
    
    except json.JSONDecodeError:
        return f"Error: Could not parse the structured query. Output was: {structured_query}"
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Main loop to take user queries and process them
if __name__ == "__main__":
    print("Welcome! Ask me anything about the water levels on different roads.")
    print("Available roads are:", ", ".join([col for col in df.columns if col.startswith('Road_')]))
    print("Available timestamps are:", ", ".join(df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()))
    print("Type 'exit' to stop.")
    
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        # Generate structured query
        structured_query = generate_structured_query(query)
        
        # Execute the structured query and print the result
        result = execute_query(structured_query)
        print(result)