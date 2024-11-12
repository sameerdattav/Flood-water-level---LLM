import pandas as pd
import spacy

# Load the CSV file
df = pd.read_csv('road_water_levels.csv', parse_dates=['Timestamp'])

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

# Function to extract entities from the query
def parse_query(query):
    doc = nlp(query)
    road_id = None
    timestamp = None
    for ent in doc.ents:
        if ent.label_ == "DATE":
            timestamp = pd.to_datetime(ent.text)
        if ent.label_ == "CARDINAL":
            road_id = f"Road_{ent.text}"
    return road_id, timestamp

# Function to get water level based on the parsed query
def get_water_level(query):
    road_id, timestamp = parse_query(query)
    if road_id and timestamp:
        # Check if the timestamp is in the DataFrame
        row = df[df['Timestamp'] == timestamp]
        if not row.empty:
            water_level = row[road_id].values[0]
            return f"Water level on {road_id} at {timestamp} was {water_level} meters."
        else:
            return "No data available for the specified time."
    else:
        return "Could not understand the query. Please specify road number and time."

# Take user input
query = input("Please enter your query (e.g., 'What was the water level on road 101 at 12:00 PM on 15th October?'): ")

# Process the query and output the result
response = get_water_level(query)
print(response)
