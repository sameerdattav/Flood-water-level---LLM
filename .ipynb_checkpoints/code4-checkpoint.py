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

# Function to generate structured output from the natural language query
def generate_structured_query(query):
    prompt = (
        "You are an assistant that converts questions about water levels into a structured JSON format."
        " The output must be a complete JSON object enclosed in curly braces {}."
        " The JSON object should include the following keys: 'action', 'road_id', and 'timestamp' (optional)."
        " The road_id should always start with 'Road_' followed by the exact road number from the query."
        " Handle potential spelling mistakes in the query."
        "\nHere are some examples of questions and their expected outputs:\n"
        "- Question: What is the higest water level on road 101?\n"
        "  Output: {\"action\": \"retrieve_max_water_level\", \"road_id\": \"Road_101\"}\n"
        "- Question: What was the water levl on road 102 at 2024-10-15 08:00:00?\n"
        "  Output: {\"action\": \"retrieve_water_level\", \"road_id\": \"Road_102\", \"timestamp\": \"2024-10-15 08:00:00\"}\n"
        "- Question: What are all the water levels on road 103?\n"
        "  Output: {\"action\": \"retrieve_all_water_levels\", \"road_id\": \"Road_103\"}\n"
        f"\nQuestion: {query}\n"
        "Output (just JSON): "
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    
    structured_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debugging: Print the generated structured query
    print(f"Generated structured query: {structured_query}")
    
    # Ensure the output is enclosed in curly braces
    if not structured_query.startswith("{"):
        structured_query = "{" + structured_query
    if not structured_query.endswith("}"):
        structured_query = structured_query + "}"
    
    # Post-process to correct road number if necessary
    try:
        query_data = json.loads(structured_query)
        road_match = re.search(r'road\s*(\d+)', query, re.IGNORECASE)
        if road_match:
            correct_road_id = f"Road_{road_match.group(1)}"
            if query_data["road_id"] != correct_road_id:
                query_data["road_id"] = correct_road_id
                structured_query = json.dumps(query_data)
                print(f"Corrected structured query: {structured_query}")
    except json.JSONDecodeError:
        pass  # If JSON parsing fails, we'll handle it in the execute_query function
    
    return structured_query

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
    
    except json.JSONDecodeError as e:
        return f"Error: Could not parse the structured query. Output was: {structured_query}. Error details: {str(e)}"
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
        
        # Generate structured query from the model
        structured_query = generate_structured_query(query)
        
        # Execute the structured query and print the result
        result = execute_query(structured_query)
        print(result)