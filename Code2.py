from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import json

# Load the CSV file containing water level data
df = pd.read_csv('road_water_levels.csv', parse_dates=['Timestamp'])

# Load a pre-trained model and tokenizer from Hugging Face
model_name = "google/flan-t5-small"  # T5 model for generating structured responses
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to generate structured output from the natural language query
def generate_structured_query(query):
    prompt = f"Convert the following question into a JSON format with keys: action, road_id, and timestamp.\nQuestion: {query}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Set max_new_tokens to a suitable value (e.g., 50)
    outputs = model.generate(**inputs, max_new_tokens=50)
    structured_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debugging: Print the generated structured query
    print(f"Generated structured query: {structured_query}")
    return structured_query

# Function to execute the structured query on the CSV data
def execute_query(structured_query):
    try:
        # Parse the structured query as JSON
        query_data = json.loads(structured_query)

        if query_data.get("action") == "retrieve_max_water_level":
            road_id = query_data["road_id"]
            max_level = df[road_id].max()
            return f"The highest water level on {road_id} was {max_level} meters."
        
        elif query_data.get("action") == "retrieve_water_level":
            road_id = query_data["road_id"]
            timestamp = pd.to_datetime(query_data["timestamp"])
            row = df[df['Timestamp'] == timestamp]
            if not row.empty:
                water_level = row[road_id].values[0]
                return f"Water level on {road_id} at {timestamp} was {water_level} meters."
            else:
                return "No data available for the specified time."
        
        elif query_data.get("action") == "retrieve_all_water_levels":
            road_id = query_data["road_id"]
            levels = df[['Timestamp', road_id]]
            return levels.to_string(index=False)
        
        else:
            return "Unknown action."
    
    except json.JSONDecodeError:
        return "Error: Could not parse the structured query."
    except Exception as e:
        return f"Error processing query: {e}"

# Main loop to take user queries and process them
if __name__ == "__main__":
    print("Welcome! Ask me anything about the water levels on different roads.")
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
