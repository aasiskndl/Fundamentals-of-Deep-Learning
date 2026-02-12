import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

#managing secrets
api_key = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL")

client = Groq(api_key=api_key)   # initialize the model

def get_weather(location):
    """Simple mock function to simulate a weather API"""
    if "kathmandu" in location.lower():
        return json.dumps({"location": "Kathmandu", "temperature": "22°C", "condition": "Sunny"})
    return json.dumps({"location": location, "temperature": "unknown"})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specific city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name, e.g., San Francisco"}
                },
                "required": ["location"],
            },
        },
    }
]

messages = [{"role": "user", "content": "What is the weather like in Kathmandu?"}]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls


# handle the Tool Call
if tool_calls:
    # add ai's request to message history
    messages.append(response_message)
    
    for tool_call in tool_calls:
        function_args = json.loads(tool_call.function.arguments)
        # Execute the local function
        result = get_weather(location=function_args.get("location"))
        
        # add result to information
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": "get_weather",
            "content": result,
        })

    # added final response to the model's messages
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    print(final_response.choices[0].message.content)