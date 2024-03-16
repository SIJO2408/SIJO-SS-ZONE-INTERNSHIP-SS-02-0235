pip install transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define conversation history
conversation_history = []

# Define function to generate response
def generate_response(user_input, max_length=50):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    bot_input_ids = input_ids
    with torch.no_grad():
        output = model.generate(bot_input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Start conversation loop
print("Chatbot: Hello! How can I assist you today?")

while True:
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break

    conversation_history.append("You: " + user_input)
    response = generate_response(user_input)
    conversation_history.append("Chatbot: " + response)
    print("Chatbot:", response)
