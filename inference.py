from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer from the saved checkpoint
tokenizer = AutoTokenizer.from_pretrained("checkpoints")
model = AutoModelForSeq2SeqLM.from_pretrained("checkpoints")

# Replace 'your_input_text' with the actual text you want to use
input_text = "predict: I have 30 apples. My friend has 20 apples. How many apples do we have total?"
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

# Generate a prediction
outputs = model.generate(**inputs)

# Convert the output to text
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# For example, print the predicted text
print(predicted_text)
