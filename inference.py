from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer from the saved checkpoint
tokenizer = AutoTokenizer.from_pretrained("checkpoints")
model = AutoModelForSeq2SeqLM.from_pretrained("checkpoints")

for _ in range(10):
    prompt = input("enter a prompt: ")
    pred_text = "predict: {}".format(prompt)
    expl_text = "explain: {}".format(prompt)
    inputs = tokenizer(pred_text, return_tensors="pt", truncation=True, padding=True)
    expls = tokenizer(expl_text, return_tensors="pt", truncation=True, padding=True)

    # Generate a prediction
    outputs = model.generate(**inputs, max_new_tokens=32)
    expl_outputs = model.generate(**expls, max_new_tokens=32)

    # Convert the output to text
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    expl_out_text = tokenizer.decode(expl_outputs[0], skip_special_tokens=True)

    # For example, print the predicted text
    print(expl_out_text)
    print("the answer is: ", predicted_text)
