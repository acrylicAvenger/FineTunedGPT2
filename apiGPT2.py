import torch.cuda
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # enable CORS for all routes

# Load the saved model and tokenizer
model_path = "C:\\Users\\Darshan\\PycharmProjects\\huggingFaceGPT\\savedModel"
tokenizer_path = "C:\\Users\\Darshan\\PycharmProjects\\huggingFaceGPT\\savedModel"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizers = GPT2Tokenizer.from_pretrained(tokenizer_path)

# set the device to GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


@app.route("/generate", methods=['POST'])
def generate_prompt():
    input_text = request.json["input_text"]
    # if request.method == "POST":
    #     input_text = request.json["input_text"]
    # else:  # GET request
    #     input_text = request.args.get("input_text")
    # Tokenize and encode the prompt
    encoded_input = tokenizers.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(encoded_input).to(device)

    # Generate text based on the input text
    output = model.generate(encoded_input, max_length=100, num_return_sequences=1, attention_mask=attention_mask)

    # Decode and convert the generated output to text
    generated_text = tokenizers.decode(output[0], skip_special_tokens=True)

    # Print the generated text in the server logs
    # app.logger.info("Generated Text: %s", generated_text)

    print("Generated Text:", generated_text)
    # Return the generated text as the API response
    return jsonify({"output_text": generated_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6500)

