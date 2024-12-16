import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = "./model"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained("./tokenizer")

def summarize_text_(text, max_input_length=512, max_output_length=128):
    """
    Summarizes input text using a T5 model.
    """
    input_text = "summarise:" + text.strip().lower()
    
    # Tokenize
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=max_input_length)
    max_output_length = min(len(input_text)/3, 128)
    if len(input_text) < 20:
        min_length = 2
    elif len(input_text) < 50:
        min_length = 15
    else:
        min_length = 20
    
    # Generate the summary
    outputs = model.generate(inputs, max_length=max_output_length, min_length=min_length, length_penalty=2.0, num_beams=4)
    
    # Decode the summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

interface = gr.Interface(
    fn=summarize_text_, 
    inputs=gr.Textbox(lines=10, placeholder="Enter Text here..", label="Input Text"), 
    outputs= gr.Textbox(label="Summary"),
    title="Text-Summarization",
    )

interface.launch()



















# from huggingface_hub import InferenceClient

# """
# For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
# """
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


# def respond(
#     message,
#     history: list[tuple[str, str]],
#     system_message,
#     max_tokens,
#     temperature,
#     top_p,
# ):
#     messages = [{"role": "system", "content": system_message}]

#     for val in history:
#         if val[0]:
#             messages.append({"role": "user", "content": val[0]})
#         if val[1]:
#             messages.append({"role": "assistant", "content": val[1]})

#     messages.append({"role": "user", "content": message})

#     response = ""

#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content

#         response += token
#         yield response


# """
# For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
# """
# demo = gr.ChatInterface(
#     respond,
#     additional_inputs=[
#         gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
#         gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
#         gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
#         gr.Slider(
#             minimum=0.1,
#             maximum=1.0,
#             value=0.95,
#             step=0.05,
#             label="Top-p (nucleus sampling)",
#         ),
#     ],
# )


# if __name__ == "__main__":
#     demo.launch()
