from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

src_text =["""When you declare a default value for non-path parameters (for now, we have only seen query parameters), then it is not required.
If you don't want to add a specific value but just make it optional, set the default as None.
But when you want to make a query parameter required, you can just not declare any default value:"""]
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="t5_summarization",  # Where to save the model
    evaluation_strategy="steps",    # Evaluate after every `eval_steps`
    eval_steps=500,                 # Number of steps between evaluations
    logging_dir="./logs",           # Where to save logs
    logging_steps=500,              # Log training stats every 500 steps
    save_steps=1000,                # Save model checkpoints every 1000 steps
    save_total_limit=2,             # Keep only the last two saved models
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,   # Batch size for evaluation
    num_train_epochs=3,             # Number of epochs
    learning_rate=5e-5,             # Learning rate
    weight_decay=0.01,              # Weight decay for regularization
    warmup_steps=500,               # Number of warmup steps for learning rate scheduler
    fp16=torch.cuda.is_available(),  # Use FP16 for faster training if GPU is available
    report_to="none"                # Disable reporting to external tools (e.g., wandb)
)


# input_text = "summarize: " + src_text[0]
# inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
# summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print(summary)