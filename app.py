from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi import FastAPI, Request


tokenizer = T5Tokenizer.from_pretrained("./results")
model = T5ForConditionalGeneration.from_pretrained("./results")

app = FastAPI()

@app.get("/")
def read_root():
    return {"res": "Welcome to summarisation with t5"} 

@app.get("/summarize/") 
async def summarize(request: Request):
    payload = await request.json()
    text = payload.get("text", "").strip()
    if text == "":
        return {"error": "Please provide text to summarize."}
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
    print(summary_ids)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)