from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name = "thainq107/gpt-small-c4"):
  model = AutoModelForCausalLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  return tokenizer, model

tokenizer, model = load_model()

if __name__ == "__main__":
    prompt = "I go to"
    inputs = tokenizer(
        prompt, return_tensors="pt"
    ).to(model.device)
    
    
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    print(output)
  
