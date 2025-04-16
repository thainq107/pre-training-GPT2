from transformers import GPT2Config, GPT2LMHeadModel

def load_model(toknizer, n_positions=512, n_ctx=512, n_embd=512, n_layer=6, n_head=8):
  config = GPT2Config(
      vocab_size=tokenizer.vocab_size,
      n_positions=n_positions,
      n_ctx=n_ctx,
      n_embd=n_embd,
      n_layer=n_layer,
      n_head=n_head,
      bos_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id
  )
  
  model = GPT2LMHeadModel(config)
  return model
