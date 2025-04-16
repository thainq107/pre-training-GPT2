from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

def build_tokenizer(ds, vocab_size=50257):
  # Initialize BPE tokenizer
  tokenizer = Tokenizer(BPE())
  tokenizer.pre_tokenizer = ByteLevel()
  tokenizer.normalizer = NFKC()
  tokenizer.decoder = ByteLevelDecoder()
  
  trainer = BpeTrainer(
      vocab_size=vocab_size,
      special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
  )
  tokenizer_file = "gpt_tokenizer.json"
  tokenizer.train_from_iterator(ds["train"]["text"], trainer)
  tokenizer.save(tokenizer_file)
  return tokenizer_file

def load_tokenizer(tokenizer_file="gpt_tokenizer.json"):
  tokenizer = PreTrainedTokenizerFast(tokenizer_file)
  tokenizer.add_special_tokens({
      "bos_token": "<s>",
      "eos_token": "</s>",
      "unk_token": "<unk>",
      "pad_token": "<pad>",
      "mask_token": "<mask>",
  })
  return tokenizer
