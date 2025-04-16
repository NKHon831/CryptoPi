from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

def classify_sentiment(texts):
  # https://huggingface.co/ElKulako/cryptobert
  model_name = "ElKulako/cryptobert"
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
  pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding='max_length')

  preds = pipe(texts)
  # print(preds)

  return preds

if __name__ == "__main__":
  # Example usage
  texts = [
    "The price of Bitcoin is going to the moon!",
    "I think Ethereum is a bad investment.",
    "Litecoin is the future of finance.",
    "I don't like Dogecoin."
  ]

  preds = classify_sentiment(texts)
  print(preds)

  for text, pred in zip(texts, preds):
    print(f"Text: {text}")
    print(f"Predicted sentiment: {pred['label']} with score: {pred['score']}")
    print()