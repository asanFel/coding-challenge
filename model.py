from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
from transformers.tokenization_utils_base import BatchEncoding


# https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
class Classifier:
    def __init__(self, model_name: str = "finiteautomata/bertweet-base-sentiment-analysis", max_sequence_length: int = 128):
        self.max_sequence_length = max_sequence_length
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        print("Load tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def _load_model(self):
        print("Load model...")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return model

    def process_long_sequences(self, tokenized):
        input_ids = list(torch.split(tokenized["input_ids"][0], self.max_sequence_length))
        token_type_ids = list(torch.split(tokenized["token_type_ids"][0], self.max_sequence_length))
        attention_mask = list(torch.split(tokenized["attention_mask"][0], self.max_sequence_length))

        len_to_pad = self.max_sequence_length - len(input_ids[-1])
        padding_input_ids_token_type_ids = torch.IntTensor([self.tokenizer.pad_token_id] * len_to_pad)
        padding_attention_mask = torch.IntTensor([0] * len_to_pad)

        input_ids_padded = torch.cat((input_ids[-1], padding_input_ids_token_type_ids), 0)
        token_type_ids_padded = torch.cat((token_type_ids[-1], padding_attention_mask), 0)
        attention_mask_padded = torch.cat((attention_mask[-1], padding_attention_mask), 0)

        input_ids[-1] = input_ids_padded
        token_type_ids[-1] = token_type_ids_padded
        attention_mask[-1] = attention_mask_padded

        input_ids = torch.stack(input_ids, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)

        tokenized = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }

        return BatchEncoding(tokenized)

    def predict_sentiment(self, text_sequence: List[str]) -> List[int]:
        """
        :param text_sequence: list of texts which we want to predict the sentiment on
        :return: list of integers which represent the sentiment class (1: positiv, 2: medium, 3: negative)
        """
        predictions = []

        for text in text_sequence:
            tokenized = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")

            if len(tokenized["input_ids"][0]) > self.max_sequence_length:
                tokenized = self.process_long_sequences(tokenized)

            with torch.no_grad():
                logits = torch.mean(self.model(**tokenized).logits, dim=0)
            prediction = int(logits.argmax(-1).item())
            predictions.append(prediction)
        return predictions
