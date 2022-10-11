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

    def process_long_sequences(self, tokenized: BatchEncoding) -> BatchEncoding:
        """
        The classifier can only process inputs with a maximum of 128 tokens. The complaints might be longer than that.
        Therefore, if a tokenized complaint is longer than 128 tokens, I split it up in subsequences of 128 tokens.
        Since the last subsequence is usually smaller than 128 tokens, we have to pad input_ids, token_type_ids,
        attention_mask with appropriate values.
        """
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


if __name__ == '__main__':
    classifier = Classifier()
    long_test_str = "I closed my {$500.00} limit credit card account # XXXX with XXXX XXXX XXXX in XXXX 2016. This was a secured card ( I paid the {$500.00} upfront ). I contacted XXXX XXXX in XXXX 2016 to indicate I wanted to close the card due to the fact I no longer needed this card. I was asked multiple times to please not close the card, however to clI was insistent they close my account and I paid the {$330.00} balance at the time. I asked if I needed to contact them back regarding any other issues to close my card, they stated it could take 30 days for any accuring interest to show. I called back in XXXX XXXX and asked if any unpaid interest was I was told may card was closed and I had a {$0.00} balance, and if any accrued interest occurred I would be sent a bill. I received no bill. I did receive mulitple solicitations to reinstate my card, which I declined. I recently pulled my credit report and after XXXX XXXX XXXX XXXX charged me a {$38.00} fee- never disclosed to me and never billed. I contacted them to indicate this was inaccurate and they stated an annual charge must have accidently been applied after my card was closed.I belive this is a fraudulent practice in an attempt to knowingly charge a fee once I closed my card, and worse yet they reported it 90 days late. I am so dissapointed at such a shameless attempt at defrauding hard working people. And STILL it is on my credit and has dropped my score nearly 80 points! For a secured card that I had a {$500.00} deposit on! They stated they would send my my secured deposit within 30 days, it took nearly 8 months to receive my deposit- once again a shameless tactic."
    predictions = classifier.predict_sentiment([long_test_str, "some other string"])




