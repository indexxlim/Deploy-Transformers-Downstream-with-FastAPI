

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.config = AutoConfig.from_pretrained('model/electra/CoLA')
        self.tokenizer = AutoTokenizer.from_pretrained('model/electra/CoLA')
        self.model = AutoModelForSequenceClassification.from_pretrained('model/electra/CoLA',config=self.config)

        self.classifier = self.model.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask)[0], dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return probabilities


model = Model()


def get_model():
    return model
