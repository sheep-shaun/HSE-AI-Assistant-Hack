import torch
from transformers import BertModel, BertTokenizer
from catboost import CatBoostClassifier


class Ranker:
    def __init__(self, model_path: str = "ranker", device: str = "cpu"):
        self.device = device
        
        self.model_name = "DeepPavlov/rubert-base-cased-sentence"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device).eval()

        self.ranker = CatBoostClassifier(
            random_state=42,
            verbose=10,
        )
        self.ranker.load_model(model_path)


    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=256)
        
        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
    
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embedding

    def predict_proba(self, sentence):
        embedding = self.get_sentence_embedding(sentence)
        return self.ranker.predict_proba(embedding)[:, 1][0]