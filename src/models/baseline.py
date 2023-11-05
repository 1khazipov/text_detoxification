import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class ToxicityClassifier:
    def __init__(self, model_name='s-nlp/roberta_toxicity_classifier_v1'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = RobertaForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def text_toxicity(self, texts):
        """
        baseline model
        https://huggingface.co/s-nlp/roberta_toxicity_classifier_v1
        """
        with torch.no_grad():
            input_ids = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)
            logits = self.model(**input_ids).logits
            probabilities = torch.softmax(logits, -1)
            toxicity_scores = probabilities[:, 1].cpu().numpy()
        return toxicity_scores

    def delete_toxic(self, toxic_sentences, threshold=0.5):
        """
        remove toxic words from a list of sentences
        """
        nontoxic_text = []
        for toxic_sentence in toxic_sentences:
            words = toxic_sentence.split()
            toxic_scores = self.text_toxicity(words)
            nontoxic_words = []
            for word, score in zip(words, toxic_scores):
                if score < threshold:
                    nontoxic_words.append(word)
            nontoxic_text.append(" ".join(nontoxic_words))
        return nontoxic_text
