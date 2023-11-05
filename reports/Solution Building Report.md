# Solution Building Report

## Baseline
For baseline, I decided to get a classifier that can be used to identify toxic words to then remove them. Using the s-nlp/roberta_toxicity_toxicity_classifier_v1 model, I determine the toxicity probabilities of words to remove words with a high probability of toxicity from the dataset. This model works well, however it does not replace words in sentences, due to which the sentence may lose meaning.  

## Hypothesis 1: Text-To-Text Transfer Transformer (T5-base)
A combination of supervised and unsupervised tasks were used to pre-train the encoder-decoder model T5, which was then trained on each task using a text-to-text format. T5 applies an appropriate prefix to the input according to the job, which allows it to perform effectively on a wide range of tasks straight out of the box.
When using corrupted tokens for self-supervised training, 15% of the tokens are randomly removed and replaced with unique sentinel tokens (if many consecutive tokens are designated for removal, the entire group is replaced with a single sentinel token). The original sentence is the input of the decoder, the corrupted sentence is the input of the encoder, and the dropped out tokens are the target, separated by their sentinel tokens. Relative scalar embeddings are used in T5. Padding encoder input can be done both to the left and to the right.

## Hypothesis 2: Fine-tuned T5
To improve the model results, I used a fine-tuned model for text detoxification tasks. The s-nlp/t5-paranmt-detox is suitable for my task, so I want to fine-tune this model at 15 epochs and then save it in the `models` folder. 

## Hypothesis 3: Pre-trained Bart
Initially text is corrupted using an arbitrary noising function, and then BART is pre-trained by building a model to reconstruct the original text. It makes use of a conventional Transformer-based neural machine translation architecture, which, in spite of its simplicity, may be understood as a generalization of many other more current pre-training schemes as well as BERT, GPT, and left-to-right decoder-based pre-training schemes. 
I will use pre-trained s-nlp/bart-base-detox, fine-tune it at 5 epochs and then also save my model to the `models` folder.

## Results
input: It was perfectly planned, but stupidly done.
baseline: It was perfectly planned, but done.
bert: it was perfectly planned, but expertly done..
bart: It was perfectly planned and horribly done.
T5: It was perfectly planned, but it was a waste of time.

*I used only part of the dataset (about 10-15%) because I didn't have a powerful gpu to handle the whole dataset. I used google colab to train and test my models, but still did not get high speed in model training. 
