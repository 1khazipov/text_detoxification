import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import spacy
import warnings
from transformers import AutoTokenizer
warnings.warn("ignore")


def create_dataframe(zip_file_path):
    # check if zip file exists
    if not os.path.exists(zip_file_path):
        print(f"zip file '{zip_file_path}' does not exist.")
        return None

    # extract zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all files in the ZIP archive to a directory
        extraction_path = "./tmp_extraction"
        zip_ref.extractall(extraction_path)

    # find .tsv file
    tsv_file = None
    for root, dirs, files in os.walk(extraction_path):
        for file in files:
            if file.endswith(".tsv"):
                tsv_file = os.path.join(root, file)
                break

    if tsv_file is None:
        print("No .tsv file found in the extracted ZIP archive.")
        return None

    # create pandas DataFrame
    try:
        df = pd.read_csv(tsv_file, delimiter='\t')
        return df
    except Exception as e:
        print(f"Error while creating DataFrame: {str(e)}")
        return None
    finally:
        # remove the temporary extraction directory
        if os.path.exists(extraction_path):
            for root, dirs, files in os.walk(extraction_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
            os.rmdir(extraction_path)


def create_new_columns(dataset):
    # create new columns
    # dataset['toxic'] = [''] * dataset.shape[0]
    # dataset['nontoxic'] = [''] * dataset.shape[0]
    # dataset['toxic_tox'] = [''] * dataset.shape[0]
    # dataset['nontoxic_tox'] = [''] * dataset.shape[0]

    # create the 'toxic' and 'nontoxic' columns
    dataset['toxic'] = dataset.apply(lambda row: row['reference'] if row['ref_tox'] > row['trn_tox'] else row['translation'], axis=1)
    dataset['nontoxic'] = dataset.apply(lambda row: row['translation'] if row['ref_tox'] > row['trn_tox'] else row['reference'], axis=1)

    # create the 'toxic_tox' and 'nontoxic_tox' columns
    dataset['toxic_tox'] = dataset.apply(lambda row: row['ref_tox'] if row['ref_tox'] > row['trn_tox'] else row['trn_tox'], axis=1)
    dataset['nontoxic_tox'] = dataset.apply(lambda row: row['trn_tox'] if row['ref_tox'] > row['trn_tox'] else row['ref_tox'], axis=1)

    # drop unuseful columns
    dataset = dataset.drop(columns=['reference', 'translation', 'similarity', 'lenght_diff', 'ref_tox', 'trn_tox'])

    return dataset
    # return dataset['toxic'].tolist(), dataset['nontoxic'].tolist()


def remove_unuseful_data(dataset):
    # toxic sentences > 0.75 of toxic_tox: 0.9243254750535241
    dataset = dataset[dataset['toxic_tox'] > 0.75]

    # detoxed sentences < 0.25 of nontoxic_tox: 0.959344176040237
    dataset = dataset[dataset['nontoxic_tox'] < 0.25]

    return dataset


def get_sentences(dataset):
    # toxic and detoxed sentences
    return dataset['toxic'].tolist(), dataset['nontoxic'].tolist()


def split_train_test(toxic, nontoxic, path):
    toxic_train, toxic_test, nontoxic_train, nontoxic_test = train_test_split(
        toxic,
        nontoxic,
        test_size=0.25,
        random_state=42,
    )

    with open(os.path.join(path, 'toxic_train'), "w", encoding="UTF-8") as file:
        file.write("\n".join(toxic_train))
    with open(os.path.join(path, 'toxic_test'), "w", encoding="UTF-8") as file:
        file.write("\n".join(toxic_test))
    with open(os.path.join(path, 'nontoxic_train'), "w", encoding="UTF-8") as file:
        file.write("\n".join(nontoxic_train))
    with open(os.path.join(path, 'nontoxic_test'), "w", encoding="UTF-8") as file:
        file.write("\n".join(nontoxic_test))

def save_csv(dataset):
    dataset.to_csv("../data/interim/converted.csv")

def tokenize(dataset, model_checkpoint='s-nlp/bart-base-detox'):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    max_input_length = 80
    max_target_length = 80
    prefix = ""

    inputs = [prefix + ex for ex in dataset["toxic"]]
    targets = [ex for ex in dataset["nontoxic"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
