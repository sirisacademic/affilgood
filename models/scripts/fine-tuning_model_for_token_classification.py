import pandas as pd
import datasets
import re
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, AutoModelForPreTraining, DistilBertConfig, DistilBertForTokenClassification,RobertaTokenizerFast, RobertaTokenizer
from datasets import DatasetDict
import evaluate
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Fine-tuning model for NER task')
parser.add_argument('-d','--dataset_path', help='dataset path', required=True)
parser.add_argument('-m','--model_path', help='model path', required=True)
parser.add_argument('-o','--output_dir', help='output dir', required=True)

args = vars(parser.parse_args())

dataset_path = args['dataset_path']
model_path = args['model_path']
output_dir = args['output_dir']


seqeval = evaluate.load("seqeval")

'''Functions for formating annotations'''

def formatted_annotation(text, labels):
  formatted_entities = []

  # Initialize the start index for the next labeled entity
  start_index = 0

  for start, end, entity_type in labels:
      # Append the non-labeled part of the text
      formatted_entities.append(text[start_index:start])

      # Append the labeled part in the specified format
      entity_text = text[start:end]
      formatted_entity = f'{{{entity_text}}}[{entity_type}]'
      formatted_entities.append(formatted_entity)

      # Update the start index for the next iteration
      start_index = end

  # Append any remaining non-labeled text
  formatted_entities.append(text[start_index:])

  formatted_text = ''.join(formatted_entities)
  return formatted_text

def preprocess_punctuation(raw_text: str):
  punctuations = '\'!"#$%&\'()*+,-./:;<=>?@[\\]^_`|~'
  labels = re.findall(r'(?<=\[)[a-zA-Z0-9]+(?=\])', raw_text)
  for punct in list(punctuations):
    raw_text = raw_text.replace(punct, f' {punct} ')
  raw_text = re.sub(r'\s+', ' ', raw_text)
  for label in labels:
    raw_text = raw_text.replace(f'[ {label} ]', f'[{label}]')
  raw_text = re.sub(r'\s*\}\s*', '}', raw_text).strip()
  # Added PA - Add missing spaces.
  #raw_text = re.sub(r'([^\s])\{', r'\1 {', raw_text)
  return raw_text

def get_tokens_with_entities(raw_text: str):
    raw_text = preprocess_punctuation(raw_text)
    raw_tokens = re.split(r"\s(?![^\{]*\})", raw_text)
    entity_value_pattern = r"\{(?P<value>.+?)\}\[(?P<entity>.+?)\]"
    entity_value_pattern_compiled = re.compile(entity_value_pattern, flags=re.I|re.M)
    tokens_with_entities = []
    for raw_token in raw_tokens:
        match = entity_value_pattern_compiled.match(raw_token)
        if match:
            raw_entity_name, raw_entity_value = match.group("entity"), match.group("value")
            for i, raw_entity_token in enumerate(re.split("\s", raw_entity_value)):
                entity_prefix = "B" if i == 0 else "I"
                entity_name = f"{entity_prefix}-{raw_entity_name}"
                tokens_with_entities.append((raw_entity_token, entity_name))
        else:
            tokens_with_entities.append((raw_token, "O"))
    return tokens_with_entities

class NERDataMaker:
    def __init__(self, texts):
        self.unique_entities = []
        self.processed_texts = []
        temp_processed_texts = []
        for text in texts:
            tokens_with_entities = get_tokens_with_entities(text)
            for _, ent in tokens_with_entities:
                if ent not in self.unique_entities:
                    self.unique_entities.append(ent)
            temp_processed_texts.append(tokens_with_entities)
        self.unique_entities.sort(key=lambda ent: ent if ent != "O" else "")
        for tokens_with_entities in temp_processed_texts:
            self.processed_texts.append([(t, self.unique_entities.index(ent)) for t, ent in tokens_with_entities])

    @property
    def id2label(self):
        return dict(enumerate(self.unique_entities))

    @property
    def label2id(self):
        return {v:k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        def _process_tokens_for_one_text(id, tokens_with_encoded_entities):
            ner_tags = []
            tokens = []
            for t, ent in tokens_with_encoded_entities:
                ner_tags.append(ent)
                tokens.append(t)
            return {
                "id": id,
                "ner_tags": ner_tags,
                "tokens": tokens
            }
        tokens_with_encoded_entities = self.processed_texts[idx]
        if isinstance(idx, int):
            return _process_tokens_for_one_text(idx, tokens_with_encoded_entities)
        else:
            return [_process_tokens_for_one_text(i+idx.start, tee) for i, tee in enumerate(tokens_with_encoded_entities)]

    def as_hf_dataset(self, tokenizer):
        from datasets import Dataset, Features, Value, ClassLabel, Sequence
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        ids, ner_tags, tokens, strings = [], [], [], []

        for i, pt in enumerate(self.processed_texts):
            ids.append(i)
            pt_tokens,pt_tags = list(zip(*pt))
            ner_tags.append(pt_tags)
            tokens.append(pt_tokens)
            strings.append(' '.join(pt_tokens))

        data = {
            "id": ids,
            "ner_tags": ner_tags,
            "tokens": tokens,
            "strings": strings
        }
        features = Features({
            "strings": Value("string"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=self.unique_entities)),
            "id": Value("int32")
        })
        ds = Dataset.from_dict(data, features)
        tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
        return tokenized_ds

def preprocess_punctuation(raw_text: str):
  punctuations = '\'!"#$%&\'()*+,-./:;<=>?@[\\]^_`|~'
  labels = re.findall(r'(?<=\[)[a-zA-Z0-9]+(?=\])', raw_text)
  for punct in list(punctuations):
    raw_text = raw_text.replace(punct, f' {punct} ')
  raw_text = re.sub(r'\s+', ' ', raw_text)
  for label in labels:
    raw_text = raw_text.replace(f'[ {label} ]', f'[{label}]')
  raw_text = re.sub(r'\s*\}\s*', '}', raw_text).strip()
  # Added PA - Add missing spaces.
  #raw_text = re.sub(r'([^\s])\{', r'\1 {', raw_text)
  return raw_text

def get_tokens_with_entities(raw_text: str):
    raw_text = preprocess_punctuation(raw_text)
    raw_tokens = re.split(r"\s(?![^\{]*\})", raw_text)
    entity_value_pattern = r"\{(?P<value>.+?)\}\[(?P<entity>.+?)\]"
    entity_value_pattern_compiled = re.compile(entity_value_pattern, flags=re.I|re.M)
    tokens_with_entities = []
    for raw_token in raw_tokens:
        match = entity_value_pattern_compiled.match(raw_token)
        if match:
            raw_entity_name, raw_entity_value = match.group("entity"), match.group("value")
            for i, raw_entity_token in enumerate(re.split("\s", raw_entity_value)):
                entity_prefix = "B" if i == 0 else "I"
                entity_name = f"{entity_prefix}-{raw_entity_name}"
                tokens_with_entities.append((raw_entity_token, entity_name))
        else:
            tokens_with_entities.append((raw_token, "O"))
    return tokens_with_entities

class NERDataMaker:
    def __init__(self, texts):
        self.unique_entities = []
        self.processed_texts = []
        temp_processed_texts = []
        for text in texts:
            tokens_with_entities = get_tokens_with_entities(text)
            for _, ent in tokens_with_entities:
                if ent not in self.unique_entities:
                    self.unique_entities.append(ent)
            temp_processed_texts.append(tokens_with_entities)
        self.unique_entities.sort(key=lambda ent: ent if ent != "O" else "")
        for tokens_with_entities in temp_processed_texts:
            self.processed_texts.append([(t, self.unique_entities.index(ent)) for t, ent in tokens_with_entities])

    @property
    def id2label(self):
        return dict(enumerate(self.unique_entities))

    @property
    def label2id(self):
        return {v:k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        def _process_tokens_for_one_text(id, tokens_with_encoded_entities):
            ner_tags = []
            tokens = []
            for t, ent in tokens_with_encoded_entities:
                ner_tags.append(ent)
                tokens.append(t)
            return {
                "id": id,
                "ner_tags": ner_tags,
                "tokens": tokens
            }
        tokens_with_encoded_entities = self.processed_texts[idx]
        if isinstance(idx, int):
            return _process_tokens_for_one_text(idx, tokens_with_encoded_entities)
        else:
            return [_process_tokens_for_one_text(i+idx.start, tee) for i, tee in enumerate(tokens_with_encoded_entities)]

    def as_hf_dataset(self, tokenizer):
        from datasets import Dataset, Features, Value, ClassLabel, Sequence
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        ids, ner_tags, tokens, strings = [], [], [], []

        for i, pt in enumerate(self.processed_texts):
            ids.append(i)
            pt_tokens,pt_tags = list(zip(*pt))
            ner_tags.append(pt_tags)
            tokens.append(pt_tokens)
            strings.append(' '.join(pt_tokens))

        data = {
            "id": ids,
            "ner_tags": ner_tags,
            "tokens": tokens,
            "strings": strings
        }
        features = Features({
            "strings": Value("string"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=self.unique_entities)),
            "id": Value("int32")
        })
        ds = Dataset.from_dict(data, features)
        tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
        return tokenized_ds

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

data_raw = pd.read_json(dataset_path, lines=True)

data_raw.text = data_raw.text.astype('str')
data_raw = data_raw.sample(frac=1,random_state=42)

# Format annotations 
data_raw['text_annotated'] = data_raw.apply(lambda row: formatted_annotation(row['text'],row['label']),axis=1)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True, trim_offsets=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

examples = data_raw['text_annotated'].tolist()
dm = NERDataMaker(examples)

# Training dataset
example_ds = dm.as_hf_dataset(tokenizer=tokenizer)

# 80% train, 10% test + 10% validation
train_testvalid = example_ds.train_test_split(test_size=0.8, seed=42)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})

label_list_train = train_test_valid_dataset["train"].features[f"ner_tags"].feature.names
label_list_val = train_test_valid_dataset["val"].features[f"ner_tags"].feature.names
label_list_test = train_test_valid_dataset["test"].features[f"ner_tags"].feature.names
label_list = label_list_train

model = AutoModelForTokenClassification.from_pretrained(model_path, ignore_mismatched_sizes=True, \
                                                        num_labels=len(dm.unique_entities), id2label=dm.id2label, label2id=dm.label2id)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_valid_dataset['train'],
    eval_dataset=train_test_valid_dataset['val'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
