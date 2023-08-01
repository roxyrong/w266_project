import random
from collections import Counter
import torch
import nltk
from nltk.corpus import stopwords
from string import punctuation
from peft import get_peft_config, PromptTuningConfig, PromptTuningInit, TaskType

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation = list(punctuation)

def init_most_common_vocab(df, tokenizer, num_tokens=100):
    corpus = list(df['question'])
    token_counter = Counter()
    for text in corpus:
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
        token_counter.update(cleaned_tokens)

    most_common_tokens = token_counter.most_common(num_tokens)
    most_common_tokens = " ".join([token for token, freq in most_common_tokens])[:num_tokens]
    return most_common_tokens

def init_most_common_label(df, tokenizer, num_tokens=100):
    corpus = list(df['query'])
    token_counter = Counter()
    for text in corpus:
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
        token_counter.update(cleaned_tokens)

    most_common_tokens = token_counter.most_common(num_tokens, num_tokens=100)
    most_common_tokens = " ".join([token for token, freq in most_common_tokens])[:num_tokens]
    return most_common_tokens
  
def init_random_vocab_from_tokenizer(tokenizer, num_tokens=100):
    vocabulary = list(tokenizer.get_vocab().keys())
    random_tokens = random.sample(vocabulary, num_tokens)
    random_vocab = " ".join(random_tokens)
    return random_vocab

def init_random_vocab_from_dataset(df, tokenizer, num_tokens=100):
    corpus = " ".join(list(df['question']))
    tokens = tokenizer.tokenize(corpus)
    cleaned_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    random_tokens = random.sample(cleaned_tokens, num_tokens)
    random_vocab = " ".join(random_tokens)
    return random_vocab

def init_label_embedding(df, tokenizer):
    label_embedding = []
    for query in df['query']:
        inputs = tokenizer.encode(
            query,
            max_length=100,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt'
        )
        embedding = inputs.float()
        label_embedding.append(embedding)
    stacked_embeddings = torch.stack(label_embedding)
    average_embeddings = torch.mean(stacked_embeddings, dim=0)
    return average_embeddings

def set_peft_config_with_random_init(num_tokens=100):
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=num_tokens,
        inference_mode=False,
        tokenizer_name_or_path="t5-base",
        token_dim=768,
        num_transformer_submodules=1,
        num_attention_heads=12,
    )
    return peft_config


def set_peft_config_with_init_text(init_text, num_tokens=100):
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_tokens,
        inference_mode=False,
        prompt_tuning_init_text=init_text,
        token_dim=768,
        num_transformer_submodules=1,
        num_attention_heads=12,
        tokenizer_name_or_path="t5-base",
    )
    return peft_config


def set_peft_config_with_embedding(init_embedding, num_tokens=100):
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=init_embedding,
        num_virtual_tokens=num_tokens,
        inference_mode=False,
        token_dim=768,
        num_transformer_submodules=1,
        num_attention_heads=12,
        tokenizer_name_or_path="t5-base",
    )    
    return peft_config