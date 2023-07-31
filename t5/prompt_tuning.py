import random
from collections import Counter
import nltk
from nltk.corpus import stopwords
from string import punctuation
from peft import get_peft_config, PromptTuningConfig, PromptTuningInit, TaskType

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation = list(punctuation)

def init_most_common_vocab(df, tokenizer):
    corpus = list(df['question'])
    token_counter = Counter()
    for text in corpus:
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
        token_counter.update(cleaned_tokens)

    most_common_tokens = token_counter.most_common(500)
    most_common_tokens = " ".join([token for token, freq in most_common_tokens])[:512]
    return most_common_tokens

def init_most_common_label(df, tokenizer):
    corpus = list(df['query'])
    token_counter = Counter()
    for text in corpus:
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
        token_counter.update(cleaned_tokens)

    most_common_tokens = token_counter.most_common(500)
    most_common_tokens = " ".join([token for token, freq in most_common_tokens])[:512]
    return most_common_tokens
  
def init_random_vocab_from_tokenizer(tokenizer):
    vocabulary = list(tokenizer.get_vocab().keys())
    random_tokens = random.sample(vocabulary, 512)
    random_vocab = " ".join(random_tokens)
    return random_vocab

def init_random_vocab_from_dataset(df, tokenizer):
    corpus = " ".join(list(df['question']))
    tokens = tokenizer.tokenize(corpus)
    cleaned_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    random_tokens = random.sample(cleaned_tokens, 512)
    random_vocab = " ".join(random_tokens)
    return random_vocab

def set_peft_config_with_random_init(num_tokens=100):
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=num_tokens,
        inference_mode=False,
        tokenizer_name_or_path="t5-base",
    )
    return peft_config


def set_peft_config_with_init_text(init_text, num_tokens=100):
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_tokens,
        inference_mode=False,
        prompt_tuning_init_text=init_text,
        tokenizer_name_or_path="t5-base",
    )
    return peft_config


def set_peft_config_with_embedding(init_embedding, num_tokens=100):
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=init_embedding,
        num_virtual_tokens=num_tokens,
        inference_mode=False,
        tokenizer_name_or_path="t5-base",
    )    
    return peft_config