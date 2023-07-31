import collections
import numpy as np
import pandas as pd

def get_schema_string(table_json):
    """Returns the schema serialized as a string."""
    table_id_to_column_names = collections.defaultdict(list)
    for table_id, name in table_json["column_names_original"]:
        table_id_to_column_names[table_id].append(name.lower())
        tables = table_json["table_names_original"]

    table_strings = []
    for table_id, table_name in enumerate(tables):
        column_names = table_id_to_column_names[table_id]
        table_string = " | %s : %s" % (table_name.lower(), " , ".join(column_names))
        table_strings.append(table_string)
    return "".join(table_strings)


def schema_linking(train_spider, others_spider, dev_spider, schema_dict):
    prefix = 'translate English to SQL:'

    train_spider['schema'] = train_spider['db_id'].map(schema_dict)
    train_spider['prompt'] = prefix + train_spider['question'] + '\nDatabse schema is ' + train_spider['schema']
    others_spider['schema'] = others_spider['db_id'].map(schema_dict)
    others_spider['prompt'] = prefix + others_spider['question'] + '\nDatabse schema is ' + others_spider['schema']
    dev_spider['schema'] = dev_spider['db_id'].map(schema_dict)
    dev_spider['prompt'] = prefix + dev_spider['question'] + '\nDatabse schema is ' + dev_spider['schema']
    
    return train_spider, others_spider, dev_spider


def load_spider_datasets():
    # datasets
    with open('../spider/train_spider.json', 'r') as f:
        train_spider = pd.read_json(f)
    with open('../spider/train_others.json', 'r') as f:
        others_spider = pd.read_json(f)
    with open('../spider/dev.json', 'r') as f:
        dev_spider = pd.read_json(f)
        
    # load schema for all tables
    with open('spider/tables.json', 'r') as f:
        schema_df = pd.read_json(f)
    
    schema_dict = {}
    for _, row in schema_df.iterrows():
        db_id = row['db_id']
        schema = get_schema_string(row)
        schema_dict[db_id] = schema
        
    train_spider, others_spider, dev_spider = schema_linking(train_spider, others_spider, dev_spider, schema_dict)
    return train_spider, others_spider, dev_spider


def preprocess_data(text_pair, tokenizer, input_max_length=512, output_max_length=128):
    orig_text, target_text = text_pair
    orig_encoded = tokenizer.batch_encode_plus(
        [orig_text],
        max_length=input_max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    orig_input_ids = orig_encoded['input_ids'][0]
    orig_attention_mask = orig_encoded['attention_mask'][0]

    target_encoded = tokenizer.batch_encode_plus(
        [target_text],
        max_length=output_max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    label_ids = target_encoded['input_ids'][0]

    return {'input_ids': orig_input_ids,
            'attention_mask': orig_attention_mask,
            'labels': label_ids}


class DatasetIterator:

    def __init__(self,
                 df,
                 tokenizer,
                 max_load_at_once=100,
                 input_max_length=512,
                 output_max_length=128,
                 shuffle=True):

        self.df = df
        self.tokenizer = tokenizer
        self.n_examples = len(df)
        self.max_load_at_once = max_load_at_once
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.shuffle = shuffle

        # Initialize row order, call on_epoch_end to shuffle row indices
        self.row_order = np.arange(1, self.n_examples+1)
        self.on_epoch_end()

        # Load first chunk of max_load_at_once examples
        self.df_curr_loaded = self._load_next_chunk(0)
        self.curr_idx_in_load = 0

    def _load_next_chunk(self, idx):
        load_start = idx
        load_end = idx + self.max_load_at_once

        # Indices to skip are the ones in the shuffled row_order before and
        # after the chunk we'll use for this chunk
        self.df_curr_loaded = self.df.iloc[load_start:load_end].sample(frac=1)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        if self.df_curr_loaded is None or self.curr_idx_in_load >= len(self.df_curr_loaded):
            self._load_next_chunk(idx)
            self.curr_idx_in_load = 0

        text_pair = self.df_curr_loaded[['prompt', 'query']].values.astype(str)[self.curr_idx_in_load]
        self.curr_idx_in_load += 1

        item_data = preprocess_data(
            text_pair,
            self.tokenizer,
            self.input_max_length,
            self.output_max_length
        )

        return item_data

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__()-1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.row_order = list(np.random.permutation(self.row_order))