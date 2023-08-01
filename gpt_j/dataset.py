import collections
import numpy as np
import pandas as pd


PROMPT_1_SHOT = """
Context: | ref_document_types : document_type_code , document_type_description | roles : role_code , role_description | addresses : address_id , address_details | ref_document_status : document_status_code , document_status_description | ref_shipping_agents : shipping_agent_code , shipping_agent_name , shipping_agent_description | documents : document_id , document_status_code , document_type_code , shipping_agent_code , receipt_date , receipt_number , other_details | employees : employee_id , role_code , employee_name , other_details | document_drafts : document_id , draft_number , draft_details | draft_copies : document_id , draft_number , copy_number | circulation_history : document_id , draft_number , copy_number , employee_id | documents_mailed : document_id , mailed_to_address_id , mailing_date
Question: Which employee has showed up in most circulation history documents. List the employee's name and the number of drafts and copies.\n"
Answer: SELECT Employees.employee_name , count(*) FROM Employees JOIN Circulation_History ON Circulation_History.employee_id = Employees.employee_id GROUP BY Circulation_History.document_id , Circulation_History.draft_number , Circulation_History.copy_number ORDER BY count(*) DESC LIMIT 1;
###
"""


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
    fixed_few_shot_prefix = """
               Context:| member : member_id , card_number , name , hometown , level | branch : branch_id , name , open_year , address_road , city , membership_amount | membership_register_branch : member_id , branch_id , register_year | purchase : member_id , branch_id , year , total_pounds
               Question: What are names for top three branches with most number of membership?
               Answer: SELECT name FROM branch ORDER BY membership_amount DESC LIMIT 3
               ###
               Context: """

    fixed_few_shot_infix = "\n Question: "
    fixed_few_shot_postfix = "\n Answer: "

    train_spider['schema'] = train_spider['db_id'].map(schema_dict)
    train_spider['prompt'] = fixed_few_shot_prefix + train_spider['schema'] + fixed_few_shot_infix + train_spider['question'] + fixed_few_shot_postfix

    others_spider['schema'] = others_spider['db_id'].map(schema_dict)
    others_spider['prompt'] = fixed_few_shot_prefix + others_spider['schema'] + fixed_few_shot_infix + others_spider['question'] + fixed_few_shot_postfix

    dev_spider['schema'] = dev_spider['db_id'].map(schema_dict)
    dev_spider['prompt'] = fixed_few_shot_prefix + dev_spider['schema'] + fixed_few_shot_infix + dev_spider['question'] + fixed_few_shot_postfix
    return train_spider, others_spider, dev_spider


def load_spider_datasets():
    # datasets
    with open('spider/train_spider.json', 'r') as f:
        train_spider = pd.read_json(f)
    with open('spider/train_others.json', 'r') as f:
        others_spider = pd.read_json(f)
    with open('spider/dev.json', 'r') as f:
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