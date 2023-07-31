from t5.dataset import DatasetIterator
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


def set_train_arguments(train_path,
                        batch_size,
                        num_epochs,
                        learning_rate,
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        **kwargs):
    args = Seq2SeqTrainingArguments(
        train_path,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        **kwargs
    )
    return args


class BaseModel:
    def __init__(self,
                 model,
                 hug_model_name:str,
                 train_data_iterator:DatasetIterator,
                 valid_data_iterator:DatasetIterator,
                 seq2seq_train_args: Seq2SeqTrainingArguments,
                 **kwargs
                 ):
        self.model = model
        self.hug_model_name = hug_model_name

        self.trainer = Seq2SeqTrainer(
            self.model,
            seq2seq_train_args,
            train_dataset=train_data_iterator,
            eval_dataset=valid_data_iterator,
            **kwargs
        )
    
    def train(self):
        self.trainer.train()
        
    def train_from_checkpoint(self, last_check_point):
        self.trainer.train(resume_from_checkpoint=last_check_point)
        
    def evaluate(self):
        print(self.trainer.evaluate())
        
    def model_save(self, model_path):
        self.trainer.save_model(model_path)
    
    def model_upload(self):
        self.model.push_to_hub(f"RoxyRong/{self.hug_model_name}", use_auth_token=True)