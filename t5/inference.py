import subprocess
import pprint

def inference(dev_spider, model, tokenizer, result_path):
    input_max_length = 512,
    output_max_length = 128
    step = 100

    for idx in range(0, 1100, step):
        print(idx)
        inputs = tokenizer.batch_encode_plus(
            list(dev_spider.iloc[idx:idx+step]['prompt']),
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        output_tokens = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=output_max_length
        )

        outputs = [tokenizer.decode(i, skip_special_tokens=True) for i in output_tokens]

        with open(result_path, 'a', encoding='utf-8') as f:
            for idx, output in enumerate(outputs):
                db_id = dev_spider.iloc[idx]['db_id']
                f.write(output + '\t' + db_id + '\n')

def evaluate_result(result_path):
    eval_path = f"third_party/spider/evaluation.py"
    gold = f"third_party/spider/evaluation_examples/gold_example.txt"
    pred = result_path
    db_dir = f"spider/database"
    table = f"spider/tables.json"
    etype = "all"

    cmd_str = f"python3 \"{eval_path}\" --gold \"{gold}\" --pred \"{pred}\" --db \"{db_dir}\" --table \"{table}\" --etype {etype} "
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    pprint.pprint(result.stdout[-4633:])