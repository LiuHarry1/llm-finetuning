from atlassian import Jira
import pandas as pd

# Connect to your JIRA instance
jira = Jira(
    url='https://your-jira-instance.atlassian.net',
    username='your-email',
    password='your-api-token'
)

# JIRA Query to extract issues from a specific project
query = 'project=YOUR_PROJECT_KEY'
issues = jira.jql(query, limit=1000)

# Extract relevant fields
jira_data = []
for issue in issues['issues']:
    jira_data.append({
        'issue_key': issue['key'],
        'summary': issue['fields']['summary'],
        'description': issue['fields']['description'],
        'issue_type': issue['fields']['issuetype']['name']
    })

# Convert the data into a Pandas DataFrame
df = pd.DataFrame(jira_data)

# Save the extracted data
df.to_csv('jira_data.csv', index=False)


# Load the extracted data
df = pd.read_csv('jira_data.csv')

# Prepare the dataset in instruction-response format
fine_tuning_data = []
for _, row in df.iterrows():
    if pd.notna(row['description']):  # Ensure there's a description
        fine_tuning_data.append({
            "instruction": row['summary'],  # Use summary as the instruction
            "response": row['description']  # Use description as the response
        })

# Convert to a DataFrame and save it as a JSON file
fine_tuning_df = pd.DataFrame(fine_tuning_data)
fine_tuning_df.to_json('fine_tuning_data.json', orient='records')



from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Load the fine-tuning data
dataset = load_dataset('json', data_files='fine_tuning_data.json')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['instruction'], examples['response'], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define PEFT configuration
peft_config = PeftConfig(
    task_type="causal_lm",
    lora_rank=4
)

# Fine-tune the model using PEFT
peft_model = PeftModel(model, peft_config)

# Use Trainer API for training (replace with your trainer code if needed)
peft_model.train()

# Save the fine-tuned model
peft_model.save_pretrained('./fine_tuned_llama2_model')

user_question = "How do I enable 2FA in our app?"

# Tokenize and generate a response
inputs = tokenizer(user_question, return_tensors="pt")
response = peft_model.generate(**inputs)

# Decode and print the response
print(tokenizer.decode(response[0], skip_special_tokens=True))
