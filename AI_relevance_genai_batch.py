
#------------
# reference: https://cookbook.openai.com/examples/batch_processing
# use this code to generate the synthetic data using GPT for AI relevance classification
# the input data file: Data/AI_relevance_training/AI_nonAI_sampled_abstracts.csv (20057 rows)
# the output data file: Data/AI_relevance_training/AI_nonAI_sampled_abstracts.csv (20057 rows, but added column AI_revelance_keywords)
#------------
#%%
import json
from openai import OpenAI
import pandas as pd
import os
import sys
import re
sys.path.append('/home/q25046ld/Project/AI_type_classification')
os.chdir(sys.path[-1])
dataset_path = "Data/human_annotation.csv"
json_file_name = "Data/batch_tasks.jsonl"
result_file_name = "Data/batch_job_results.jsonl"
# Load the dataset
df = pd.read_csv(dataset_path)
df.head()
OPENAI_KEY = ''
client = OpenAI(api_key=OPENAI_KEY)  # Initialize the OpenAI client
#%%
categorize_system_prompt =  """
You are a research assistant tasked with identifying whether a scientific paper is relevant to Artificial intelligence (AI) technology, as indicated by AI-related terms in its title and abstract. Output 1 if the paper is AI-relevant. Output 0 if the paper is not AI-relevant.
 and return the results in the specified JSON format.
{
  "AI_relevance_GPT": "0 | 1",
}
"""


#%%

#-------------------------
# STEP 1: Prepare the batch data
#--------------------------

tasks = []

for index, row in df.iterrows():
    
    description = row['title_abs']
    
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "temperature": 0,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": categorize_system_prompt
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
        }
    }
    
    tasks.append(task)



with open(json_file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')


#%%
#-------------------------
# STEP 2: Update the job
#--------------------------
batch_file = client.files.create(
  file=open(json_file_name, "rb"),
  purpose="batch"
)

batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)




#%%
#-------------------------
# STEP 3: Read the results
#--------------------------

batch_job = client.batches.retrieve(batch_job.id)
print("job id: {}".format(batch_job))
result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content
with open(result_file_name, 'wb') as file:
    file.write(result)
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)

df['AI_relevance_GPT'] = ''
for res in results:
    task_id = res['custom_id']
    # Getting index from task id
    index = task_id.split('-')[-1]
    row = df.iloc[int(index)]
    description = row['title_abs']
    result=res['response']['body']['choices'][0]['message']['content'].replace('json:','').replace('json\n','').strip().replace("'","").replace('```','')
    cleaned_string = re.sub(r"[\\']", '',  result)  
    cleaned_string = re.sub(r"[()]", '', cleaned_string)  
    format_res=json.loads(cleaned_string)
    df['AI_relevance_GPT'].iloc[int(index)] = format_res['AI_relevance_GPT']
    print(f"OVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")
df.to_csv('Data/human_annotation.csv',index=False)

