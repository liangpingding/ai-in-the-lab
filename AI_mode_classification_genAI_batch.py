
#------------
# reference: https://cookbook.openai.com/examples/batch_processing
# using this code to generate the synthetic data using GPT for AI type classification
#------------
#%%
import json
import openai
import pandas as pd
import os
import sys
import re

dataset_path = "Data/human_annotation.csv"
json_file_name = "Data/batch_tasks.jsonl"
result_file_name = "Data/batch_job_results.jsonl"
# Load the dataset
df = pd.read_csv(dataset_path)
df.head()

OPENAI_KEY = ''
client = openai.OpenAI(api_key=OPENAI_KEY)  # Initialize the OpenAI client
#%%
categorize_system_prompt =  """
You are a research assistant tasked with analyzing the title and abstract of a scientific paper. Your goal is to classify the paper into one of the following categories:
- **Discussion**: The paper focuses on analyzing, critiquing, or reflecting on AI without implementing AI models in its research methodology. This includes literature reviews, ethical or societal commentary, perception studies, interviews, qualitative research, and bibliometric or meta-analyses.
- **Foundational**: The paper proposes new AI models, algorithms, training methods, or theoretical contributions that improve AI’s core capabilities.
- **Tool** The paper uses existing AI models (without changing their architecture) to solve domain-specific problems (e.g., in natural science, engineering, agriculture, medicine, education, law, social science, or the humanities).
-**Adaptation**. The paper modifies or adapts the model architecture of existing AI models to better suit specific tasks or domains, without proposing fundamentally new AI methods.
- **Unclear**: The abstract is too vague, lacks sufficient detail, or does not clearly fit any of the above categories.
**Output Format**
Return a JSON object in the following format:
{
  "AI_mode_GPT":   "<Discussion | Foundational | Tool | Adaptation | Unclear >",
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
            # This is what you would have in your Chat Completions API call
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
# STEP 3: save the results
#--------------------------
# batch_job = client.batches.retrieve(batch_job.id)
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



df['AI_mode_GPT'] = ''
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
    df['AI_mode_GPT'].iloc[int(index)] = format_res['AI_mode_GPT']

    print(f"OVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")
    
df.to_csv('Data/human_annotation.csv',index=False)


