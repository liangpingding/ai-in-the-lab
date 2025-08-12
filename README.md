# AI mode classfication
This repo contains code and data for paper **Assessing how AI is being used in science: a public tool for policy analysis**


## GPT prompts
### Stage I: AI relevance classfication 
![AI relevance GPT prompt](docs/AI_relevance_classification.png)

### Stage II: AI mode classfication 

![AI relevance GPT prompt](docs/AI_mode_classification.png)

## Environment
Python 3.11, transformers v4.37.2, openai v1.98.0


## File reporsitory
- `AI_relevance_genai_batch.py`: code for generating synthetic data for AI relevance classification using GPT
- `AI_mode_classification_genAI_batch.py`: code for generating synthetic data for AI mode classification using GPT
- `run_classification.py`: code for running the classification tasks using SciBERT
- `Scripts/run_classification_AI_relevance.sh`: script for running the AI relevance classification task
- `Scripts/run_classification_AI_mode.sh`: script for running the AI mode classification task