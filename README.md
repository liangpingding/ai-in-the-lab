# AI mode classfication
This repo contains code and data for paper **Tracking AI’s Scientific Anatomy: A Novel Framework for Analyzing the Use and Diffusion of AI in Science**


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

## Descriptive Plots
- Fig. 1. Trends in number/percentage of scientific publications by AI modes, 2000-2024, showing the slow growth of foundational AI research in science but the relatively recent rapid growth of AI adaption, tool, and discussion research in science.

![fig1](docs/figure2_yearly_trend.png)

- Fig. 2. Field distributions of AI relevant publications by AI mode. The left panel illustrates the distribution of AI publications within each domain by AI mode, with each bar summing to 100% of the AI publications in that field. The right panel presents the overall distribution of AI publications across fields based on their primary field classification, with the shares likewise summing to 100%. 
![domain_distributio](docs/figure3_domain_distribution.png)

- Fig. 3. Scientific publications by AI mode, classified by foundational, adaptation, tool, and discussion modes for 20 leading counties by all AI publications
![fig2](docs/Figure4_AI_mode_across_countries.png)  

- Fig. 4. Radar charts for six selected disciplines showing percentage of AI research publications in the top coutries by foundational, adaptation, tool, and discussion modes. The disciplines shown are computer science, engineering, materials science, medicine, biochemistry, genetics and molecular biology, and social sciences. 
![radar_country](docs/Picture5.png)


## Citation
Ding L, Lawson C, Shapira P (2025) Tracking AI’s Scientific Anatomy: A Novel Framework for Analyzing the Use and Diffusion of AI in Science. doi: 10.31235/osf.io/7ed2b_v1