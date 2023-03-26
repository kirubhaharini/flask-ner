# flask-ner
Deploying Tranformer-based NER Models from any domain

****
Instructions to Run:
******
1. Clone folder to local 
2. Create virtual env and install requirements using 'pip install -r requirements.txt'
3. Replace 'model.bin' and 'tokenizer' with your own model
4. Edit model_config.yaml file to change model configurations
5. Under ner_model.py, make changes to map_label() and map_color() functions to change entities and their respective colours according to the model
6. Run 'python app.py' to start flask app
******
