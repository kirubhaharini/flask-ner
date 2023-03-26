from ner_model import model_predict, map_color, map_label
import yaml
from yaml.loader import SafeLoader

def return_html_from_model(input_text):

    with open('model_config.yaml', 'r') as f:
        data = list(yaml.load_all(f, Loader=SafeLoader))[0]

    model_path = data['model_path']
    tokenizer_path = data['tokenizer_path']
    archi = data['archi']
    pretrained = data['pretrained']
    tag_scheme = tuple(data['tag_scheme'])
    hyperparameters = data['hyperparameters']
    tokenizer_parameters = data['tokenizer_parameters']
    max_len = data['max_len']
    device = data['device']

    output = model_predict(input_text,model_path,tokenizer_path,archi,pretrained,tag_scheme,hyperparameters,tokenizer_parameters,max_len,device)
    print('finished running prediction')

    basic_html_start = '<div class="entities" style="line-height: 2.5; direction: ltr">'
    basic_html_end = '</div>'

    html = ''
    html += basic_html_start
    for i in range(len(output)):
        pair = output[i]
        phrase = pair[0]
        entity = pair[1]

        add_entity_html = ''
        if entity != 'O':
            entity = map_label(entity)
            color = map_color(entity)
            add_entity_html = '<mark class="entity" style="background: {}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;"> {} <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">{}</span> </mark>'.format(color,phrase,entity)
        else: add_entity_html = phrase

        html += add_entity_html

    html += basic_html_end

        
    return html
