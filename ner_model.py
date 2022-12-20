from NERP.models import NERP
from NERP.inference import inference_pipeline, load_model
import functools
from functools import lru_cache
from frozendict import frozendict

def freezeargs(func):
    # To transform dict arguments in function (mutable) to immutable form so we can cache the function   

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped

#fn to load model weights: (cached)
@freezeargs
@lru_cache(maxsize=None)
def pre_load_model(model_path,tokenizer_path,archi,bert_variant,tag_scheme,hyperparameters,tokenizer_parameters,max_len,device):
    tag_scheme = list(tag_scheme)
    model = load_model(archi, device, tag_scheme, bert_variant, max_len,
                       model_path, tokenizer_path, hyperparameters, tokenizer_parameters)
    return model


#fn to perform prediction after loading model weights:
@freezeargs
@lru_cache(maxsize=None)
def model_predict(input_text,model_path,tokenizer_path,archi,bert_variant,tag_scheme,hyperparameters,tokenizer_parameters,max_len,device):
    model = pre_load_model(model_path,tokenizer_path,archi,bert_variant,tag_scheme,hyperparameters,tokenizer_parameters,max_len,device)
    output = [model.predict_text(input_text), "Predicted successfully!"]

    entities = output[0][0]
    labels = output[0][1]

    output_list = combine_similar_entities(entities,labels)
    final_output_list = preprocess_list(output_list)
    return final_output_list

#fns to preprocess output
def combine_similar_entities(entities,labels):
  output_list = []
  skip_tokens = [] #tokens to skip
  for i in range(len(entities)): #for each sentence
    sentence = entities[i]
    label = labels[i]
    for j in range(len(sentence)): #for word in sentence
      if j not in skip_tokens:
        current_token = sentence[j]
        current_label = label[j]
        
        for k in range(j+1,len(sentence)): #next tokens
      
        #if j != len(sentence)-1: #if j not the last token: check token+1 (next token) to see if its 'I' tag 
          next_label = label[k]
          next_token = sentence[k]
          if next_label[0] == 'I': #intermediate token - combine with previous token
            token_to_add = ' ' + next_token
            current_token += token_to_add
            skip_tokens.append(k) #skip this token
          else: break
            
        if current_label != 'O':
          current_label = current_label[1:] #remove BI tags
          current_label = current_label.replace('-','')
    
        token_label_pair = (current_token,current_label)
        
        output_list.append(token_label_pair)
      
  return output_list


def preprocess_list(output_list):
  final_output_list = []
  skip_j = []

  for i in range(len(output_list)):

    if i not in skip_j:
      current_pair = output_list[i]
      current_phrase = current_pair[0]
      current_entity = current_pair[1]
      
      combine_phrases = '' #combine if tag is O (consecutively)
      if current_entity == 'O':
        combine_phrases = current_phrase
      
        for j in range(i+1,len(output_list)):
          if j not in skip_j:
            next_pair = output_list[j]
            next_phrase = next_pair[0]
            next_entity = next_pair[1]    
            
            if next_entity == 'O':
              combine_phrases += ' '
              combine_phrases += next_phrase
              skip_j.append(j)
            
            else:
              break
        
      if (combine_phrases == '') or (combine_phrases == current_phrase):
        if current_entity != 'O':
          current_entity = map_label(current_entity)
        append_pair = (current_phrase,current_entity) #no change
      else:
        append_pair = (combine_phrases, 'O')
        
        
      final_output_list.append(append_pair)
      
  return final_output_list
    

def map_label(label):
  label_map = {
      'PER'  : 'Person',
      'ORG'  : 'Organization',
      'LOC'  : 'Location',
      'MISC' : 'Miscellaneous'
  }
  return label_map[label]

def map_color(label):
  color_map = {
    'Person' : '#94CDDF',
    'Organization' : '#fabed4',
    'Location' : '#F0C77E',
    'Miscellaneous' : '#dcbeff'
  }
  return color_map[label]