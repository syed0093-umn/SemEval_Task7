# %%
# %pip install --upgrade numpy pandas matplotlib sentence-transformers torch tqdm

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Iterator
import ast
import numpy as np
# from sentence_transformers import SentenceTransformer
import torch
import gc
from tqdm import tqdm

# %%
import ast
import os

import pandas as pd

our_dataset_path = '/home/csgrads/syed0093/SemEval_Task7/Task_Data/'

posts_path = os.path.join(our_dataset_path, 'posts.csv')
fact_checks_path = os.path.join(our_dataset_path, 'fact_checks.csv')
fact_check_post_mapping_path = os.path.join(our_dataset_path, 'pairs.csv')

for path in [posts_path, fact_checks_path, fact_check_post_mapping_path]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    
parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
for col in ['claim', 'instances', 'title']:
    df_fact_checks[col] = df_fact_checks[col].apply(parse_col)


df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
for col in ['instances', 'ocr', 'verdicts', 'text']:
    df_posts[col] = df_posts[col].apply(parse_col)


df_fact_check_post_mapping = pd.read_csv(fact_check_post_mapping_path) 

# %%
# # Save the filtered DataFrame to a new CSV file
# df_fact_checks.to_csv('processed_fact_checks.csv', index=False)
# df_fact_checks.head()


# %%
# df_posts.to_csv('processed_posts.csv', index=False)
# df_posts.head()


# %%
# df_fact_check_post_mapping.to_csv('processed_pairs.csv', index=False)
# df_fact_check_post_mapping.head()

# %% [markdown]
# ### TFIDF

# %%
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

def parse_text_tuple(text_str):
    """Parse the tuple string format and extract text content."""
    try:
        # Convert string representation of tuple to actual tuple
        data = ast.literal_eval(text_str)
        if isinstance(data, tuple) and len(data) >= 2:
            # Return both original and translated text if available
            return ' '.join([str(data[0]), str(data[1])])
        return str(data[0])
    except:
        return text_str

# %%
def parse_instances(instances_str):
    """Parse the instances string to extract URLs and timestamps."""
    try:
        data = ast.literal_eval(instances_str)
        return [item[1] if isinstance(item, tuple) and len(item) > 1 else str(item) 
                for item in data]
    except:
        return []

# %%
def preprocess_post(row):
    """Combine relevant text fields from a post."""
    texts = []
    
    # Process text field
    if pd.notna(row.get('text')):
        try:
            text_data = ast.literal_eval(row['text'])
            if isinstance(text_data, list):
                for item in text_data:
                    if isinstance(item, tuple) and len(item) > 0:
                        texts.append(str(item[0]))  # Original text
            else:
                texts.append(str(text_data))
        except:
            texts.append(str(row['text']))
    
    return ' '.join(texts)

# %%
def create_retrieval_system(fact_checks_df, posts_df, task_config, language):
    """Create and train the retrieval system for a specific language."""
    # Filter fact checks for the specified language
    valid_fact_check_ids = task_config['monolingual'][language]['fact_checks']
    fact_checks = fact_checks_df[fact_checks_df['fact_check_id'].isin(valid_fact_check_ids)]
    
    # Prepare fact check texts
    fact_check_texts = []
    for _, row in fact_checks.iterrows():
        texts = []
        if pd.notna(row['claim']):
            texts.append(parse_text_tuple(row['claim']))
        if pd.notna(row['title']):
            texts.append(parse_text_tuple(row['title']))
        fact_check_texts.append(' '.join(texts))
    
    # Create TF-IDF vectors for fact checks
    vectorizer = TfidfVectorizer(max_features=5000)
    fact_check_vectors = vectorizer.fit_transform(fact_check_texts)
    
    return vectorizer, fact_check_vectors, fact_checks['fact_check_id'].tolist()

# %%
def retrieve_fact_checks(post_text, vectorizer, fact_check_vectors, fact_check_ids, top_k=10):
    """Retrieve the most relevant fact checks for a given post."""
    post_vector = vectorizer.transform([post_text])
    similarities = cosine_similarity(post_vector, fact_check_vectors).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [fact_check_ids[i] for i in top_indices]

# %%
def generate_predictions(posts_df, task_config, vectorizer, fact_check_vectors, 
                        fact_check_ids, language, split='posts_dev'):
    """Generate predictions for the development set."""
    predictions = {}
    valid_post_ids = task_config['monolingual'][language][split]
    
    for post_id in valid_post_ids:
        post = posts_df[posts_df['post_id'] == post_id].iloc[0]
        post_text = preprocess_post(post)
        retrieved_fact_checks = retrieve_fact_checks(
            post_text, vectorizer, fact_check_vectors, fact_check_ids
        )
        predictions[str(post_id)] = retrieved_fact_checks
    
    return predictions

# %%
def main():
    # Load data
    fact_checks = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/fact_checks.csv')
    posts = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/posts.csv')
    with open('tasks.json') as f:
        tasks = json.load(f)
    
    # Process for each language
    all_predictions = {}
    for language in tasks['monolingual'].keys():
        vectorizer, fact_check_vectors, fact_check_ids = create_retrieval_system(
            fact_checks, posts, tasks, language
        )
        
        predictions = generate_predictions(
            posts, tasks, vectorizer, fact_check_vectors, fact_check_ids, language
        )
        all_predictions.update(predictions)
    
    # Save predictions
    with open('monolingual_predictions_tfidf.json', 'w') as f:
        json.dump(all_predictions, f)

if __name__ == "__main__":
    main()

# %% [markdown]
# ## 1. BERT

# %%
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import ast
from tqdm import tqdm

class BERTRetriever:
    def __init__(self, model_name='xlm-roberta-base', max_length=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts):
        embeddings = []
        
        for text in tqdm(texts, desc="Generating embeddings"):
            encoded = self.tokenizer(text, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=self.max_length, 
                                   return_tensors='pt')
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded)
            
            sentence_embedding = self.mean_pooling(model_output, encoded['attention_mask'])
            embeddings.append(sentence_embedding.cpu().numpy()[0])
        
        return np.array(embeddings)

def parse_text_tuple(text_str):
    try:
        data = ast.literal_eval(text_str)
        if isinstance(data, tuple) and len(data) >= 2:
            return ' '.join([str(data[0]), str(data[1])])
        return str(data[0])
    except:
        return text_str

def create_retrieval_system(fact_checks_df, posts_df, task_config, language):
    retriever = BERTRetriever()
    
    # Filter fact checks
    valid_fact_check_ids = task_config['monolingual'][language]['fact_checks']
    fact_checks = fact_checks_df[fact_checks_df['fact_check_id'].isin(valid_fact_check_ids)]
    
    # Prepare fact check texts
    fact_check_texts = []
    for _, row in fact_checks.iterrows():
        texts = []
        if pd.notna(row['claim']):
            texts.append(parse_text_tuple(row['claim']))
        if pd.notna(row['title']):
            texts.append(parse_text_tuple(row['title']))
        fact_check_texts.append(' '.join(texts))
    
    # Generate embeddings
    fact_check_vectors = retriever.get_embeddings(fact_check_texts)
    
    return retriever, fact_check_vectors, fact_checks['fact_check_id'].tolist()

def retrieve_fact_checks(post_text, retriever, fact_check_vectors, fact_check_ids, top_k=10):
    post_vector = retriever.get_embeddings([post_text])
    similarities = cosine_similarity(post_vector, fact_check_vectors).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [fact_check_ids[i] for i in top_indices]

def preprocess_post(row):
    texts = []
    if pd.notna(row.get('text')):
        try:
            text_data = ast.literal_eval(row['text'])
            if isinstance(text_data, list):
                for item in text_data:
                    if isinstance(item, tuple) and len(item) > 0:
                        texts.append(str(item[0]))
            else:
                texts.append(str(text_data))
        except:
            texts.append(str(row['text']))
    return ' '.join(texts)

def main():
    # Load data
    fact_checks = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/fact_checks.csv')
    posts = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/posts.csv')
    with open('tasks.json') as f:
        tasks = json.load(f)
    
    all_predictions = {}
    for language in tasks['monolingual'].keys():
        print(f"\nProcessing language: {language}")
        
        retriever, fact_check_vectors, fact_check_ids = create_retrieval_system(
            fact_checks, posts, tasks, language
        )
        
        valid_post_ids = tasks['monolingual'][language]['posts_dev']
        for post_id in tqdm(valid_post_ids, desc="Generating predictions"):
            post = posts[posts['post_id'] == post_id].iloc[0]
            post_text = preprocess_post(post)
            retrieved_fact_checks = retrieve_fact_checks(
                post_text, retriever, fact_check_vectors, fact_check_ids
            )
            all_predictions[str(post_id)] = retrieved_fact_checks
    
    # Save predictions
    with open('monolingual_predictions_bert_1.json', 'w') as f:
        json.dump(all_predictions, f)

if __name__ == "__main__":
    main()

# %% [markdown]
# ## BERT 2

# %%
class BERTRetriever:
    def __init__(self, model_name='xlm-roberta-base'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def get_embeddings(self, texts):
        embeddings = []
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(batch, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=512, 
                                   return_tensors='pt')
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use CLS token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

def process_text(text_str):
    if pd.isna(text_str):
        return ""
    try:
        data = ast.literal_eval(text_str)
        if isinstance(data, tuple):
            return str(data[0])  # Use original text
        elif isinstance(data, list):
            return ' '.join(str(item[0]) if isinstance(item, tuple) else str(item) for item in data)
        return str(data)
    except:
        return str(text_str)

def main():
    print("Loading data...")
    fact_checks = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/fact_checks.csv')
    posts = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/posts.csv')
    pairs = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/pairs.csv')  # Load gold pairs
    with open('tasks.json') as f:
        tasks = json.load(f)
    
    retriever = BERTRetriever()
    all_predictions = {}
    
    for language in tasks['monolingual'].keys():
        print(f"\nProcessing {language}")
        
        # Filter fact checks for language
        valid_fact_checks = tasks['monolingual'][language]['fact_checks']
        language_fact_checks = fact_checks[fact_checks['fact_check_id'].isin(valid_fact_checks)]
        
        # Prepare fact check texts
        fact_check_texts = []
        for _, row in language_fact_checks.iterrows():
            text = process_text(row['claim']) + " " + process_text(row['title'])
            fact_check_texts.append(text)
        
        print("Generating fact check embeddings...")
        fact_check_vectors = retriever.get_embeddings(fact_check_texts)
        fact_check_ids = language_fact_checks['fact_check_id'].tolist()
        
        # Process dev posts
        dev_post_ids = tasks['monolingual'][language]['posts_dev']
        dev_posts = posts[posts['post_id'].isin(dev_post_ids)]
        
        print("Processing posts...")
        for _, post in tqdm(dev_posts.iterrows()):
            post_text = process_text(post['text'])
            post_vector = retriever.get_embeddings([post_text])
            
            # Calculate similarities and get top matches
            similarities = cosine_similarity(post_vector, fact_check_vectors).flatten()
            top_indices = np.argsort(similarities)[-10:][::-1]
            
            # Store predictions with correct ID type
            all_predictions[str(int(post['post_id']))] = [str(fact_check_ids[i]) for i in top_indices]
    
    print("\nSaving predictions...")
    with open('monolingual_predictions_bert_2.json', 'w') as f:
        json.dump(all_predictions, f)

if __name__ == "__main__":
    main()

# %% [markdown]
# ### Infloat E5 multilingual large

# %%
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json
from tqdm import tqdm
import numpy as np

class E5Retriever:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(self.device)
    
    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def get_embeddings(self, texts, batch_size=8):
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_dict = self.tokenizer(batch, max_length=512, padding=True, 
                                      truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            emb = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings.extend(F.normalize(emb, p=2, dim=1).cpu().numpy())
            
        return np.array(embeddings)

def format_text(row, type='fact_check'):
    if type == 'fact_check':
        claim = row.get('claim', '')
        title = row.get('title', '')
        return f"passage: {claim} {title}"
    else:
        return f"query: {row.get('text', '')}"

def main():
    retriever = E5Retriever()
    fact_checks = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/fact_checks.csv')
    posts = pd.read_csv('/home/csgrads/syed0093/SemEval_Task7/Task_Data/posts.csv')
    
    with open('tasks.json') as f:
        tasks = json.load(f)
    
    predictions = {}
    
    for language in tasks['monolingual'].keys():
        valid_fact_checks = tasks['monolingual'][language]['fact_checks']
        language_fact_checks = fact_checks[fact_checks['fact_check_id'].isin(valid_fact_checks)]
        
        fact_check_texts = [format_text(row) for _, row in language_fact_checks.iterrows()]
        fact_check_vectors = retriever.get_embeddings(fact_check_texts)
        fact_check_ids = language_fact_checks['fact_check_id'].tolist()
        
        dev_post_ids = tasks['monolingual'][language]['posts_dev']
        dev_posts = posts[posts['post_id'].isin(dev_post_ids)]
        
        for _, post in tqdm(dev_posts.iterrows(), desc=f"Processing {language}"):
            post_text = format_text(post, type='post')
            post_vector = retriever.get_embeddings([post_text])
            
            scores = (post_vector @ fact_check_vectors.T)[0]
            top_indices = np.argsort(scores)[-10:][::-1]
            
            predictions[str(int(post['post_id']))] = [str(fact_check_ids[i]) for i in top_indices]
    
    with open('monolingual_predictions_e5_large.json', 'w') as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()

# %%



