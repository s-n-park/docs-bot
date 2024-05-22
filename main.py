import pandas as pd
import os
from openai import OpenAI
import numpy as np
from typing import List
from scipy import spatial
import redis
import pickle

# Global variable to store the dataset
df = None

def initialize_client(openai_key: str):
    # set environment key
    os.environ['OPENAI_API_KEY'] = openai_key

    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return client

def get_embeddings(filename: str,use_redis=True):
    if use_redis==True:
        # Connect to Redis
        r = redis.Redis(host='<YOUR_HOST_HERE>', port=6379, db=0)

        # Try to get the embeddings from the Redis cache
        df = r.get('embeddings')

        # If the embeddings are not in the cache, compute them and store them in the cache
        if df is None:
            if os.path.exists(filename):
                df = pd.read_csv(filename,index_col=0)
                df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
            # Store the dataframe in the Redis cache
            r.set('embeddings', pickle.dumps(df))
        else:
            # If the embeddings are in the cache, load them
            df = pickle.loads(df)
    
    else: # Version where we don't use Redis
        if os.path.exists(filename):
            df = pd.read_csv(filename,index_col=0)
            
            # Convert the embeddings from strings to arrays
            df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    return df

# taken from: https://github.com/openai/openai-python/blob/release-v0.28.0/openai/embeddings_utils.py
def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances

def create_context(client, question, df, size="ada",articles=3):
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-3-large').data[0].embedding
    q_embeddings = np.array(q_embeddings)

    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    relevant_articles = df.sort_values('distances', ascending=True).head(articles)
    return relevant_articles

def answer_question(client,df, model="gpt-4o", question="", size="ada", debug=False):
    relevant_articles = create_context(client, question, df, size=size)
    context = ' '.join(relevant_articles['text'].tolist())
    prompt = context + "\n\n" + question
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    articles = '\n'.join('* [' + url + '](' + url + ')' for url in relevant_articles['url'].tolist())
    if response.choices[0].message.content.strip():
        return response.choices[0].message.content.strip() + "\n\nSource:\n" + articles
        # return response.to_json() # OPTIONAL
    else:
        return "I'm not sure the answer, but here are links to relevant documentation:\n" + articles

def get_answer(openai_key: str, question: str) -> str:
    global df
    # Load the dataset only if it's not already loaded
    if df is None:
        file_name = 'doc_text_embeddings_3_large.csv'
        df = get_embeddings(file_name,use_redis=False)
    client = initialize_client(openai_key)
    answer = answer_question(client, df, question=question)
    return answer

if __name__ == "__main__":
    main()