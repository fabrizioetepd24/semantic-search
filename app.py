import os
import pandas as pd
import numpy as np

import openai
from openai.embeddings_utils import get_embedding

from openai.embeddings_utils import cosine_similarity

openai.api_key ="sua senha openai"

df = pd.read_csv('/home/fabrizio/projetos-llms/projeto-search/esportes.csv')

print(df)

df['embedding'] = df['texto'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

df.to_csv('base_embeddings.csv')

# converter para array numpy
df['embedding'] = df['embedding'].apply(np.array)

user_search = input('Digite o nome de um time de algum esporte: ')

user_search_embedding = get_embedding(user_search, engine='text-embedding-ada-002')


# calcula a similaridade do cosseno entre o termo de pesquisa e cada palavra no dataframe
df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, user_search_embedding))

#print(df)

df = df.sort_values(by='similarity', ascending=False)

# top 10
print(df.head(10))
