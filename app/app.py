from sentence_transformers import SentenceTransformer
import pandas as pd
import json

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

emb = model.encode(docs)

# import model and vectorize the books descriptions and get everything done in vectors
# check that no fine tuninig is needed
# hit the model to check everything is alright
# endpotins
#   adding books vectors
#   add user vector
#   search by text(wether genre or title)
# add logger in both apps
