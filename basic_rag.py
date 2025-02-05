import json
import torch
import numpy as np
import os
from annoy import AnnoyIndex  # Spotify's Annoy library
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer  # For embeddings
import getpass

# Ensure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load crypto descriptions
with open("data/cryptos.json", "r", encoding="utf-8") as f:
    crypto_data = json.load(f)

# Filter out empty descriptions
crypto_data = [c for c in crypto_data if c["description"].strip()]
project_descriptions = [entry["description"] for entry in crypto_data]
project_names = [f'{entry["name"]} ({entry["symbol"].upper()})' for entry in crypto_data]

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode descriptions into embedding vectors
print("Encoding descriptions...")
crypto_embeddings = embedding_model.encode(
    project_descriptions, convert_to_numpy=True, show_progress_bar=True
)

# Build Annoy index
d = crypto_embeddings.shape[1]  # Dimensionality of embeddings
annoy_index = AnnoyIndex(d, metric="angular")  # Cosine similarity

# Add items to Annoy index
for i, vector in enumerate(crypto_embeddings):
    annoy_index.add_item(i, vector)

annoy_index.build(10)  # 10 trees

# Load Generative Model (GPT-based)
gen_model = ChatOpenAI(model="gpt-3.5-turbo-0125")


def retrieve_top_k(query: str, k: int = 2):
    """
    Encode the query and find the top-k most similar crypto descriptions using Annoy.
    Returns a list of (description, name_symbol, similarity).
    """
    query_vec = embedding_model.encode([query], convert_to_numpy=True)[0]
    indices = annoy_index.get_nns_by_vector(query_vec, k, include_distances=True)
    
    results = [
        (project_descriptions[i], project_names[i], dist)
        for i, dist in zip(indices[0], indices[1])
    ]
    return results


def generate_response(query: str, retrieved_docs: list):
    """
    Generate a response using GPT-3.5 with RAG-style retrieval.
    """
    context = "\n\n".join(
        [f"{name}: {desc}" for desc, name, _ in retrieved_docs]
    )

    prompt_template = """Use the following context to answer the question concisely.
    If you don't know, say that you don't know. Limit your answer to three sentences.

    Context:
    {context}

    Question: {query}

    Answer:"""

    prompt = prompt_template.format(context=context, query=query)

    # Generate response using OpenAI's model
    response = gen_model.invoke(prompt)
    
    return response.content.strip()


if __name__ == "__main__":
    print("Welcome to the Crypto RAG Demo!")
    
    while True:
        query = input("\nEnter your query (type 'quit' to exit): ")
        if query.lower() == "quit":
            break

        top_matches = retrieve_top_k(query, k=2)
        response = generate_response(query, retrieved_docs=top_matches)
        print(response)