from llama_index.embeddings.voyageai import VoyageEmbedding
import voyageai
import os
from dotenv import load_dotenv # Import the function

load_dotenv() # Load variables from .env file into environment

voyage_api_key = os.getenv("VOYAGE_API_KEY")
print(voyage_api_key)
vo = voyageai.Client(api_key=voyage_api_key)
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

result = vo.embed(["hello world"],  model="voyage-3",  input_type="document").embeddings

print(result)
# # with input_typ='search_query'
embed_model = VoyageEmbedding(
    voyage_api_key="pa-T-3hCgvsF3hsSelQ_Mp3nfPaQhBRX2yxcxe_SXOa86y",
    model_name="voyage-3",
    truncation=False,
    input_type="document"
)

embeddings = embed_model.get_text_embedding("Hello VoyageAI!")

print(len(embeddings))
# print(embeddings[:5])