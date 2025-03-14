from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import os

app = FastAPI(title="User Similarity Search")

# Initialize Qdrant client
qdrant_host = os.environ.get("QDRANT_HOST", "qdrant")
qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

# Constants
COLLECTION_NAME = "users1"
VECTOR_SIZE = 768  # Adjust based on Gemma3's embedding size

try:
    # Check if collection exists by listing all collections
    collections = qdrant.get_collections().collections
    collection_exists = any(collection.name == COLLECTION_NAME for collection in collections)
    
    if not collection_exists:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")
except Exception as e:
    print(f"Error during collection setup: {str(e)}")
    # Re-raise if this is a critical error
    raise

class User(BaseModel):
    name: str
    bio: str
    interests: list[str]
    location: str
    age: int = None

class UserResponse(BaseModel):
    id: str
    user: User
    similarity_score: float = None

async def get_embedding(text: str) -> list[float]:
    """Generate embedding using local Ollama model"""
    ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
    
    # Try different models in order of preference
    models_to_try = ["nomic-embed-text"]
    
    for model in models_to_try:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"http://{ollama_host}:11434/api/embeddings",            
                    json={
                        "model": model,
                        "prompt": text
                    }
                )
                
                if response.status_code != 200:
                    print(f"Model {model} failed with status {response.status_code}")
                    continue
                    
                result = response.json()
                if result.get('embedding') is None:
                    print(f"Model {model} returned null embeddings")
                    continue
                    
                return result['embedding']
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")
    
    # If we get here, all models failed
    raise HTTPException(
        status_code=500, 
        detail="Failed to generate embedding with any available model"
    )

def user_to_text(user: User) -> str:
    """Convert user object to text representation for embedding"""
    interests = ", ".join(user.interests)
    return f"Name: {user.name}. Bio: {user.bio}. Interests: {interests}. Location: {user.location}. Age: {user.age if user.age else 'Unknown'}"

@app.post("/users/", response_model=UserResponse)
async def create_user(user: User):
    """Store a new user with embedding in the vector database"""
    user_text = user_to_text(user)
    embedding = await get_embedding(user_text)
    
    user_id = str(uuid.uuid4())
    
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=user_id,
                vector=embedding,
                payload=user.dict()
            )
        ]
    )
    
    return {"id": user_id, "user": user}

@app.post("/users/find-similar/", response_model=UserResponse)
async def find_similar_user(user: User):
    """Find the most similar user to the provided one"""
    user_text = user_to_text(user)
    embedding = await get_embedding(user_text)
    
    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=1
    )
    
    if not search_results:
        raise HTTPException(status_code=404, detail="No similar users found")
    
    top_result = search_results[0]
    return {
        "id": top_result.id,
        "user": User(**top_result.payload),
        "similarity_score": top_result.score
    }

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get a specific user by ID"""
    results = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[user_id]
    )
    
    if not results:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {
        "id": results[0].id,
        "user": User(**results[0].payload)
    }

@app.get("/users/", response_model=list[UserResponse])
async def list_users():
    """List all users"""
    # Get the first 100 users (you might want to add pagination)
    results = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=100
    )[0]
    
    return [
        {"id": point.id, "user": User(**point.payload)}
        for point in results
    ]