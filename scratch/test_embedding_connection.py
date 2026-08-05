import asyncio
import grpc
from app.proto import embedding_pb2, embedding_pb2_grpc

async def test_embed():
    print("Connecting to gRPC on localhost:50051...")
    channel = grpc.aio.insecure_channel("localhost:50051")
    stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
    try:
        response = await stub.GenerateEmbedding(
            embedding_pb2.EmbeddingRequest(text="Hello world from memory service test client!"),
            timeout=5.0
        )
        print("Success!")
        print(f"Model: {response.model}")
        print(f"Dimension: {response.dimension}")
        print(f"Embedding length: {len(response.embedding)}")
        print(f"First 5 elements: {response.embedding[:5]}")
    except Exception as e:
        print(f"gRPC call failed: {e}")
    finally:
        await channel.close()

if __name__ == "__main__":
    asyncio.run(test_embed())
