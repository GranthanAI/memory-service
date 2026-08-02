import uuid
from datetime import datetime, timezone
from app.db.cassandra import connect_cassandra, get_session
from app.repositories.cassandra_repository import CassandraRepository

connect_cassandra()
session = get_session()
repo = CassandraRepository(session)

job_id = uuid.uuid4()
created_at = datetime.now(timezone.utc)

job = {
    "job_id": job_id,
    "topic": "test.topic",
    "conversation_id": "conv-1",
    "payload": "payload",
    "created_at": created_at
}

repo.insert_outbox_job(job)
pending = repo.get_pending_outbox_jobs(limit=10)
target_job = [j for j in pending if j["job_id"] == job_id][0]

print("Initial target job in DB:", target_job)

first_claim = repo.claim_outbox_job(target_job)
print("First claim applied status:", first_claim)

# Fetch PENDING row again
pending_after = repo.get_pending_outbox_jobs(limit=10)
print("Pending after first claim (should be empty for this job):", [j for j in pending_after if j["job_id"] == job_id])

# Fetch PROCESSING row
processing_row = repo.get_outbox_job("PROCESSING", target_job["created_at"], job_id)
print("Processing row in DB:", processing_row)

second_claim = repo.claim_outbox_job(target_job)
print("Second claim applied status:", second_claim)
