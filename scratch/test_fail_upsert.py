import uuid
from datetime import datetime, timezone
from app.db.cassandra import connect_cassandra, get_session
from app.repositories.cassandra_repository import CassandraRepository

connect_cassandra()
session = get_session()
repo = CassandraRepository(session)

session.execute("TRUNCATE outbox_jobs")

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
target_job = pending[0]

repo.claim_outbox_job(target_job)
job_processing = repo.get_outbox_job("PROCESSING", created_at, job_id)

# Perform upsert update without delete
repo.session.execute(repo._insert_outbox, (
    "PROCESSING",
    job_processing["created_at"],
    job_id,
    job_processing["topic"],
    job_processing["conversation_id"],
    job_processing["payload"],
    job_processing["attempt_count"] + 1,
    "some error",
    job_processing["claimed_at"]
))

# Select all
rows = list(session.execute("SELECT * FROM outbox_jobs"))
print("All rows after upsert:")
for r in rows:
    print(r._asdict())
