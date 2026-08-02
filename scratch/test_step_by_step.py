import uuid
import datetime
from app.db.cassandra import connect_cassandra, get_session

connect_cassandra()
session = get_session()

job_id = uuid.uuid4()
created_at = datetime.datetime.now(datetime.timezone.utc)

# 1. Insert
session.execute(
    "INSERT INTO outbox_jobs (status, created_at, job_id, topic, conversation_id, payload) "
    "VALUES ('PENDING', %s, %s, 'topic', 'conv', 'payload')",
    (created_at, job_id)
)

rows = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("Step 1 (After Insert PENDING) count:", len(rows))

# 2. Delete LWT
r = session.execute(
    "DELETE FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s IF EXISTS",
    (rows[0].created_at, job_id)
)
print("Step 2 (Delete LWT) applied:", r.one().applied)

# Select immediately after Delete
rows = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("Step 3 (After Delete, Before Batch) count:", len(rows))

# 3. Batch Insert PROCESSING
from cassandra.query import BatchStatement, BatchType
batch = BatchStatement(batch_type=BatchType.LOGGED)
insert_stmt = session.prepare("""
    INSERT INTO outbox_jobs (status, created_at, job_id, topic, conversation_id, payload)
    VALUES ('PROCESSING', ?, ?, ?, ?, ?)
""")
batch.add(insert_stmt, (created_at, job_id, 'topic', 'conv', 'payload'))
session.execute(batch)

# Select PENDING after Batch
rows_pending = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("Step 4 (After Batch) PENDING count:", len(rows_pending))

# Select PROCESSING after Batch
rows_proc = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PROCESSING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("Step 4 (After Batch) PROCESSING count:", len(rows_proc))
