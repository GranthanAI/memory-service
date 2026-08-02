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

# 2. Select to verify it's there
rows = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("After insert, count:", len(rows))
if rows:
    print("Inserted row created_at in python:", rows[0].created_at)

# 3. Delete LWT
r = session.execute(
    "DELETE FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s IF EXISTS",
    (rows[0].created_at if rows else created_at, job_id)
)
print("Delete LWT applied:", r.one().applied)

# 4. Select again
rows_after = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("After delete, count:", len(rows_after))
