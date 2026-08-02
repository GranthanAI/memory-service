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
db_created_at = rows[0].created_at

print("Insert created_at type:", type(created_at), repr(created_at))
print("Select created_at type:", type(db_created_at), repr(db_created_at))

# Delete LWT using initial created_at
r1 = session.execute(
    "DELETE FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s IF EXISTS",
    (created_at, job_id)
)
print("Delete with initial created_at applied:", r1.one().applied)

# Check count
rows1 = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("Count after delete with initial:", len(rows1))

# Insert again to test with database created_at
session.execute(
    "INSERT INTO outbox_jobs (status, created_at, job_id, topic, conversation_id, payload) "
    "VALUES ('PENDING', %s, %s, 'topic', 'conv', 'payload')",
    (created_at, job_id)
)

# Delete LWT using database created_at
r2 = session.execute(
    "DELETE FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s IF EXISTS",
    (db_created_at, job_id)
)
print("Delete with db_created_at applied:", r2.one().applied)

# Check count
rows2 = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("Count after delete with db:", len(rows2))
