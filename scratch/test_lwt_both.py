import uuid
import datetime
from app.db.cassandra import connect_cassandra, get_session

connect_cassandra()
session = get_session()

job_id = uuid.uuid4()
created_at = datetime.datetime.now(datetime.timezone.utc)

# 1. Insert using LWT (IF NOT EXISTS)
r_ins = session.execute(
    "INSERT INTO outbox_jobs (status, created_at, job_id, topic, conversation_id, payload) "
    "VALUES ('PENDING', %s, %s, 'topic', 'conv', 'payload') IF NOT EXISTS",
    (created_at, job_id)
)
print("Insert LWT applied:", r_ins.one().applied)

# 2. Select to verify
rows = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("After insert count:", len(rows))

# 3. Delete LWT
r_del = session.execute(
    "DELETE FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s IF EXISTS",
    (created_at, job_id)
)
print("Delete LWT applied:", r_del.one().applied)

# 4. Select to verify deletion
rows_after = list(session.execute("SELECT * FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s", (created_at, job_id)))
print("After delete count:", len(rows_after))
