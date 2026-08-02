import uuid
import datetime
from app.db.cassandra import connect_cassandra, get_session

connect_cassandra()
session = get_session()

# Let's check with non-matching created_at
job_id = uuid.UUID('508edfc4-ee27-4e73-bdf3-2911dbdbb641')
wrong_time = datetime.datetime(2026, 8, 2, 8, 46, 20, 999000, tzinfo=datetime.timezone.utc)

r = session.execute(
    "DELETE FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s IF EXISTS",
    (wrong_time, job_id)
)
print("Result with wrong time:", r.one())
