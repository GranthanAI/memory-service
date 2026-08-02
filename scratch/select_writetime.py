from app.db.cassandra import connect_cassandra, get_session

connect_cassandra()
session = get_session()
rows = list(session.execute("SELECT status, WRITETIME(topic), topic FROM outbox_jobs"))
print("Write times in outbox_jobs:")
for r in rows:
    print(r._asdict())
