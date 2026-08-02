from app.db.cassandra import connect_cassandra, get_session

connect_cassandra()
session = get_session()
rows = list(session.execute("SELECT * FROM outbox_jobs"))
print("All rows in outbox_jobs:")
for r in rows:
    print(r._asdict())
