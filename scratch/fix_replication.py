from app.db.cassandra import connect_cassandra, get_session

connect_cassandra()
session = get_session()
session.execute("ALTER KEYSPACE graphgpt_memory WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}")
print("Replication factor altered successfully to 1!")
