"""
app/workers/outbox_worker.py

Outbox worker daemon for reliable task publishing to Kafka.
Uses Cassandra LWT to claim PENDING jobs, publishes to Kafka, and deletes completed rows.
"""
