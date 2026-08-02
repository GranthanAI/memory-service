import time
from pymilvus import connections, utility, Collection, FieldSchema, DataType, CollectionSchema

connections.connect(host="localhost", port=19530)

coll_name = "test_delete_debug"
if utility.has_collection(coll_name):
    utility.drop_collection(coll_name)

fields = [
    FieldSchema("fact_id", DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema("user_id", DataType.VARCHAR, max_length=64, is_partition_key=True),
    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4)
]
schema = CollectionSchema(fields)
collection = Collection(coll_name, schema)
collection.create_index("vector", {
    "index_type": "FLAT",
    "metric_type": "COSINE"
})
collection.load()

# Insert
collection.insert([
    ["fact-1", "fact-2"],
    ["user-1", "user-1"],
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
])
collection.flush()

print("Count before delete:", collection.num_entities)

# Delete
r = collection.delete("fact_id in ['fact-1']")
print("Delete result:", r)

# Search or Query immediately
res = collection.query("user_id == 'user-1'", output_fields=["fact_id"])
print("Query immediately after delete:", res)

# Flush and search
collection.flush()
res_after_flush = collection.query("user_id == 'user-1'", output_fields=["fact_id"])
print("Query after flush:", res_after_flush)
