import weaviate

class WeaviateClient:
    def __init__(self):
        self.client = weaviate.Client(
            embedded_options=weaviate.embedded.EmbeddedOptions(),
        )

    def create_class(self, class_name, properties):
        if self.client.schema.exists(class_name):
            print(f"Class {class_name} already exists.")
            return
        else:
            print(f"Creating class {class_name}...")
            class_obj = {
                "class": class_name,
                "properties": properties
            }
            new_class = self.client.schema.create_class(class_obj)
        # return new_class

    def add_data_object(self, class_name, df):
        columns = df.columns.tolist()
        vectors = df['vector'].tolist()
        for index, row in df.iterrows():
            data_object = {columns[i] : row[columns[i]] for i in range(len(columns)) if columns[i] != 'vector'}
            # if not self.client.collections.exists(data_object):
            self.client.batch.add_data_object(data_object, class_name, vector=vectors[index])
        self.client.batch.create_objects()

    def get_nearby_objects(self, class_name, vector, retrieval_columns, limit=10):
        near_vec = {"vector": vector}
        res = self.client \
            .query.get(class_name, retrieval_columns + ["_additional {certainty}"]) \
            .with_near_vector(near_vec) \
            .with_limit(limit) \
            .do()
        return res