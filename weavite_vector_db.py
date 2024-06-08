import weaviate

class WeaviateClient:
    def __init__(self):
        # Initialize a Weaviate client using the embedded options
        # This means Weaviate will run in-memory without requiring a separate server
        self.client = weaviate.Client(
            embedded_options=weaviate.embedded.EmbeddedOptions(),
        )

    def create_class(self, class_name, properties):
        # Check if the class already exists in the Weaviate schema
        if self.client.schema.exists(class_name):
            print(f"Class {class_name} already exists.")
            return
        else:
            print(f"Creating class {class_name}...")
            
            # Define the class object with the given name and properties
            class_obj = {
                "class": class_name,
                "properties": properties
            }
            
            # Create the class in Weaviate schema
            new_class = self.client.schema.create_class(class_obj)
            
            # Note: The return value 'new_class' is not used (commented out)
            # return new_class

    def add_data_object(self, class_name, df):
        # Extract column names and vectors from the DataFrame
        columns = df.columns.tolist()
        vectors = df['vector'].tolist()
        
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Create a data object dictionary from the row, excluding the 'vector' column
            data_object = {columns[i]: row[columns[i]] for i in range(len(columns)) if columns[i] != 'vector'}
            
            # Add the data object to the batch, along with its vector
            self.client.batch.add_data_object(data_object, class_name, vector=vectors[index])
        
        # Create all objects in the batch
        self.client.batch.create_objects()

    def get_nearby_objects(self, class_name, vector, retrieval_columns, limit=10):
        # Define the query vector
        near_vec = {"vector": vector}
        
        # Build and execute the query:
        # 1. Get objects of the specified class
        # 2. Retrieve specified columns and the certainty score
        # 3. Find objects near the given vector
        # 4. Limit the results
        res = self.client \
            .query.get(class_name, retrieval_columns + ["_additional {certainty}"]) \
            .with_near_vector(near_vec) \
            .with_limit(limit) \
            .do()
        
        # Return the query results
        return res