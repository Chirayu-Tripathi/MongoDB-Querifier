from sentence_transformers import CrossEncoder
import google.generativeai as genai


class QueryGeneration:
    def __init__(self, model):
        # Initialize the QueryGeneration class with a sentence transformer model and a prompt
        self.model = self.get_gemini_model()  # Get the Gemini model
        self.encoding_model = model  # Sentence transformer model for encoding questions
        # self.prompt = prompt  # Prompt template for query generation
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')  # Cross-encoder for re-ranking

    def get_gemini_model(self):
        # Configure and return a Gemini Pro model with safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"}
        ]
        model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
        return model

    def generate_query(self, class_name, schema, question, db, prompt, rag=True):
        # Generate a query based on the given class, schema, question, and database
        # rag: boolean to enable/disable Retrieval-Augmented Generation

        if rag:
            # Encode the question using the sentence transformer model
            vector = self.encoding_model.encode(question)
            
            # Retrieve nearby objects from the database using the question vector
            res = db.get_nearby_objects(class_name, vector, ['db_schema', 'question', 'gemini_mql'], limit=10)
            hits = res["data"]["Get"][class_name]
            
            # Prepare inputs for cross-encoder (question pairs)
            cross_inp = [[question, hit['question']] for hit in hits]
            
            # Get cross-encoder scores for re-ranking
            cross_scores = self.cross_encoder.predict(cross_inp)
            
            # Add cross-encoder scores to hits
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]

            # Sort hits by cross-encoder scores in descending order
            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
            
            # Get the top two hits as examples
            example1 = hits[0]
            example2 = hits[1]

            # Extract schema, question, and query from the examples
            EXAMPLE1_SCHEMA, EXAMPLE1_QUESTION, EXAMPLE1_QUERY = example1['db_schema'], example1['question'], example1['gemini_mql']
            EXAMPLE2_SCHEMA, EXAMPLE2_QUESTION, EXAMPLE2_QUERY = example2['db_schema'], example2['question'], example2['gemini_mql']

            # Replace placeholders in the prompt with actual values
            prompt = prompt.replace("{{SCHEMA}}", schema).replace("{{QUESTION}}", question)
            prompt = prompt.replace("{{EXAMPLE1_SCHEMA}}", EXAMPLE1_SCHEMA).replace("{{EXAMPLE1_QUESTION}}", EXAMPLE1_QUESTION).replace("{{EXAMPLE1_QUERY}}", EXAMPLE1_QUERY)
            prompt = prompt.replace("{{EXAMPLE2_SCHEMA}}", EXAMPLE2_SCHEMA).replace("{{EXAMPLE2_QUESTION}}", EXAMPLE2_QUESTION).replace("{{EXAMPLE2_QUERY}}", EXAMPLE2_QUERY)

        else:
            # If RAG is disabled, just replace schema and question in the prompt
            prompt = prompt.replace("{{SCHEMA}}", schema).replace("{{QUESTION}}", question)

        # Generate content using the Gemini model with the constructed prompt
        return self.model.generate_content(prompt)