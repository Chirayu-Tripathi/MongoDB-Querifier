from sentence_transformers import  CrossEncoder
import google.generativeai as genai
import copy

class QueryGeneration:
    def __init__(self, model):
        self.model = self.get_gemini_model()
        self.encoding_model = model
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')

    def get_gemini_model(self):
        safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}, {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"}]
        model = genai.GenerativeModel('gemini-pro',safety_settings = safety_settings )
        return model

    def generate_query(self,class_name, schema, question, db, prompt, rag = True):
        if rag:
            vector = self.encoding_model.encode(question)
            res = db.get_nearby_objects(class_name, vector, ['db_schema', 'question', 'gemini_mql'], limit = 10)
            # check = copy.deepcopy(res["data"]["Get"][class_name])
            hits = res["data"]["Get"][class_name][:]
            cross_inp = [[question, hit['question']] for hit in hits]
            cross_scores = self.cross_encoder.predict(cross_inp)
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]

            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
            example1 = hits[0]
            example2 = hits[1]

            EXAMPLE1_SCHEMA, EXAMPLE1_QUESTION, EXAMPLE1_QUERY = example1['db_schema'], example1['question'], example1['gemini_mql']
            EXAMPLE2_SCHEMA, EXAMPLE2_QUESTION, EXAMPLE2_QUERY = example2['db_schema'], example2['question'], example2['gemini_mql']

            prompt = prompt.replace("{{SCHEMA}}", schema).replace("{{QUESTION}}", question)
            prompt = prompt.replace("{{EXAMPLE1_SCHEMA}}", EXAMPLE1_SCHEMA).replace("{{EXAMPLE1_QUESTION}}", EXAMPLE1_QUESTION).replace("{{EXAMPLE1_QUERY}}", EXAMPLE1_QUERY)
            prompt = prompt.replace("{{EXAMPLE2_SCHEMA}}", EXAMPLE2_SCHEMA).replace("{{EXAMPLE2_QUESTION}}", EXAMPLE2_QUESTION).replace("{{EXAMPLE2_QUERY}}", EXAMPLE2_QUERY)
            res = res["data"]["Get"][class_name][:]
            res =  [{i:j[i] for i in j if i in ['gemini_mql', 'question', '_additional']} for j in res] # if i in ['gemini_mql', 'question', '_additional']
            hits = [{i:j[i] for i in j if i in ['gemini_mql', 'question', '_additional', 'cross-score']} for j in hits]
        else:
            res = {}
            hits = {}
            prompt = prompt.replace("{{SCHEMA}}", schema).replace("{{QUESTION}}", question)

        return (self.model.generate_content(prompt), res, hits)