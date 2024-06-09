import pandas as pd
from sentence_transformers import SentenceTransformer

class DataProcessor:
    def __init__(self, csv_path, txt_path):
        self.csv_path = csv_path
        self.txt_path = txt_path
        self.schema_append_df = None
        self.df_append = None
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    def read_data(self):
        self.schema_append_df = pd.read_csv(self.csv_path)
        with open(self.txt_path, 'r') as file:
            data = file.read().split('\n')[3:-1]  # Skip the header and footer lines

        rows = [row.split('|') for row in data]
        rows = [[row.rstrip().lstrip().replace('```','').replace('`','') for row in data2 if row != ''] for data2 in rows]
        rows = [row for row in rows if len(row) <= 3]
        for row in rows:
            if len(row) > 3:
                print(row[0], '-------', row[1], '------', row[2], '-------', row[3])
        self.df_append = pd.DataFrame(rows, columns=['question', 'gemini_mql', 'db_id'])
    
    def clean_data(self):
        self.df_append = self.df_append.dropna(how='all')
        self.df_append = pd.merge(self.schema_append_df, self.df_append, on='db_id')

        self.df_append['db_schema'] = self.df_append['db_schema'].apply(lambda x: x.replace("\n","")).apply(lambda x: x.replace("  ",""))
        self.df_append['gemini_mql'] = self.df_append['gemini_mql'].apply(lambda x: x.replace("\n","")).apply(lambda x: x.replace("  ",""))
        self.df_append['gemini_mql'] = self.df_append['gemini_mql'].apply(lambda x: x.replace("```","")).apply(lambda x: x.replace("  ",""))

    def generate_embeddings(self):
        self.df_append['vector'] = self.df_append['question'].apply(lambda x: self.model.encode(x))

    def process(self):
        self.read_data()
        self.clean_data()
        self.generate_embeddings()
        return self.df_append

# Usage

if __name__ == "__main__":
    csv_path = '/content/drive/MyDrive/weavite/mongodb_array_object.csv'
    txt_path = '/content/drive/MyDrive/weavite/mongodb_array_object.txt'

    processor = DataProcessor(csv_path, txt_path)
    processed_df = processor.process()
    print(processed_df)