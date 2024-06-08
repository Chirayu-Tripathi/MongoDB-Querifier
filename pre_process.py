import pandas as pd
from sentence_transformers import SentenceTransformer

class DataProcessor:
    def __init__(self, csv_path, txt_path):
        """
        Initialize the DataProcessor with paths to the CSV and TXT files.

        :param csv_path: Path to the CSV file containing schema information.
        :param txt_path: Path to the TXT file containing question-query pairs.
        """
        self.csv_path = csv_path
        self.txt_path = txt_path
        self.schema_append_df = None  # DataFrame to store schema data
        self.df_append = None  # DataFrame to store processed data
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')  # Load pre-trained sentence transformer model

    def read_data(self):
        """
        Read data from CSV and TXT files, process the TXT data into a structured format.
        """
        # Read schema data from CSV
        self.schema_append_df = pd.read_csv(self.csv_path)

        # Read and process data from TXT
        with open(self.txt_path, 'r') as file:
            data = file.read().split('\n')[3:-1]  # Split by newline, skip header and footer

        # Process each line into a list of items
        rows = [row.split('|') for row in data]
        
        # Clean each item in the rows
        rows = [[item.strip().replace('```', '').replace('`', '') for item in row if item != ''] for row in rows]
        
        # Filter rows with more than 3 items (assuming a valid row has 3 items)
        rows = [row for row in rows if len(row) <= 3]

        # Print rows with more than 3 items for debugging
        for row in rows:
            if len(row) > 3:
                print(row[0], '-------', row[1], '------', row[2], '-------', row[3])

        # Create DataFrame from processed rows
        self.df_append = pd.DataFrame(rows, columns=['question', 'gemini_mql', 'db_id'])

    def clean_data(self):
        """
        Clean the data by removing empty rows, merging with schema data, and cleaning string values.
        """
        # Drop rows where all columns are NaN
        self.df_append = self.df_append.dropna(how='all')

        # Merge with schema data based on 'db_id'
        self.df_append = pd.merge(self.schema_append_df, self.df_append, on='db_id')

        # Clean 'db_schema' and 'gemini_mql' columns by removing newlines and extra spaces
        self.df_append['db_schema'] = self.df_append['db_schema'].apply(lambda x: x.replace("\n", "")).apply(lambda x: x.replace("  ", ""))
        self.df_append['gemini_mql'] = self.df_append['gemini_mql'].apply(lambda x: x.replace("\n", "")).apply(lambda x: x.replace("  ", ""))
        self.df_append['gemini_mql'] = self.df_append['gemini_mql'].apply(lambda x: x.replace("```", "")).apply(lambda x: x.replace("  ", ""))

    def generate_embeddings(self):
        """
        Generate sentence embeddings for each question using the pre-trained model.
        """
        self.df_append['vector'] = self.df_append['question'].apply(lambda x: self.model.encode(x))

    def process(self):
        """
        Process the data by reading, cleaning, and generating embeddings.

        :return: Processed DataFrame
        """
        self.read_data()
        self.clean_data()
        self.generate_embeddings()
        return self.df_append


if __name__ == "__main__":
    # Paths to input files
    csv_path = 'mongodb_array_object.csv'
    txt_path = 'mongodb_array_object.txt'

    # Create DataProcessor instance and process data
    processor = DataProcessor(csv_path, txt_path)
    processed_df = processor.process()
    
    # Print the processed DataFrame
    print(processed_df)