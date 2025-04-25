import pandas as pd
from db_utils import get_db_connection


def load_parquet_to_postgres(parquet_file_path, table_name):
    """
    Reads a .parquet file and writes its content to a PostgreSQL table.

    Args:
        parquet_file_path (str): Path to the .parquet file.
        table_name (str): The name of the destination PostgreSQL table.
    """
    # Step 1: Read the .parquet file into a Pandas DataFrame
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Loaded {len(df)} rows from {parquet_file_path}")
    except Exception as e:
        print(f"Error reading .parquet file: {e}")
        return

    # Step 2: Write DataFrame to the PostgreSQL database
    try:
        # Get the SQLAlchemy engine
        engine = get_db_connection()

        # Use df.to_sql() to send the dataframe to PostgreSQL
        df.to_sql(table_name, con=engine, index=False, if_exists='replace', method='multi')
        print(f"Data successfully written to table: {table_name}")


    except Exception as e:
        print(f"Error writing to PostgreSQL table: {e}")

    finally:
        engine.dispose()

if __name__ == '__main__':
    parquet_file_path = 'recipes_dataset_loaded/recipes.parquet'
    table_name = 'recipes_dataset'
    load_parquet_to_postgres(parquet_file_path, table_name)