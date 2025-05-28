import pyarrow.parquet as pq
import pyarrow as pa
from sqlalchemy import create_engine, types
import json
import numpy as np
from db_utils import get_db_connection

def load_parquet_chunked(parquet_file_path, table_name, chunk_size=10000):
    """
    Reads a .parquet file in chunks using PyArrow and writes to PostgreSQL.
    Handles arrays, special characters, and explicit schema for SQL.
    """
    engine = None
    try:
        engine = get_db_connection()
        parquet_file = pq.ParquetFile(parquet_file_path)

        # Визначаємо SQL-схему для складних стовпців (наприклад, масивів)
        dtype = {
            # Приклад: якщо є стовпці з масивами, вказуємо TEXT або JSON
            "Keywords_m1": types.JSON,
            "RecipeIngredientParts_m1": types.JSON,
            "RecipeIngredientQuantities_m1": types.JSON,
            # Додаткові стовпці з дивними символами (наприклад, '1⁄3')
            "RecipeIngredientQuantities_m9998": types.String,
            # Числові стовпці
            "AggregatedRating_m1": types.Float,
            "Calories_m1": types.Float,
            # ... додайте інші стовпці за необхідності
        }

        for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            print(f"Processing chunk {i + 1} with {batch.num_rows} rows")
            chunk = batch.to_pandas()

            # Конвертуємо масиви у JSON (якщо SQL підтримує)
            for col in chunk.columns:
                if isinstance(chunk[col].iloc[0], (list, np.ndarray)):
                    chunk[col] = chunk[col].apply(lambda x: json.dumps(x.tolist() if hasattr(x, 'tolist') else x))

            # Записуємо у PostgreSQL
            if_exists = 'replace' if i == 0 else 'append'
            chunk.to_sql(
                table_name,
                con=engine,
                index=False,
                if_exists=if_exists,
                method='multi',
                dtype=dtype
            )

        print(f"Data successfully written to table: {table_name}")

    except Exception as e:
        print(f"Error processing parquet file: {e}")
        raise

    finally:
        if engine is not None:
            engine.dispose()

if __name__ == '__main__':
    parquet_file_path = 'recipes_dataset_loaded/recipes.parquet'
    table_name = 'recipes_dataset'
    load_parquet_chunked(parquet_file_path, table_name)