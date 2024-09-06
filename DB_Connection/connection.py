import psycopg2
import pandas as pd

class PostgresConnection:
    def __init__(self, dbname, user, password, host='localhost', port='5432'):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.cursor = self.conn.cursor()
            print("Connected to PostgreSQL database!")
        except Exception as e:
            print(f"Error: {e}")
            self.conn = None
            self.cursor = None

    def execute_query(self, query):
        if self.cursor is None:
            print("Connection not established. Query cannot be executed.")
            return None
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def close_connection(self):
        if self.conn is not None:
            self.cursor.close()
            self.conn.close()
            print("Connection closed.")
        else:
            print("No connection to close.")

# Example usage:
# Replace dbname, user, password with your actual database credentials
db = PostgresConnection(dbname='telecom', user='postgres', password='ab1234')
db.connect()

# Example query
query = "SELECT * FROM xdr_data"
result = db.execute_query(query)

# Check if the query returned any results
if result is not None:
    # Convert the result to a Pandas DataFrame, if there are rows
    if db.cursor.description:
        df = pd.DataFrame(result, columns=[desc[0] for desc in db.cursor.description])
        print(df.head())  # Display the first few rows of the DataFrame
    else:
        print("No data or query returned no description.")
else:
    print("Query execution failed.")

# Close the connection when done
db.close_connection()
