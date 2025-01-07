from openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import psycopg2
from typing import Optional, Dict, List
import os
import re

load_dotenv()

class DatabaseQueryGenerator:
    def __init__(
        self,
        db_host: str,
        db_name: str,
        db_user: str,
        db_password: str,
        db_port: int
    ):
        # Initialize instance variables
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_port = db_port

        # Initialize OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        self.db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Test connection before proceeding
        self.test_connection()
        
        self.db = SQLDatabase.from_uri(self.db_url)
        self.schema = self.read_schema_from_file(os.getenv("SCHEMA_PATH"))

        # Enhanced prompt template
        self.query_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template="""
            ### Instructions:
            Your task is to convert a question into a SQL query, given a Postgres database schema.
            Adhere to these rules:
            - **Deliberately go through the question and database schema word by word** to appropriately answer the question
            - **ALWAYS** respond with **ONLY** the SQL query with no further elaboration, explanation, comments, or justification.
            - Your response should always begin with "SELECT", do NOT write "sql" as the first word in your response.
            - **ALWAYS** ensure that your reponse follows standard PostgreSQL syntax.

            ### Input:
            Generate a SQL query that answers the question `{question}`.
            This query will run on a database whose schema is represented in this string:
            
            {schema}

            ### Response:
            Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
    """
    )


    def read_schema_from_file(self, file_path: str) -> str:
        """Read the schema from a SQL file."""
        try:
            with open(file_path, 'r') as file:
                schema_content = file.read()
            return schema_content
        except FileNotFoundError:
            raise Exception(f"Schema file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading schema file: {str(e)}")

    def test_connection(self) -> None:
        """Test database connection and raise informative errors"""
        try:
            conn = psycopg2.connect(
                dbname=self.db_url.split('/')[-1],
                user=self.db_user,
                password=self.db_url.split(':')[2].split('@')[0],
                host=self.db_url.split('@')[1].split(':')[0],
                port=self.db_url.split(':')[-1].split('/')[0]
            )
            conn.close()
        except psycopg2.OperationalError as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def generate_query(self, question: str) -> str:
        """Generate SQL query based on the question"""
        response = self.client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": self.query_prompt.format(
                        schema=self.schema,
                        question=question
                    )
                }
            ],
            temperature=0.2
        )

        response_content = response.choices[0].message.content
        response_content = re.sub(r'<.*?>', '', response_content)
        response_content = response_content.strip().strip('`').strip()
        response_content = response_content.replace('\u200b', '')
        return response_content

    def execute_query(self, query: str) -> List[Dict]:
        """Execute the generated query and return results"""
        try:
            with psycopg2.connect(
                dbname=self.db_url.split('/')[-1],
                user=self.db_user,
                password=self.db_url.split(':')[2].split('@')[0],
                host=self.db_url.split('@')[1].split(':')[0],
                port=self.db_url.split(':')[-1].split('/')[0]
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    results = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in results]
                    
        except psycopg2.Error as e:
            return [{"error": f"Database error: {str(e)}"}]
        except Exception as e:
            return [{"error": f"Unexpected error: {str(e)}"}]

    def query_database(self, question: str) -> Dict:
        """Generate and execute query, then return results"""
        try:
            generated_query = self.generate_query(question)
            results = self.execute_query(generated_query)
            return {
                "query": generated_query,
                "results": results
            }
        except Exception as e:
            return {
                "query": None,
                "results": [{"error": f"Query generation error: {str(e)}"}]
            }

def main():
    try:
        # Load and validate environment variables
        required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_PORT', 'OPENROUTER_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        generator = DatabaseQueryGenerator(
            db_host=os.getenv('DB_HOST'),
            db_name=os.getenv('DB_NAME'),
            db_user=os.getenv('DB_USER'),
            db_password=os.getenv('DB_PASSWORD'),
            db_port=int(os.getenv('DB_PORT', 5432))
        )

        questions = [
            "How many houses for sale in minneapolis right now?",
        ]

        for question in questions:
            print(f"\nQuestion: {question}")
            result = generator.query_database(question)
            print("\nGenerated Query:")
            print(result["query"])
            print("\nResults:")
            for row in result["results"]:
                print(row)

    except ValueError as e:
        print(f"Configuration error: {str(e)}")
    except ConnectionError as e:
        print(f"Database connection error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()