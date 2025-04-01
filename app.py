from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime

from langchain_ollama import ChatOllama  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import psycopg2
from psycopg2.extras import RealDictCursor
from langgraph.graph import StateGraph, END


llm = ChatOllama(model="llama3.2", temperature=0) 


class AgentState(TypedDict):
    query: str
    sql: Optional[str]
    result: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    error: Optional[str]


DB_SCHEMA = """
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    country VARCHAR(50),
    signup_date DATE
);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2)
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    order_date DATE,
    status VARCHAR(20)
);

CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id),
    product_id INT REFERENCES products(product_id),
    quantity INT
);

CREATE TABLE payments (
    payment_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id),
    amount DECIMAL(10,2),
    payment_date DATE,
    status VARCHAR(20)
);
"""

def create_sql_agent():
    system_prompt = f"""You are an expert SQL query generator. Your task is to convert natural language queries into PostgreSQL SQL statements.
    
    Here is the database schema:
    {DB_SCHEMA}
    
    The user will provide a natural language query. Convert it into a valid PostgreSQL SQL query.
    - Be precise and ensure your SQL query addresses the actual intent of the question
    - Include appropriate JOINs when working with multiple tables
    - Ensure proper column references and table aliases
    - Do not include any explanations, just return the SQL query
    - Ensure the SQL is valid for PostgreSQL
    
    IMPORTANT: Return ONLY the SQL query with no additional text, comments, or explanations.
    """
    
    def sql_generator(state):
        natural_language_query = state["query"]
        
        # Create prompt using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Convert this natural language query to SQL: {query}")
        ])
        
        try:
           
            chain = prompt | llm | StrOutputParser()

            sql_query = chain.invoke({"query": natural_language_query})
            
 
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            print(f"Generated SQL: {sql_query}")
            

            return {"sql": sql_query}
        except Exception as e:
            print(f"SQL Generation Error: {str(e)}")
            return {"error": f"SQL Generation Error: {str(e)}"}
    
    return sql_generator


def create_query_executor():

    db_params = {
        "host": "",
        "database": "",
        "user": "",
        "password": "",
        "port": 5432
    }
    
    def execute_query(state):
        sql_query = state.get("sql")
        if not sql_query:
            return {"error": "No SQL query was generated"}
        
        try:
            conn = psycopg2.connect(**db_params, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            return {"result": str(result).replace("{","")}
                
        except Exception as e:
            print(f"Query Execution Error: {str(e)}")
            return {"error": f"Query Execution Error: {str(e)}"}
    
    return execute_query


def create_summarizer():
    system_prompt = """You are an expert data analyst. Your task is to summarize SQL query results into clear, 
    concise language that a non-technical person can understand.
    
    Provide a brief, focused summary that captures the key insights from the data. 
    Highlight important patterns, notable outliers, or significant metrics.
    Keep your summary short and to the point.
    """
    
    def summarize_results(state):
        query_results = state.get("result", [])
        original_query = state.get("query", "")
        
        if not query_results:
            return {"summary": "No results were returned from the query."}
        

        results_str = str(query_results)
        

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Original natural language query: \"{query}\"\n\nQuery results: {results}\n\nPlease provide a concise summary of these results:")
        ])
        
        try:

            chain = prompt | llm | StrOutputParser()
            summary = chain.invoke({
                "query": original_query,
                "results": results_str
            })
            
            return {"summary": summary.strip()}
        except Exception as e:
            print(f"Summarization Error: {str(e)}")
            return {"error": f"Summarization Error: {str(e)}"}
    
    return summarize_results


def create_db_query_graph():

    sql_generator = create_sql_agent()
    query_executor = create_query_executor()
    summarizer = create_summarizer()
    

    workflow = StateGraph(AgentState)
    

    workflow.add_node("sql_generator", sql_generator)
    workflow.add_node("query_executor", query_executor)
    workflow.add_node("summarizer", summarizer)

    workflow.set_entry_point("sql_generator")

    workflow.add_edge("sql_generator", "query_executor")
    workflow.add_edge("query_executor", "summarizer")
    workflow.add_edge("summarizer", END)
    
    return workflow.compile()


def process_query(query: str):
    graph = create_db_query_graph()
    

    initial_state = {"query": query, "sql": None, "result": None, "summary": None, "error": None}
    

    result = graph.invoke(initial_state)
    

    if result.get("error"):
        return {"status": "error", "message": result["error"]}
    
    return {
        "status": "success",
        "query": result["query"],
        "sql": result["sql"],
        "result": result["result"],
        "summary": result["summary"]
    }


if __name__ == "__main__":

    test_queries = [
        "How many customers do we have from USA? Give me all names along with count.",
        "List all customers",
        "Which customer has ordered maximum no. of times?"
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        print(f"Processing query: '{query}'")
        print("="*80)
        
        result = process_query(query)
        
        if result["status"] == "error":
            print(f"Error: {result['message']}")
        else:
            print(f"SQL: {result['sql']}")
            print("\nResults:")
            print(result["result"])
            
            print(f"\nSummary: {result['summary']}")