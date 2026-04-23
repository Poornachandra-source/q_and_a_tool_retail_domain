from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableLambda
import os
import pdb
import streamlit as st
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma     #Chroma database 
from sentence_transformers import SentenceTransformer    #for converting text to vectors hugging face transformers 
from langchain_core.example_selectors import SemanticSimilarityExampleSelector 
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

sys.stdout.reconfigure(encoding='utf-8')

def get_few_shot_db_chain(db):

    #llm initialization 
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    task = "text-generation"
    hf_pipeline = pipeline(task, model=model_name, max_new_tokens = 500)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # few shot learning 
    few_shots = [
        {
            'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?",
            'SQLQuery': "SELECT SUM(a.total_amount *((100-COALESCE(d.pct_discount,0))/100)) AS revenue \
                        FROM ( \
                        SELECT SUM(price*stock_quantity) as total_amount,t_shirt_id from t_shirts \
                        where brand  = 'Levi' \
                        group by t_shirt_id) \
                        a left join discounts d on a.t_shirt_id = d.t_shirt_id;",
            'SQLResult': "Result of the SQL Query",
            'Answer': db.run("SELECT SUM(a.total_amount *((100-COALESCE(d.pct_discount,0))/100)) AS revenue \
                            FROM ( \
                            SELECT SUM(price*stock_quantity) as total_amount,t_shirt_id from t_shirts \
                            where brand  = 'Levi' \
                            group by t_shirt_id) \
                            a left join discounts d on a.t_shirt_id = d.t_shirt_id;").strip("[,()]").strip('Decimal').strip("'[,()]'")
        },
        {
            'Question': "How many t-shirts do we have left for Nike in XS size and white color?",
            'SQLQuery': "SELECT SUM(stock_quantity) \
                        FROM t_shirts \
                        WHERE brand = 'Nike' AND size = 'XS' AND color = 'White';",
            'SQLResult': "Result of the SQL Query",
            'Answer': db.run("SELECT SUM(stock_quantity) FROM t_shirts \
            WHERE brand = 'Nike' AND size = 'XS' AND color = 'White';").strip("[,()]").strip('Decimal').strip("'[,()]'")
        },
        {
            'Question': "How much is the total price of the inventory for all S-size t-shirts?",
            'SQLQuery': "SELECT SUM(price*stock_quantity) as total_price \
                        FROM t_shirts \
                        WHERE size = 'S';",
            'SQLResult': "Result of the SQL Query",
            'Answer': db.run("SELECT SUM(price*stock_quantity) as total_price \
                            FROM t_shirts  \
                            WHERE size = 'S';").strip("[,()]").strip('Decimal').strip("'[,()]'")
        },
        {
            'Question': "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?”",
            'SQLQuery': "SELECT SUM(price*stock_quantity) as revenue \
                        FROM t_shirts \
                        WHERE brand = 'Levi';",
            'SQLResult': "Result of the SQL Query",
            'Answer': db.run("SELECT SUM(price*stock_quantity) as revenue \
                            FROM t_shirts \
                            WHERE brand = 'Levi';").strip("[,()]").strip('Decimal').strip("'[,()]'")
        },
        {
            'Question': "How many white color Levi's shirt I have?",
            'SQLQuery': "SELECT SUM(stock_quantity) \
                        FROM t_shirts \
                        WHERE brand = 'Levi' AND color = 'White';",
            'SQLResult': "Result of the SQL Query",
            'Answer': db.run("SELECT SUM(stock_quantity) \
                            FROM t_shirts \
                            WHERE brand = 'Levi' AND color = 'White';").strip("[,()]").strip('Decimal').strip("'[,()]'")
        },
        {
            'Question': "how much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?",
            'SQLQuery': "SELECT SUM(a.total_amount *((100-COALESCE(d.pct_discount,0))/100)) AS sales_amount \
                        FROM ( \
                        SELECT SUM(price*stock_quantity) as total_amount,t_shirt_id from t_shirts \
                        where brand  = 'Nike' and size = 'L' \
                        group by t_shirt_id) \
                        a left join discounts d on a.t_shirt_id = d.t_shirt_id;",
            'SQLResult': "Result of the SQL Query",
            'Answer': db.run("SELECT SUM(a.total_amount *((100-COALESCE(d.pct_discount,0))/100)) AS sales_amount \
                            FROM ( \
                            SELECT SUM(price*stock_quantity) as total_amount,t_shirt_id from t_shirts \
                            where brand  = 'Nike' and size = 'L' \
                            group by t_shirt_id) \
                            a left join discounts d on a.t_shirt_id = d.t_shirt_id;").strip("[,()]").strip('Decimal').strip("'[,()]'")
        }    
    ]

    #Embedding of few shots data 
    embeddings = HuggingFaceEmbeddings(
            model_name= "sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},                        # e.g., "cuda" if you have GPU
        )

    #Storing of vectors 
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize,embedding=embeddings,metadatas=few_shots)

    #Semantic search of user question 
    selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore) # here k refers to no of similar items  ,k=2

    #SQL_Prompt
    SQL_Prompt_prefix = r"""
                    You are a MySQL Expert.
                    Generate a valid SQL query only .

                    Return ONLY a valid MySQL query.
                    DO NOT include anything else in the prompt other than what is given .
                    DO NOT include explanations.
                    DO NOT include markdown.
                    DO NOT include labels.
                    DO NOT include text other than SQL.
                    Unless the user specifies in the question a specific number of examples to obtain , query for at most 
                    {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the 
                    database .

                    Never query for all columns from a table .You must query only the columns that are needed for the question. Pay attention 
                    to use only the column names you can see in the tables below .Be careful to not query for columns that do not exist .
                    Also pay attention to which column is in which table .Pay attention to use CURDATE() function to get the current date , if the question involves 
                    "today".

                    Use the following format :

                    Question : Question here 
                    SQLQuery : SQL Query to run 
                    SQLResult: Result of the SQL Query 
                    Answer : Final answer here             
                """
    SQL_Prompt_suffix =r"""
                    
                    Only use the following tables:
                    {table_info}

                    Question:
                    {input}
                    
                    Output the SQL Query only inside the following block:\n
                                        ```Query\n
                                        <your only SQL Query here>\n
                                        ```\n
                    """
                
    example_prompt = PromptTemplate(
        input_variables=["Question","SQLQuery","SQLResult","Answer"],
        template= "\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer:{Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector = selector,   # binding LLM with vector database 
        example_prompt = example_prompt,
        prefix = SQL_Prompt_prefix,   # these 3 together will form a single prompt 
        suffix = SQL_Prompt_suffix,
        input_variables = ["input","table_info","top_k"]
    )

    # sql_generator = RunnableLambda(lambda x : SQL_Prompt.format(**x) )|llm
    sql_generator = RunnableLambda(lambda x : few_shot_prompt.format(**x) )|llm

    return sql_generator
    

# e= embeddings.embed_query("How many white color Levi's shirt I have?")
# print(e[:5])



# print(selector.select_examples({"Question":"How many Adidas Tshirts I have left in my store?"})) to print selected examples from few shot for a question 
# pdb.set_trace()
# print(clean_sql)
# print(db.table_info)
def database_creation():
    #database parameters
    #db_user = "root" for local run 
    db_user = "user" #for docker run 
    #db_password = "12345678" for local run 
    db_password = "password"
    #db_host = "localhost"  for local run 
    db_host = "db" # for docker run 
    db_name = "atliq_tshirts"

    #creating database object to connect to sql data base 
    #db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info = 3) for local run 
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:3306/{db_name}",sample_rows_in_table_info = 3)

    return db

if __name__ == "__main__":
    db = database_creation()
    sql_generator = get_few_shot_db_chain(db)
    query = sql_generator.invoke({"input":"If we have to sell all the Nike T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?",
                                "table_info":db.table_info,
                                "top_k": "2"})
    clean_sql = query.split("```Query")[-1].split("```")[0].strip()
    try:
        ans = db.run(clean_sql)
    except:
        pdb.set_trace()
    print(ans.strip("[,()]").strip("Decimal(").strip("'"),clean_sql)