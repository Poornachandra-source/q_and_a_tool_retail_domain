from langchain_helper import database_creation, few_shot_db,get_few_shot_db_chain
import sys
import streamlit as st
import pdb
import re
from langchain_core.callbacks import BaseCallbackHandler

sys.stdout.reconfigure(encoding='utf-8')

class VerboseHandler(BaseCallbackHandler):

    def on_chain_start(self, serialized, inputs, **kwargs):
        print("\n🔹 Chain Start")
        print("Inputs:", inputs)

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n🧠 Prompt sent to LLM:")
        for p in prompts:
            print(p)
    
    def on_llm_end(self, response, **kwargs):
        print("\n✅ LLM Response:")
        print(response)
    
    def on_chain_end(self, outputs, **kwargs):
        print("\n🔹 Chain End")
        print("Outputs:", outputs)

handler = VerboseHandler()  # to see background logs similar to verbose , own callback handler due to langchain version issues 

db = database_creation()


st.title("AtliQ T shirts: Q&A Database 👕")
question = st.text_input("Question: ")
if question:
    llm,selector = few_shot_db(db)
    sql_generator = get_few_shot_db_chain(llm,selector)
    # query = ''

    query = sql_generator.invoke({"input":question,
                                "table_info":db.table_info,
                                "top_k": "2"},
                                config={"callbacks": [handler]})
    clean_sql = query.split("```Query")[-1].split("```")[0].strip()
    # try:
    ans = db.run(clean_sql)
        # pdb.set_trace()
    # except:
    #     pdb.set_trace()
    st.header("Answer: ")
    st.write(ans.strip("[,()]").strip("Decimal(").strip("'"))    