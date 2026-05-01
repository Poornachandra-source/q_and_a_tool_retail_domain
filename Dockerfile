FROM python:3.12-slim

WORKDIR /q_and_a_tool_retail_domain

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings;HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]