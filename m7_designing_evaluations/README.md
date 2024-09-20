# Setup
1. Create an `.env` file and set the values for `OPENAI_API_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_HOST`.
2. Install dependencies `pip install python-dotenv llama_index langfuse`
3. Feel free to replace the files inside the `data/` folder.
4. Run `generate_dataset.py` to create a new dataset that will become available in your Langfuse account.
5. Run `evaluate_rag.py` to run an experiment on the new dataset. This will evaluate how well RAG answers compare to the evaluation dataset.
