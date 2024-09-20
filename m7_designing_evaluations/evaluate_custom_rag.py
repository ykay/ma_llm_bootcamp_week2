from langfuse import Langfuse
import openai
import json
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader("data").load_data()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

langfuse = Langfuse()

# we use a very simple eval here, you can use any eval library
# see https://langfuse.com/docs/scores/model-based-evals for details
def llm_evaluation(output, expected_output):
    client = openai.OpenAI()
    
    prompt = f"""
    Compare the following output with the expected output and evaluate its accuracy:
    
    Output: {output}
    Expected Output: {expected_output}
    
    Provide a score ranging between 0.0 (inaccurate) and 1.0 (accurate) and a brief reason for your evaluation. Do not nitpick about minor differences in punctuation or formatting and focus on the overall correctness of the response.
    Return your response in the following JSON format:
    {{
        "score": 0.0-1.0,
        "reason": "Your explanation here"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the accuracy of responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    evaluation = response.choices[0].message.content
    print(f"Evaluation: {evaluation}")
    try:
      result = json.loads(evaluation)  # Convert the JSON string to a Python dictionary
    except json.JSONDecodeError:
      print("Invalid JSON string")
      result["score"] = 0.0
      result["reason"] = "<Invalid Evaluation JSON string>"
    
    # Debug printout
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    print(f"Evaluation Result: {result}")
    
    return result["score"], result["reason"]

def query_openai(input):
  client = openai.OpenAI()

  # Create a retriever to fetch relevant documents
  retriever = index.as_retriever(retrieval_mode='similarity', k=3)

  # Retrieve relevant documents
  relevant_docs = retriever.retrieve(input)

  context = ""
  context += f"Number of relevant snippets: {len(relevant_docs)}"
  context += "\n" + "="*50 + "\n"
  for i, doc in enumerate(relevant_docs):
      context += f"Snippet {i+1}:\n"
      context += f"Text sample: {doc.node.get_content()[:500]}...\n"  # First 500 characters
      context += f"Source: {doc.node.metadata['file_name']}\n"
      context += f"Score: {doc.score}\n"
      context += "\n" + "="*50 + "\n"
    
  prompt = f"""
  Question: {input}
  Context: {context}
  """

  print("Prompt: ", prompt)
  
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {"role": "system", "content": "You answer questions about details that were discussed in a video using the provided relevant snippets from video transcripts."},
      {"role": "user", "content": prompt}
    ],
    temperature=0.2
  )
  
  return response.choices[0].message.content

from datetime import datetime
 
def rag_query(input):
  
  generationStartTime = datetime.now()

  output = query_openai(input)

  print(output)
 
  langfuse_generation = langfuse.generation(
    name="video-analysis-qa",
    input=input,
    output=output,
    model="gpt-3.5-turbo",
    start_time=generationStartTime,
    end_time=datetime.now()
  )
 
  return output, langfuse_generation

def run_experiment(experiment_name):
  dataset = langfuse.get_dataset("video_analysis_qa_pairs")
 
  for item in dataset.items:
    completion, langfuse_generation = rag_query(item.input)
 
    item.link(langfuse_generation, experiment_name) # pass the observation/generation object or the id
 
    score, reason = llm_evaluation(completion, item.expected_output)
    langfuse_generation.score(
      name="accuracy",
      value=score,
      comment=reason
    )

run_experiment("Experimenting w/ CUSTOM RAG using Video Transcripts #1")