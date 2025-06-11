from memory_management import G_VDB
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

api_key_1 = st.secrets["api_key_1"]
api_key_2 = st.secrets["api_key_2"]


class GV_RAG:
    def __init__(self):
        """
        Initialize the GV_RAG system with a vector database and a generative AI client.
        """
        self.vector_db = G_VDB(api_key = api_key_1)
        self.client = genai.Client(api_key=api_key_2)
        
    def document_preprocessing_and_insertion(self, doc:str) -> None:
        """
        Splits a document into overlapping chunks and inserts them into the vector database.

        Args:
            doc (str): The raw text document to be processed and stored.
        """
        doc_splitter = RecursiveCharacterTextSplitter(
                     chunk_size=300,
                     chunk_overlap=30)
        chunks = doc_splitter.split_text(doc)
        self.vector_db.document_insertion_chroma(chunks)
    
    def prompt_creation(self,doc: set[str],quest:str) -> str:
        """
        Creates a structured prompt for the language model using retrieved context and a question.

        Args:
            doc (set[str]): Set of relevant document chunks.
            quest (str): The user's question.

        Returns:
            str: A formatted prompt string.
        """
        context = "\n\n---\n\n".join(doc)
        prompt = (
            "You are a knowledgeable assistant. Use the provided context to answer the question "
            "accurately and concisely. If the answer is not found in the context, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{quest}\n\n"
            "Answer:"
        )
        return prompt
    
    def LLM(self,quest:str) -> str:
        """
        Uses a large language model to answer a question based on retrieved context.

        Args:
            quest (str): The user's question.

        Returns:
            str: The generated answer from the language model.
        """
        document = self.retrive_top_n(quest)
        prompt = self.prompt_creation(document,quest)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
            
    def retrive_top_n(self,query:str, n:int = 3, k:int = 2) -> set[str]:
        """
        Retrieves top-n documents based on query and further expands context using nearest neighbors.

        Args:
            query (str): The query string to retrieve documents.
            n (int): Number of top documents to initially retrieve.
            k (int): Number of neighbor documents to expand context for each top result.

        Returns:
            set[str]: A set of unique document chunks relevant to the query.
        """
        result = self.vector_db.collection.query(
            query_texts=[query],
            n_results=n,
            include=['embeddings',"documents"]
        ) 
        exclude_ids = set(result["ids"][0])
        store = set()

        for embedding in result['embeddings'][0]:
            sub = self.vector_db.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where={"id": {"$nin": list(exclude_ids)}}
            )

            exclude_ids.update(sub["ids"][0])

            for doc in sub["documents"][0]:
                store.add(doc)
            
        return store
    


