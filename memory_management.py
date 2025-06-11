import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3
import chromadb.utils.embedding_functions as embedding_functions
import chromadb



class G_VDB:
    def __init__(self, api_key:str, directory:str = "./memo"):
        """
        Initialize the vector database using ChromaDB and Google Generative AI embeddings.

        Args:
            api_key (str): API key for Google Generative AI embedding function.
            directory (str): Directory path to persist ChromaDB storage.
        """
        self.client = chromadb.PersistentClient(path=directory)
        self.google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key) 

        self.collection = self.client.get_or_create_collection(
            name="entha_collection", 
            embedding_function=self.google_ef)
        
    def document_insertion_chroma(self,docs: list[str]) -> None:
        """
        Insert a list of documents into the ChromaDB collection after clearing existing documents.

        Args:
            docs (list[str]): A list of string documents to insert.
        """
        self.clear_collection()
        len_doc = len(docs)
        ids = [f"doc_{i}" for i in range(1,len_doc+1)]
        self.collection.add(
                documents=docs,
                ids=ids,
            )
    def clear_collection(self) -> None:
        """
        Delete all documents currently in the collection.
        """
        results = self.collection.get()
        all_ids = results['ids']

        if all_ids:
            self.collection.delete(ids=all_ids)



