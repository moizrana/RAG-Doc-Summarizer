"""
RAG Pipeline and Result Presenter
"""

import time
import json
import logging
from typing import List, Dict, Any

from models import SummaryResult
from core_components import DocumentProcessor, DocumentChunker, VectorStore, SummaryGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSummarizer:
    """Main RAG Summarization Pipeline"""
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        generation_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.doc_processor = DocumentProcessor()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(embedding_model)
        self.generator = SummaryGenerator(generation_model)
        
    def summarize_document(
        self, 
        document_path: str, 
        query: str = "Summarize this document",
        k: int = 5,
        max_summary_length: int = 200
    ) -> SummaryResult:
        """Complete RAG summarization pipeline"""
        
        start_time = time.time()
        
        try:
            # 1. Document Ingestion
            logger.info(f"Loading document: {document_path}")
            text = self.doc_processor.load_document(document_path)
            
            # 2. Document Chunking
            logger.info("Chunking document...")
            chunks = self.chunker.chunk_document(text)
            
            # 3. Build Vector Index
            logger.info("Building vector index...")
            self.vector_store.build_index(chunks)
            
            # 4. Retrieve Relevant Chunks
            logger.info(f"Retrieving top-{k} relevant chunks...")
            retrieved_chunks = self.vector_store.retrieve(query, k)
            
            # 5. Generate Summary
            logger.info("Generating summary...")
            summary, token_usage = self.generator.generate_summary(
                retrieved_chunks, 
                max_summary_length
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # 6. Prepare Result
            result = SummaryResult(
                summary=summary,
                retrieved_chunks=retrieved_chunks,
                token_usage=token_usage,
                latency=latency,
                metadata={
                    'document_path': document_path,
                    'total_chunks': len(chunks),
                    'document_length': len(text),
                    'query': query
                }
            )
            
            logger.info(f"Summarization completed in {latency:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in summarization pipeline: {e}")
            raise


class ResultPresenter:
    """Handles output presentation and visualization"""
    
    @staticmethod
    def display_results(result: SummaryResult, show_chunks: bool = True) -> None:
        """Display summarization results"""
        print("=" * 80)
        print("RAG DOCUMENT SUMMARIZATION RESULTS")
        print("=" * 80)
        
        print(f"\n📄 Document: {result.metadata.get('document_path', 'Unknown')}")
        print(f"🔍 Query: {result.metadata.get('query', 'Unknown')}")
        print(f"⏱️  Processing Time: {result.latency:.2f} seconds")
        print(f"📊 Token Usage: {result.token_usage.get('total_tokens', 0)} tokens")
        
        print("\n" + "=" * 40)
        print("GENERATED SUMMARY")
        print("=" * 40)
        print(f"\n{result.summary}\n")
        
        if show_chunks:
            print("=" * 40)
            print("RETRIEVED CONTEXT")
            print("=" * 40)
            
            for i, chunk_result in enumerate(result.retrieved_chunks):
                print(f"\n[Rank {chunk_result.rank + 1}] Similarity: {chunk_result.similarity_score:.3f}")
                print(f"Chunk ID: {chunk_result.chunk.chunk_id}")
                print(f"Text: {chunk_result.chunk.text[:200]}...")
                if i < len(result.retrieved_chunks) - 1:
                    print("-" * 40)
        
        print("\n" + "=" * 80)
    
    @staticmethod
    def save_results(result: SummaryResult, output_path: str) -> None:
        """Save results to JSON file"""
        output_data = {
            'summary': result.summary,
            'retrieved_chunks': [
                {
                    'rank': chunk.rank,
                    'similarity_score': chunk.similarity_score,
                    'chunk_id': chunk.chunk.chunk_id,
                    'text': chunk.chunk.text
                }
                for chunk in result.retrieved_chunks
            ],
            'token_usage': result.token_usage,
            'latency': result.latency,
            'metadata': result.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")