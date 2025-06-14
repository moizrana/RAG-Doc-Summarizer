"""
Document Summarization using Retrieval-Augmented Generation (RAG)
Main CLI interface for the RAG-based document summarization system.
"""

import argparse
import logging

from rag_pipeline import RAGSummarizer, ResultPresenter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="RAG Document Summarization System")
    parser.add_argument("document", help="Path to document to summarize")
    parser.add_argument("--query", default="Summarize this document", help="Summarization query")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--max-length", type=int, default=250, help="Maximum summary length")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for document splitting")
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--generation-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", help="Generation model name")
    parser.add_argument("--hide-chunks", action="store_true", help="Hide retrieved chunks in output")
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = RAGSummarizer(
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
        chunk_size=args.chunk_size
    )
    
    try:
        # Run summarization
        result = summarizer.summarize_document(
            document_path=args.document,
            query=args.query,
            k=args.k,
            max_summary_length=args.max_length
        )
        
        # Display results
        ResultPresenter.display_results(result, show_chunks=not args.hide_chunks)
        
        # Save results if output path provided
        if args.output:
            ResultPresenter.save_results(result, args.output)
    
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())