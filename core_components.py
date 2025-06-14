"""
Core components for document processing, chunking, vector storage, and summary generation
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

import numpy as np
import PyPDF2
import markdown
from sentence_transformers import SentenceTransformer
import faiss
from together import Together

from models import ChunkInfo, RetrievalResult

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document ingestion and preprocessing"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.md', '.markdown'}
    
    def load_document(self, file_path: str) -> str:
        """Load document from various formats"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        try:
            if path.suffix.lower() == '.pdf':
                return self._load_pdf(file_path)
            elif path.suffix.lower() in {'.md', '.markdown'}:
                return self._load_markdown(file_path)
            else:  # .txt
                return self._load_text(file_path)
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise
    
    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _load_markdown(self, file_path: str) -> str:
        """Load and convert markdown to plain text"""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
        # Convert markdown to HTML then extract text
        html = markdown.markdown(md_content)
        # Simple HTML tag removal
        clean_text = re.sub('<.*?>', '', html)
        return clean_text.strip()
    
    def _load_text(self, file_path: str) -> str:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()


class DocumentChunker:
    """Handles document chunking with semantic awareness"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, text: str) -> List[ChunkInfo]:
        """Split document into semantically meaningful chunks"""
        # First, try sentence-based chunking
        sentences = self._split_into_sentences(text)
        chunks = []
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_info = ChunkInfo(
                    text=current_chunk.strip(),
                    start_idx=current_start,
                    end_idx=current_start + len(current_chunk),
                    chunk_id=chunk_id,
                    metadata={'sentence_count': len(current_chunk.split('.'))}
                )
                chunks.append(chunk_info)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text)
                chunk_id += 1
            else:
                current_chunk += sentence + " "
        
        # Add final chunk
        if current_chunk.strip():
            chunk_info = ChunkInfo(
                text=current_chunk.strip(),
                start_idx=current_start,
                end_idx=current_start + len(current_chunk),
                chunk_id=chunk_id,
                metadata={'sentence_count': len(current_chunk.split('.'))}
            )
            chunks.append(chunk_info)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be enhanced with spaCy or NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, chunk: str) -> str:
        """Get overlap text from the end of a chunk"""
        words = chunk.split()
        if len(words) <= self.overlap:
            return chunk + " "
        return " ".join(words[-self.overlap:]) + " "


class VectorStore:
    """Handles embedding generation and vector storage using FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.is_built = False
    
    def build_index(self, chunks: List[ChunkInfo]) -> None:
        """Build FAISS index from document chunks"""
        logger.info(f"Building vector index for {len(chunks)} chunks...")
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        self.is_built = True
        
        logger.info(f"Vector index built successfully with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k most similar chunks"""
        if not self.is_built:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Format results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], similarities[0])):
            if idx < len(self.chunks):  # Valid index
                result = RetrievalResult(
                    chunk=self.chunks[idx],
                    similarity_score=float(score),
                    rank=rank
                )
                results.append(result)
        
        return results
    
    def save_index(self, path: str) -> None:
        """Save FAISS index to disk"""
        if self.is_built:
            faiss.write_index(self.index, path)
            # Save chunks metadata
            chunks_data = [
                {
                    'text': chunk.text,
                    'start_idx': chunk.start_idx,
                    'end_idx': chunk.end_idx,
                    'chunk_id': chunk.chunk_id,
                    'metadata': chunk.metadata
                }
                for chunk in self.chunks
            ]
            with open(path + ".chunks", 'w') as f:
                json.dump(chunks_data, f)
    
    def load_index(self, path: str) -> None:
        """Load FAISS index from disk"""
        self.index = faiss.read_index(path)
        # Load chunks metadata
        with open(path + ".chunks", 'r') as f:
            chunks_data = json.load(f)
        
        self.chunks = [
            ChunkInfo(
                text=data['text'],
                start_idx=data['start_idx'],
                end_idx=data['end_idx'],
                chunk_id=data['chunk_id'],
                metadata=data.get('metadata', {})
            )
            for data in chunks_data
        ]
        self.is_built = True


class SummaryGenerator:
    """Generates summaries using Together AI LLM"""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        self.model_name = model_name
        self.api_key = os.getenv("TOGETHER_API_KEY")
        
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables. Please check your .env file.")
        
        self.client = Together(api_key=self.api_key)

    def generate_summary(
        self, 
        retrieved_chunks: List[RetrievalResult], 
        max_length: int = 200,
        min_length: int = 50
    ) -> Tuple[str, Dict[str, int]]:
        """Generate summary from retrieved chunks using Together AI"""

        context = self._prepare_context(retrieved_chunks)

        prompt = f"""Based on the following document excerpts, provide a comprehensive and coherent summary. 
The summary should be between {min_length} and {max_length} words and capture the main points and key information from the text.

Document excerpts:
{context}

Please provide a clear, well-structured summary:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that creates concise, accurate summaries. Focus on the main points and key information from the provided text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_length * 2,  # Allow some buffer
                temperature=0.3,  # Lower temperature for more focused summaries
                top_p=0.9,
                stop=None
            )

            summary = response.choices[0].message.content.strip()
            
            # Extract token usage
            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            logger.info(f"Summary generated successfully. Tokens used: {token_usage['total_tokens']}")
            return summary, token_usage

        except Exception as e:
            logger.error(f"Together AI API error: {e}")
            # Provide fallback summary
            fallback_summary = self._create_fallback_summary(retrieved_chunks)
            return fallback_summary, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    def _prepare_context(self, retrieved_chunks: List[RetrievalResult]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        for i, res in enumerate(retrieved_chunks):
            context_parts.append(f"[Excerpt {i+1}]: {res.chunk.text}")
        
        # Limit context to avoid token limits (roughly 3000 characters)
        full_context = "\n\n".join(context_parts)
        if len(full_context) > 3000:
            full_context = full_context[:3000] + "..."
        
        return full_context

    def _create_fallback_summary(self, retrieved_chunks: List[RetrievalResult]) -> str:
        """Create a basic fallback summary if API fails"""
        if not retrieved_chunks:
            return "No relevant content found for summarization."
        
        # Extract key sentences from chunks
        key_sentences = []
        for chunk_result in retrieved_chunks[:3]:  # Use top 3 chunks
            text = chunk_result.chunk.text
            sentences = re.split(r'[.!?]+', text)
            if sentences:
                key_sentences.append(sentences[0].strip())
        
        return "Based on the document analysis: " + " ".join(key_sentences[:3]) + "."