import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import numpy as np

@dataclass
class RankedDocument:
    """Class to store ranked document information"""
    doc_id: str
    score: float
    content: Optional[str] = None
    metadata: Optional[dict] = None
    chunk_id: Optional[int] = None  # Track different chunks of same document

class EnhancedColBERTReranker:
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        max_query_length: int = 32,
        max_doc_length: int = 256,
        similarity_metric: str = "maxsim",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize enhanced ColBERT reranker with configurable parameters
        
        Args:
            model_name: HuggingFace model to use
            max_query_length: Maximum length for query tokens
            max_doc_length: Maximum length for document tokens
            similarity_metric: Similarity computation method ('maxsim', 'meansim', 'attention')
            device: Computing device to use
        """
        self.device = device
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.similarity_metric = similarity_metric
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def _get_embeddings(self, text: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token-level embeddings and attention mask for text
        
        Returns:
            Tuple of (embeddings, attention_mask)
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state, inputs['attention_mask']

    def _compute_similarity(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor
    ) -> float:
        """
        Compute similarity between query and document embeddings using different methods
        """
        # Apply masks to zero out padding tokens
        query_emb = query_emb * query_mask.unsqueeze(-1)
        doc_emb = doc_emb * doc_mask.unsqueeze(-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(query_emb, doc_emb.transpose(-2, -1))
        
        if self.similarity_metric == 'maxsim':
            # Original ColBERT MaxSim approach
            max_similarities = torch.max(similarity_matrix, dim=-1).values
            score = max_similarities.sum().item()
            
        elif self.similarity_metric == 'meansim':
            # Alternative: use mean similarity
            mean_similarities = torch.mean(similarity_matrix, dim=-1)
            score = mean_similarities.sum().item()
            
        elif self.similarity_metric == 'attention':
            # Attention-based scoring
            attention_weights = F.softmax(similarity_matrix / np.sqrt(query_emb.size(-1)), dim=-1)
            weighted_sims = torch.sum(attention_weights * similarity_matrix, dim=-1)
            score = weighted_sims.sum().item()
            
        return score

    def _aggregate_chunk_scores(
        self,
        scored_chunks: List[RankedDocument],
        aggregation_method: str = 'max'
    ) -> List[RankedDocument]:
        """
        Aggregate scores for different chunks of the same document
        """
        doc_scores: Dict[str, List[float]] = {}
        doc_contents: Dict[str, List[str]] = {}
        
        # Group chunks by document ID
        for chunk in scored_chunks:
            if chunk.doc_id not in doc_scores:
                doc_scores[chunk.doc_id] = []
                doc_contents[chunk.doc_id] = []
            doc_scores[chunk.doc_id].append(chunk.score)
            doc_contents[chunk.doc_id].append(chunk.content)

        # Aggregate scores
        aggregated_docs = []
        for doc_id, scores in doc_scores.items():
            if aggregation_method == 'max':
                final_score = max(scores)
            elif aggregation_method == 'mean':
                final_score = sum(scores) / len(scores)
            elif aggregation_method == 'sum':
                final_score = sum(scores)
                
            # Combine chunk contents
            combined_content = ' '.join(doc_contents[doc_id])
            
            aggregated_docs.append(
                RankedDocument(
                    doc_id=doc_id,
                    score=final_score,
                    content=combined_content,
                    metadata={'original_chunks': len(scores)}
                )
            )
            
        return sorted(aggregated_docs, key=lambda x: x.score, reverse=True)

    def rerank(
        self,
        query: str,
        documents: List[dict],
        shortlist: int = 15,
        top_k: int = 5,
        aggregate_chunks: bool = True,
        aggregation_method: str = 'max'
    ) -> List[RankedDocument]:
        """
        Rerank documents using enhanced ColBERT-style token-level interactions
        
        Args:
            query: User query string
            documents: List of document dictionaries
            shortlist: Number of documents to consider from initial retrieval
            top_k: Number of documents to return after reranking
            aggregate_chunks: Whether to aggregate scores for different chunks of same document
            aggregation_method: Method to aggregate chunk scores ('max', 'mean', 'sum')
        """
        # Get query embeddings and mask
        query_embeddings, query_mask = self._get_embeddings(
            query, 
            self.max_query_length
        )
        
        # Score each document
        scored_docs = []
        for idx, doc in enumerate(documents[:shortlist]):
            # Get document embeddings and mask
            doc_embeddings, doc_mask = self._get_embeddings(
                doc['content'],
                self.max_doc_length
            )
            
            # Compute similarity score
            score = self._compute_similarity(
                query_embeddings,
                doc_embeddings,
                query_mask,
                doc_mask
            )
            
            # Store scored document
            scored_docs.append(
                RankedDocument(
                    doc_id=doc['metadata'].get('source', ''),
                    score=score,
                    content=doc['content'],
                    metadata=doc['metadata'],
                    chunk_id=idx
                )
            )
        
        # Aggregate chunks if needed
        if aggregate_chunks:
            scored_docs = self._aggregate_chunk_scores(
                scored_docs,
                aggregation_method
            )
        else:
            scored_docs.sort(key=lambda x: x.score, reverse=True)
            
        return scored_docs[:top_k]

# Example usage
if __name__ == "__main__":
    # Initialize reranker with specific configuration
    reranker = EnhancedColBERTReranker(
        max_query_length=32,
        max_doc_length=256,
        similarity_metric='maxsim'  # Try different metrics: 'maxsim', 'meansim', 'attention'
    )
    
    # Example query
    query = "what did kirsty williams am say about her plan for quality assurance?"
    documents = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=query, k=15)
    
    # Convert documents to expected format
    doc_list = [
        {
            'content': doc.page_content,
            'metadata': doc.metadata
        }
        for doc in documents
    ]
    
    # Rerank with chunk aggregation
    ranked_docs = reranker.rerank(
        query,
        doc_list,
        aggregate_chunks=True,
        aggregation_method='max'
    )
    
    # Print results
    print("\n==================================Top-5 documents==================================")
    for doc in ranked_docs:
        print(f"Document ID: {doc.doc_id}")
        print(f"Score: {doc.score:.4f}")
        if doc.metadata and 'original_chunks' in doc.metadata:
            print(f"Number of chunks: {doc.metadata['original_chunks']}")
        print("---")