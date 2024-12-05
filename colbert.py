# import torch.nn as nn

# class ColBERT(nn.Module):
#     def __init__(
#         self,
#         query_maxlen: int = 32,
#         doc_maxlen: int = 180,
#         dim: int = 128,
#         mask_punctuation: bool = True,
#         skip_compression: bool = False
#     ):
#         super().__init__()
        
#         self.query_maxlen = query_maxlen
#         self.doc_maxlen = doc_maxlen
#         self.dim = dim
#         self.mask_punctuation = mask_punctuation
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load BERT model and tokenizer
#         self.bert = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
#         # Linear compression layer
#         if not skip_compression:
#             self.linear = nn.Linear(self.bert.config.hidden_size, dim, bias=False).to(self.device)
#         else:
#             self.linear = None
            
#         self.dropout = nn.Dropout(0.1)
        
#         # Pre-compute punctuation IDs
#         self.punct_ids = set(self.tokenizer.encode('.!?,', add_special_tokens=False))

#     def _mask_punctuation(self, ids: torch.Tensor) -> torch.Tensor:
#         """Create mask to ignore punctuation tokens"""
#         mask = torch.ones_like(ids, dtype=torch.bool)
#         for id in self.punct_ids:
#             mask &= (ids != id)
#         return mask

#     def forward_query(self, query: str) -> torch.Tensor:
#         # Tokenize query
#         query_tokens = self.tokenizer(
#             query,
#             max_length=self.query_maxlen,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         # Move tensors to correct device
#         query_tokens = {k: v.to(self.device) for k, v in query_tokens.items()}
        
#         # Get BERT embeddings
#         with torch.no_grad():
#             Q = self.bert(
#                 input_ids=query_tokens['input_ids'],
#                 attention_mask=query_tokens['attention_mask']
#             )[0]
        
#         Q = self.dropout(Q)
#         if self.linear is not None:
#             Q = self.linear(Q)
#         Q = F.normalize(Q, p=2, dim=2)
        
#         query_mask = query_tokens['attention_mask'].bool()
        
#         return Q, query_mask

#     def forward_doc(self, doc: str) -> torch.Tensor:
#         """Encode document tokens"""
#         # Tokenize document with proper padding
#         doc_tokens = self.tokenizer(
#             doc,
#             max_length=self.doc_maxlen,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         # Move tensors to correct device
#         doc_tokens = {k: v.to(self.device) for k, v in doc_tokens.items()}
        
#         # Get BERT embeddings
#         with torch.no_grad():
#             D = self.bert(
#                 input_ids=doc_tokens['input_ids'],
#                 attention_mask=doc_tokens['attention_mask']
#             )[0]  # [batch_size, doc_maxlen, hidden_size]
        
#         D = self.dropout(D)
#         if self.linear is not None:
#             D = self.linear(D)
#         D = F.normalize(D, p=2, dim=2)
        
#         # Get document mask
#         doc_mask = doc_tokens['attention_mask'].bool()
        
#         if self.mask_punctuation:
#             punct_mask = self._mask_punctuation(doc_tokens['input_ids'])
#             doc_mask = doc_mask & punct_mask
        
#         return D, doc_mask

#     def score(self, Q: torch.Tensor, D: torch.Tensor, q_mask: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Calculate MaxSim scores between query and document tokens
        
#         Args:
#             Q: Query embeddings [batch_size, query_len, hidden_dim]
#             D: Document embeddings [batch_size, doc_len, hidden_dim]
#             q_mask: Query attention mask [batch_size, query_len]
#             d_mask: Document attention mask [batch_size, doc_len]
#         """
#         # Ensure all tensors are on the same device
#         Q, D = Q.to(self.device), D.to(self.device)
#         q_mask, d_mask = q_mask.to(self.device), d_mask.to(self.device)
        
#         # Compute similarity matrix
#         similarity = torch.matmul(Q, D.transpose(-2, -1))  # [batch_size, query_len, doc_len]
        
#         # Create properly shaped masks
#         q_mask = q_mask.unsqueeze(-1)  # [batch_size, query_len, 1]
#         d_mask = d_mask.unsqueeze(1)   # [batch_size, 1, doc_len]
        
#         # Broadcast masks to match similarity matrix shape
#         q_mask = q_mask.expand(-1, -1, similarity.size(-1))  # [batch_size, query_len, doc_len]
#         d_mask = d_mask.expand(-1, similarity.size(1), -1)   # [batch_size, query_len, doc_len]
        
#         # Apply masks
#         similarity = similarity.masked_fill(~q_mask, -1e9)
#         similarity = similarity.masked_fill(~d_mask, -1e9)
        
#         # MaxSim operation
#         scores = similarity.max(dim=-1).values  # [batch_size, query_len]
#         scores = scores.sum(dim=-1)  # [batch_size]
        
#         return scores

# class ColBERTRetriever:
#     def __init__(self, model_path: str = None):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = ColBERT().to(self.device)
#         if model_path:
#             self.model.load_state_dict(torch.load(model_path))
#         self.model.eval()

#     def retrieve(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
#         with torch.no_grad():
#             # Encode query
#             Q, q_mask = self.model.forward_query(query)
            
#             scores = []
#             # Encode and score each document
#             for i, doc in enumerate(documents):
#                 D, d_mask = self.model.forward_doc(doc)
#                 score = self.model.score(Q, D, q_mask, d_mask).item()
#                 scores.append((i, score))
            
#             scores.sort(key=lambda x: x[1], reverse=True)
#             return scores[:top_k]

# def rank_documents_colbert(user_query: str, shortlist: int = 15, top_k: int = 5):
#     # Get initial candidates from bi-encoder
#     retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=shortlist)
    
#     # Initialize ColBERT retriever
#     retriever = ColBERTRetriever()
    
#     # Get document contents
#     documents = [doc.page_content for doc in retrieved_docs]
    
#     # Rerank using ColBERT
#     ranked_indices = retriever.retrieve(user_query, documents, top_k)
    
#     # Return reranked document IDs
#     return [retrieved_docs[idx].metadata['source'] for idx, _ in ranked_indices]

# # Test with sample query
# user_query = "what did kirsty williams am say about her plan for quality assurance ?"
# retrieved_docs = rank_documents_colbert(user_query)

# print("\n==================================Top-5 documents==================================")
# print("\n\nRetrieved documents:", retrieved_docs)
# print("\n====================================================================\n")


##################################

from typing import Callable, Dict


class QAEvaluator:
    def __init__(
        self,
        batch_size: int = 32,
        device: str = None,
        progress_bar: bool = True
    ):
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress_bar = progress_bar
        
        # Initialize ColBERT retriever
        self.retriever = ColBERTRetriever()
        
        # Enhanced metrics accumulator with additional precision metrics
        self.metrics = {
            'hits@1': [],
            'hits@5': [],
            'mrr': [],
            'recall@5': [],
            'precision@1': [],
            'precision@5': [],
            'precision@10': [],
            'precision@15': [],
            'precision@20': [],
            'latency': []
        }

    def evaluate_batch(
        self,
        queries: List[str],
        doc_ids: List[str],
        shortlist: int = 20,  # Increased to accommodate precision@20
        top_k: int = 20      # Increased to accommodate precision@20
    ) -> Dict[str, List[float]]:
        """
        Process a batch of queries and calculate metrics
        """
        batch_metrics = {k: [] for k in self.metrics.keys()}
        start_time = time.time()

        # Get initial candidates for all queries in batch
        all_candidates = []
        for query in queries:
            candidates = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=query, k=shortlist)
            all_candidates.append(candidates)

        # Process each query in the batch
        for query, candidates, true_doc_id in zip(queries, all_candidates, doc_ids):
            # Get document contents
            documents = [doc.page_content for doc in candidates]
            
            # Get rankings
            ranked_indices = self.retriever.retrieve(query, documents, top_k)
            ranked_docs = [candidates[idx].metadata['source'] for idx, _ in ranked_indices]

            # Calculate existing metrics
            batch_metrics['hits@1'].append(1 if ranked_docs[0] == true_doc_id else 0)
            batch_metrics['hits@5'].append(1 if true_doc_id in ranked_docs[:5] else 0)
            batch_metrics['recall@5'].append(1 if true_doc_id in ranked_docs[:5] else 0)
            
            # Calculate MRR
            try:
                rank = ranked_docs.index(true_doc_id) + 1
                batch_metrics['mrr'].append(1.0 / rank)
            except ValueError:
                batch_metrics['mrr'].append(0.0)
            
            # Calculate precision@k for different k values
            ks = [1, 5, 10, 15, 20]
            for k in ks:
                is_relevant = true_doc_id in ranked_docs[:k]
                batch_metrics[f'precision@{k}'].append(1 if is_relevant else 0)

        # Calculate latency
        batch_time = time.time() - start_time
        batch_metrics['latency'].append(batch_time / len(queries))

        return batch_metrics
    
    def evaluate(
        self,
        qa_pairs: List[Tuple[str, str, str]],
        progress_callback: Callable = None
    ) -> Dict[str, float]:
        """
        Evaluate the entire dataset
        """
        # Split data
        doc_ids = [x[0] for x in qa_pairs]
        queries = [x[1] for x in qa_pairs]
        
        # Process in batches
        num_batches = (len(queries) + self.batch_size - 1) // self.batch_size
        
        if self.progress_bar:
            pbar = tqdm(total=len(queries), desc="Evaluating")
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(queries))
            
            # Get batch
            batch_queries = queries[start_idx:end_idx]
            batch_doc_ids = doc_ids[start_idx:end_idx]
            
            # Process batch
            batch_metrics = self.evaluate_batch(
                queries=batch_queries,
                doc_ids=batch_doc_ids
            )
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                self.metrics[k].extend(v)
            
            if self.progress_bar:
                pbar.update(end_idx - start_idx)
                
            if progress_callback:
                progress_callback(i, num_batches, self.get_current_metrics())
        
        if self.progress_bar:
            pbar.close()
            
        return self.get_current_metrics()

    def get_current_metrics(self) -> Dict[str, float]:
        """
        Calculate current metrics
        """
        metrics_dict = {
            'hits@1': np.mean(self.metrics['hits@1']),
            'hits@5': np.mean(self.metrics['hits@5']),
            'mrr': np.mean(self.metrics['mrr']),
            'recall@5': np.mean(self.metrics['recall@5']),
            'precision@1': np.mean(self.metrics['precision@1']),
            'precision@5': np.mean(self.metrics['precision@5']),
            'precision@10': np.mean(self.metrics['precision@10']),
            'precision@15': np.mean(self.metrics['precision@15']),
            'precision@20': np.mean(self.metrics['precision@20']),
            'mean_latency': np.mean(self.metrics['latency']),
            'processed_queries': len(self.metrics['mrr'])
        }
        return metrics_dict

def print_progress(batch_num: int, total_batches: int, metrics: Dict[str, float]):
    """Optional progress callback with organized output"""
    print(f"\nBatch {batch_num+1}/{total_batches}")
    print("Current metrics:")
    
    # Print original metrics
    print("\nOriginal metrics:")
    for k in ['hits@1', 'hits@5', 'mrr', 'recall@5']:
        print(f"{k}: {metrics[k]:.3f}")
    
    # Print precision metrics
    print("\nPrecision metrics:")
    for k in ['precision@1', 'precision@5', 'precision@10', 'precision@15', 'precision@20']:
        print(f"{k}: {metrics[k]:.3f}")
    
    print(f"\nPerformance:")
    print(f"mean_latency: {metrics['mean_latency']:.3f}")
    print(f"processed_queries: {metrics['processed_queries']:.0f}")

# Initialize evaluator
evaluator = QAEvaluator(
    batch_size=32,
    progress_bar=True
)

# Evaluate
results = evaluator.evaluate(
    qa_pairs=qa_pairs,
    progress_callback=print_progress
)

print("\nFinal Results:")
print("==============")
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")