"""
Ticket Support AI Service
A retrieval-based AI system for suggesting solutions to support tickets
based on historical resolved tickets.

Uses multi-strategy matching:
1. Full-text semantic similarity (subject + message)
2. Subject-only semantic similarity
3. Keyword overlap scoring
The best score across all strategies is used.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import os


class TicketSupportAI:
    """
    AI-powered support ticket solution suggester using sentence embeddings
    and cosine similarity for retrieval-based recommendations.
    
    Enhanced with multi-strategy matching for better accuracy with
    varied phrasings and short/informal queries.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", knowledge_base_path: str = None):
        """
        Initialize the Ticket Support AI system.

        Args:
            model_name: Name of the sentence-transformers model to use
            knowledge_base_path: Path to the JSON file containing resolved tickets
        """
        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)

        # Initialize knowledge base
        self.tickets = []
        self.embeddings = None
        self.subject_embeddings = None  # Separate embeddings for subjects only

        # Load knowledge base if path is provided
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            self.load_knowledge_base(knowledge_base_path)

    def load_knowledge_base(self, file_path: str) -> None:
        """
        Load resolved tickets from a JSON file.

        Args:
            file_path: Path to the JSON file containing resolved tickets
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.tickets = json.load(f)

        # Generate embeddings for all tickets
        self._generate_embeddings()

    def _generate_embeddings(self) -> None:
        """
        Generate embeddings for all tickets in the knowledge base.
        Creates both full-text and subject-only embeddings.
        """
        if not self.tickets:
            self.embeddings = np.array([])
            self.subject_embeddings = np.array([])
            return

        # Full-text embeddings (subject + message)
        texts = [f"{ticket['subject']} {ticket['message']}" for ticket in self.tickets]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Subject-only embeddings for better matching on short queries
        subjects = [ticket['subject'] for ticket in self.tickets]
        self.subject_embeddings = self.model.encode(subjects, convert_to_numpy=True)

    def _extract_keywords(self, text: str) -> set:
        """
        Extract meaningful keywords from text, removing common stop words.
        """
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
            'they', 'them', 'this', 'that', 'is', 'am', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'can', 'may', 'might',
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'not', 'no', 'so',
            'up', 'out', 'about', 'into', 'over', 'after', 'been', 'very',
            'just', 'also', 'than', 'too', 'please', 'help', 'need',
            'want', 'get', 'got', 'keep', 'try', 'tried', 'trying',
            't', 's', 'don', 'doesn', 'didn', 'can', 'won', 'isn', 'aren',
            've', 'll', 're', 'im', 'ive', 'cant', 'dont', 'doesnt',
        }
        # Normalize: lowercase, split on non-alpha
        words = set(re.findall(r'[a-z]+', text.lower()))
        return words - stop_words

    def _keyword_similarity(self, query_text: str, ticket_idx: int) -> float:
        """
        Calculate keyword overlap similarity between query and a knowledge base ticket.
        Returns a score between 0.0 and 1.0.
        """
        query_keywords = self._extract_keywords(query_text)
        ticket_text = f"{self.tickets[ticket_idx]['subject']} {self.tickets[ticket_idx]['message']}"
        ticket_keywords = self._extract_keywords(ticket_text)

        if not query_keywords or not ticket_keywords:
            return 0.0

        # Jaccard-like overlap: intersection / min(len) to favor short queries matching
        overlap = query_keywords & ticket_keywords
        score = len(overlap) / min(len(query_keywords), len(ticket_keywords))
        return score

    def add_ticket(self, subject: str, message: str, resolution: str, category: str = None) -> None:
        """
        Add a new resolved ticket to the knowledge base.

        Args:
            subject: Ticket subject
            message: Ticket message (problem description)
            resolution: Admin's resolution for the problem
            category: Ticket category (optional)
        """
        # Create ticket entry
        ticket = {
            "subject": subject,
            "message": message,
            "resolution": resolution,
            "category": category
        }

        # Add to knowledge base
        self.tickets.append(ticket)

        # Generate embeddings for this ticket
        text = f"{subject} {message}"
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        subject_embedding = self.model.encode([subject], convert_to_numpy=True)[0]

        # Update embeddings arrays
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = np.array([embedding])
            self.subject_embeddings = np.array([subject_embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
            self.subject_embeddings = np.vstack([self.subject_embeddings, subject_embedding])

    def add_ticket_if_new(self, subject: str, message: str, resolution: str, category: str = None) -> bool:
        """
        Add a new resolved ticket only if it is not already in the knowledge base.

        Returns:
            True if the ticket was added, False if it was a duplicate.
        """
        if self._is_duplicate(subject, message):
            return False
        self.add_ticket(subject=subject, message=message, resolution=resolution, category=category)
        return True

    def _is_duplicate(self, subject: str, message: str) -> bool:
        subject_key = self._normalize_key(subject)
        message_key = self._normalize_key(message)
        if not subject_key or not message_key:
            return False

        for ticket in self.tickets:
            if (self._normalize_key(ticket.get("subject")) == subject_key
                    and self._normalize_key(ticket.get("message")) == message_key):
                return True
        return False

    def _normalize_key(self, value: Optional[str]) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", value.strip().lower())

    def find_similar_tickets(self, subject: str, message: str, top_k: int = 3) -> List[Dict]:
        """
        Find similar tickets using multi-strategy matching:
        1. Full-text similarity (subject + message)
        2. Subject-only similarity
        3. Keyword overlap
        
        The best score across all strategies is used for each ticket.

        Args:
            subject: Ticket subject
            message: Ticket message (problem description)
            top_k: Number of similar tickets to return

        Returns:
            List of similar tickets with similarity scores
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # ---- Strategy 1: Full-text semantic similarity ----
        query_text = f"{subject} {message}"
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        full_similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # ---- Strategy 2: Subject-only semantic similarity ----
        subject_embedding = self.model.encode([subject], convert_to_numpy=True)
        if self.subject_embeddings is not None and len(self.subject_embeddings) > 0:
            subject_similarities = cosine_similarity(subject_embedding, self.subject_embeddings)[0]
        else:
            subject_similarities = np.zeros(len(self.tickets))

        # ---- Strategy 3: Keyword overlap ----
        keyword_scores = np.array([
            self._keyword_similarity(query_text, i) for i in range(len(self.tickets))
        ])

        # ---- Combine: take the best score per ticket across strategies ----
        # Weight: full_text=1.0, subject_only=0.9, keywords=0.7
        combined = np.maximum(
            full_similarities,
            np.maximum(subject_similarities * 0.9, keyword_scores * 0.7)
        )

        # Get indices of top_k most similar tickets
        top_indices = np.argsort(combined)[::-1][:top_k]

        # Prepare results
        results = []
        for idx in top_indices:
            result = {
                "ticket": self.tickets[idx],
                "similarity_score": float(combined[idx]),
                "debug_scores": {
                    "full_text": float(full_similarities[idx]),
                    "subject_only": float(subject_similarities[idx]),
                    "keyword": float(keyword_scores[idx]),
                }
            }
            results.append(result)

        return results

    def suggest_solution(self, subject: str, message: str, confidence_threshold: float = 0.3) -> Dict:
        """
        Suggest a solution for a new ticket based on similar past tickets.

        Uses multi-strategy matching and a low threshold (0.3) to ensure
        paraphrased or informal queries still get matched.

        Args:
            subject: Ticket subject
            message: Ticket message (problem description)
            confidence_threshold: Minimum confidence score to suggest a solution

        Returns:
            Dictionary containing suggested solution, confidence score, and escalation decision
        """
        # Find similar tickets
        similar_tickets = self.find_similar_tickets(subject, message, top_k=1)

        # If no similar tickets found, escalate to admin
        if not similar_tickets:
            return {
                "suggested_solution": None,
                "confidence_score": 0.0,
                "escalate_to_admin": True,
                "similar_ticket": None
            }

        # Get the most similar ticket
        most_similar = similar_tickets[0]
        confidence = most_similar["similarity_score"]

        print(f"[AI] Query: '{subject}' | Best match: '{most_similar['ticket']['subject']}' | Score: {confidence:.4f}")
        print(f"[AI] Debug scores: {most_similar.get('debug_scores', {})}")

        # Determine if we should suggest the solution or escalate
        if confidence >= confidence_threshold:
            return {
                "suggested_solution": most_similar["ticket"]["resolution"],
                "confidence_score": confidence,
                "escalate_to_admin": False,
                "similar_ticket": {
                    "subject": most_similar["ticket"]["subject"],
                    "message": most_similar["ticket"]["message"]
                }
            }
        else:
            return {
                "suggested_solution": None,
                "confidence_score": confidence,
                "escalate_to_admin": True,
                "similar_ticket": {
                    "subject": most_similar["ticket"]["subject"],
                    "message": most_similar["ticket"]["message"]
                }
            }

    def save_knowledge_base(self, file_path: str) -> None:
        """
        Save the current knowledge base to a JSON file.

        Args:
            file_path: Path where to save the JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.tickets, f, ensure_ascii=False, indent=2)

    def update_model(self) -> None:
        """
        Regenerate embeddings for all tickets in the knowledge base.
        Useful when multiple tickets have been added and you want to ensure consistency.
        """
        self._generate_embeddings()
