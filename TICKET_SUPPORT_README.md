# Ticket Support AI Service

A retrieval-based AI system for suggesting solutions to support tickets based on historical resolved tickets.

## Java Desktop Integration

The Java desktop app already integrates with this API in its ticket service layer:
- Auto-reply lookup endpoint: `POST /ai/solve`
- Continuous learning endpoint on ticket resolution: `POST /ai/knowledge-base/update`

Config keys used by Java app:
- `TICKET_AI_ENABLED=true|false` (default: `true`)
- `TICKET_AI_API_URL=http://127.0.0.1:8005` (default shown)
- `GROQ_API_KEY=...` (optional, enables paragraph-to-fields autofill in client/worker new ticket form)

When an admin closes/solves a ticket with a resolution, Java pushes that subject/message/resolution back to this API, which appends to `data/ticket_knowledge_base.json` so future similar tickets can be auto-suggested.

## Overview

The Ticket Support AI Service is a FastAPI microservice that uses sentence embeddings and cosine similarity to find similar past tickets and suggest solutions. It is designed to work with an existing Symfony-based ticket system, providing intelligent suggestions while ensuring safety by escalating to human admins when confidence is low.

## AI Approach

### Simple Explanation

The AI system works like a smart search engine for support tickets. When a new ticket arrives, it:
1. Transforms the problem text (subject + message) into a numerical representation (embedding)
2. Compares this representation with all previously solved tickets
3. Finds the most similar past problems
4. If similarity is high enough (≥75%), suggests the resolution from the most similar ticket
5. If similarity is low, escalates to a human admin

### Academic Explanation

This is a retrieval-based recommendation system using sentence embeddings and cosine similarity. We employ the all-MiniLM-L6-v2 model from the Sentence Transformers library, which maps text to a 384-dimensional dense vector space. The model is trained on diverse text sources using contrastive learning, ensuring semantically similar texts are close in the vector space.

For each ticket, we create an embedding of the problem text (subject + first message) and store it alongside the resolution. When a new ticket arrives, we compute its embedding and calculate cosine similarity with all stored embeddings. The cosine similarity score represents the semantic similarity between the new problem and past problems, ranging from -1 (opposite) to 1 (identical). We use this as our confidence metric.

The system is designed to be explainable: we can always trace back which past ticket led to a suggestion. It fails safely by defaulting to human escalation when confidence is below threshold.

## Requirements

- Python 3.10+
- pip

## Installation

```bash
cd ai_service
python -m venv venv

# Windows
venv\Scriptsctivate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Running the Service

```bash
python ticket_support_api.py
```

Or with uvicorn directly:

```bash
cd ai_service
uvicorn ticket_support_api:app --reload --host 127.0.0.1 --port 8002
```

The API will be available at: `http://127.0.0.1:8001`

## API Documentation

Once running, visit:
- Swagger UI: `http://127.0.0.1:8002/docs`
- ReDoc: `http://127.0.0.1:8002/redoc`

## API Endpoints

### POST /ai/solve

Suggest a solution for a new support ticket based on similar past tickets.

**Request:**
```json
{
  "subject": "Unable to login to my account",
  "message": "I've been trying to login to my account for the past hour but keep getting an error message saying 'Invalid credentials'.",
  "category": "Authentication"
}
```

**Response:**
```json
{
  "suggested_solution": "Please check if Caps Lock is on and ensure you're using the correct email address. If the issue persists, click on 'Forgot Password' to reset your password again.",
  "confidence_score": 0.92,
  "escalate_to_admin": false,
  "similar_ticket": {
    "subject": "Unable to login to my account",
    "message": "I've been trying to login to my account for the past hour but keep getting an error message saying 'Invalid credentials'. I'm sure my password is correct as I just reset it yesterday."
  },
  "processed_at": "2023-11-15T14:30:45.123456"
}
```

### POST /ai/knowledge-base/update

Update the knowledge base with a new resolved ticket. Call this endpoint when an admin resolves a ticket.

**Request:**
```json
{
  "subject": "Unable to login to my account",
  "message": "I've been trying to login to my account for the past hour but keep getting an error message saying 'Invalid credentials'.",
  "resolution": "Please check if Caps Lock is on and ensure you're using the correct email address. If the issue persists, click on 'Forgot Password' to reset your password again.",
  "category": "Authentication"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Knowledge base updated successfully",
  "updated_at": "2023-11-15T14:30:45.123456"
}
```

### POST /ai/knowledge-base/retrain

Retrain the model by regenerating embeddings for all tickets in the knowledge base. Call this endpoint after adding multiple tickets to ensure consistency.

**Response:**
```json
{
  "status": "success",
  "message": "Model retrained successfully",
  "retrained_at": "2023-11-15T14:30:45.123456",
  "total_tickets": 15
}
```

### GET /ai/knowledge-base/stats

Get statistics about the knowledge base.

**Response:**
```json
{
  "total_tickets": 15,
  "categories": {
    "Authentication": 5,
    "Billing": 3,
    "Technical": 4,
    "Account": 2,
    "Feature Request": 1
  },
  "last_updated": "2023-11-15T14:30:45.123456"
}
```

## Knowledge Base Format

The knowledge base is stored in a JSON file with the following format:

```json
[
  {
    "subject": "Unable to login to my account",
    "message": "I've been trying to login to my account for the past hour but keep getting an error message saying 'Invalid credentials'. I'm sure my password is correct as I just reset it yesterday.",
    "resolution": "Please check if Caps Lock is on and ensure you're using the correct email address. If the issue persists, click on 'Forgot Password' to reset your password again. Also, try clearing your browser cache and cookies, or use a different browser.",
    "category": "Authentication"
  },
  ...
]
```

## Integration with Symfony

### When a new ticket is created:

1. Symfony sends a POST request to `/ai/solve` with the ticket subject and message
2. The AI returns a suggested solution, confidence score, and escalation decision
3. If `escalate_to_admin` is false, Symfony can display the suggested solution to the user
4. If `escalate_to_admin` is true, Symfony escalates the ticket to a human admin

### When an admin resolves a ticket:

1. Symfony sends a POST request to `/ai/knowledge-base/update` with the ticket details and resolution
2. The AI adds the new ticket to its knowledge base
3. Optionally, Symfony can trigger a retrain by calling `/ai/knowledge-base/retrain`

## Continuous Learning

The system improves over time as more tickets are resolved:
1. When an admin resolves a ticket, the problem and solution are added to the knowledge base
2. The AI can then use this new knowledge to suggest solutions for future similar tickets
3. Periodically, the model can be retrained to ensure all embeddings are consistent

## Security & Robustness

1. **Fail-safe design**: The system defaults to escalating to admin if:
   - No similar tickets are found
   - Confidence score is below threshold (0.75)
   - An error occurs during processing

2. **No hallucination**: The system only suggests solutions that exist in the knowledge base
   - It does not generate new solutions
   - All suggestions can be traced back to a specific past ticket

3. **Explainability**: Each suggestion includes:
   - The confidence score
   - The similar ticket that led to the suggestion
   - This allows admins to understand why a particular solution was suggested

4. **Input validation**: The API validates all inputs to prevent injection attacks
   - Subject and message are required fields
   - Category is optional but validated if provided

5. **CORS configuration**: In production, restrict CORS to only allow requests from your Symfony domain

## Performance Considerations

1. The all-MiniLM-L6-v2 model is lightweight and fast, suitable for real-time applications
2. Embeddings are cached, so similar queries are processed quickly
3. For large knowledge bases (1000+ tickets), consider implementing:
   - Vector database (e.g., FAISS, Pinecone)
   - Caching layer (e.g., Redis)
   - Batch processing for retraining

## Troubleshooting

### Model not loading

If you encounter issues loading the sentence transformer model, ensure you have a stable internet connection the first time you run the service, as the model needs to be downloaded.

### Low confidence scores

If the system consistently returns low confidence scores:
1. Check if the knowledge base contains relevant tickets
2. Verify that the ticket subjects and messages are descriptive enough
3. Consider adjusting the confidence threshold (default is 0.75)

### Slow response times

If the API is slow to respond:
1. Check the size of your knowledge base
2. Consider implementing a vector database for large knowledge bases
3. Add caching for frequently asked questions

## Future Enhancements

1. Multi-language support
2. Category-based filtering
3. A/B testing for confidence thresholds
4. Admin feedback loop to improve suggestions
5. Integration with ticket analytics
