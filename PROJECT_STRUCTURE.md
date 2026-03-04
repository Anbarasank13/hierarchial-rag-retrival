# 📁 Hierarchical RAG Project Structure

## Complete File Listing

```
hierarchical-rag/
│
├── 📄 Core Application Files
│   ├── app_hierarchical.py          (32 KB) - Main Streamlit application
│   ├── document_processor.py        (12 KB) - Hierarchical document processing
│   ├── retrieval_strategies.py      (14 KB) - Hybrid retrieval implementations  
│   ├── knowledge_graph.py           (16 KB) - Knowledge graph construction
│   ├── clause_extractor.py          (16 KB) - Legal clause extraction
│   └── comparison_engine.py         (16 KB) - Document comparison tools
│
├── 📋 Configuration & Setup
│   ├── requirements.txt             - Python dependencies
│   ├── setup.sh                     - Linux/Mac setup script
│   ├── setup.bat                    - Windows setup script
│   └── .env                         - API keys (create this yourself)
│
├── 📖 Documentation
│   ├── README.md                    (13 KB) - Comprehensive documentation
│   ├── QUICKSTART.md                (7.7 KB) - Quick start guide
│   └── PROJECT_STRUCTURE.md         - This file
│
├── 🧪 Testing & Samples
│   ├── test_system.py               (8.9 KB) - System verification tests
│   └── generate_samples.py          (20 KB) - Sample document generator
│
└── 📁 Generated Directories
    ├── sample_documents/            - Sample legal documents (auto-generated)
    ├── venv/                        - Python virtual environment (created during setup)
    └── .streamlit/                  - Streamlit configuration (auto-created)
```

## File Descriptions

### 🎯 Core Modules

#### `app_hierarchical.py` (Main Application)
**Purpose:** Streamlit web interface for the Hierarchical RAG system

**Key Components:**
- Multi-tab interface (Documents, Chat, Knowledge Graph, Analytics)
- Document upload and processing
- Chat interface with streaming responses
- Quick action buttons
- Knowledge graph visualization
- Document comparison interface
- Analytics dashboard

**Entry Point:** Run with `streamlit run app_hierarchical.py`

#### `document_processor.py` (Document Processing)
**Purpose:** Extract hierarchical structure from documents

**Key Classes:**
- `HierarchicalDocumentProcessor`: Main processor class
- `SectionNavigator`: Navigate document sections

**Features:**
- Section hierarchy detection (Articles, Numbered sections, etc.)
- Metadata extraction (dates, parties, amounts)
- Hierarchical chunking with context
- Pattern matching for different document styles

**Demo Function:** `demo_hierarchical_processing()`

#### `retrieval_strategies.py` (Advanced Retrieval)
**Purpose:** Implement multiple retrieval strategies

**Key Classes:**
- `HybridRetriever`: Combines dense + sparse search
- `QueryExpander`: Generate query variations
- `ReRanker`: Re-rank retrieved results

**Retrieval Methods:**
- Dense search (semantic/embeddings)
- Sparse search (BM25 keyword matching)
- Hybrid search (weighted combination)
- Hierarchical search (section-aware)
- Multi-query search

**Demo Function:** `demo_hybrid_retrieval()`

#### `knowledge_graph.py` (Knowledge Graph)
**Purpose:** Build entity-relationship graphs from documents

**Key Classes:**
- `KnowledgeGraphBuilder`: Main graph builder

**Features:**
- Entity extraction (Persons, Orgs, Dates, Money, etc.)
- Relationship detection
- spaCy integration (optional)
- Rule-based patterns
- Graph querying and statistics

**Supported Entity Types:**
- PERSON, ORG, DATE, MONEY, GPE, LAW, CLAUSE

**Demo Function:** `demo_knowledge_graph()`

#### `clause_extractor.py` (Legal Clause Extraction)
**Purpose:** Identify and extract legal clauses

**Key Classes:**
- `ClauseExtractor`: Main extractor class

**Clause Types Supported:**
1. Termination clauses
2. Payment terms
3. Confidentiality agreements
4. Liability limitations
5. Intellectual property rights
6. Governing law provisions
7. Dispute resolution mechanisms
8. Force majeure clauses
9. Warranties
10. Term duration

**Features:**
- Keyword-based detection
- Pattern matching
- Confidence scoring
- Clause comparison
- Key term extraction

**Demo Function:** `demo_clause_extraction()`

#### `comparison_engine.py` (Document Comparison)
**Purpose:** Compare multiple documents side-by-side

**Key Classes:**
- `DocumentComparator`: Main comparison engine

**Comparison Types:**
- Structure comparison
- Clause comparison
- Entity comparison
- Content similarity analysis
- Metadata comparison

**Outputs:**
- Comparison reports
- Side-by-side tables
- Similarity scores
- Unique elements identification

**Demo Function:** `demo_comparison()`

### 📋 Configuration Files

#### `requirements.txt`
**Purpose:** Python package dependencies

**Key Dependencies:**
- `streamlit` - Web interface
- `langchain` - RAG framework
- `langchain-google-genai` - Gemini integration
- `faiss-cpu` - Vector store
- `rank-bm25` - Sparse retrieval
- `networkx` - Graph operations
- `pyvis` - Graph visualization
- `spacy` - NLP (optional)

#### `setup.sh` / `setup.bat`
**Purpose:** Automated setup scripts

**What They Do:**
1. Check Python version
2. Create virtual environment
3. Install dependencies
4. Download spaCy model (optional)
5. Create .env file (optional)
6. Create sample directories

### 📖 Documentation Files

#### `README.md`
**Purpose:** Comprehensive project documentation

**Sections:**
- Project overview
- Features list
- Installation instructions
- Usage guide
- Architecture details
- Evaluation metrics
- Future enhancements
- Academic context

#### `QUICKSTART.md`
**Purpose:** Get started in 5 minutes

**Sections:**
- Prerequisites
- Quick installation
- First-time usage
- Example queries
- Troubleshooting
- Next steps

### 🧪 Testing Files

#### `test_system.py`
**Purpose:** Verify installation and functionality

**Test Suites:**
1. Package imports
2. Custom modules
3. Document processor
4. Knowledge graph
5. Clause extractor
6. Retrieval strategies

**Usage:** `python test_system.py`

**Output:**
- Pass/Fail status for each test
- Detailed error messages
- Setup recommendations

#### `generate_samples.py`
**Purpose:** Create sample legal documents

**Generates:**
1. Service Agreement (Professional services contract)
2. NDA (Non-Disclosure Agreement)
3. Employment Contract

**Usage:** `python generate_samples.py`

**Output:** Creates `sample_documents/` directory with 3 test files

## Module Dependencies

```
app_hierarchical.py
    ├── document_processor.py
    ├── retrieval_strategies.py
    │   └── rank_bm25
    ├── knowledge_graph.py
    │   └── spacy (optional)
    ├── clause_extractor.py
    └── comparison_engine.py
        ├── clause_extractor.py
        └── knowledge_graph.py
```

## Data Flow

```
1. Document Upload (PDF/TXT)
   ↓
2. Document Processing (document_processor.py)
   ├── Structure extraction
   ├── Metadata extraction
   └── Hierarchical chunking
   ↓
3. Index Building
   ├── Vector store (FAISS)
   ├── BM25 index
   └── Section index
   ↓
4. Knowledge Graph (knowledge_graph.py)
   ├── Entity extraction
   └── Relationship detection
   ↓
5. User Query
   ↓
6. Retrieval (retrieval_strategies.py)
   ├── Dense search
   ├── Sparse search
   └── Hybrid combination
   ↓
7. Re-ranking & Filtering
   ↓
8. Response Generation (LLM)
   ├── Context from retrieved chunks
   ├── Structured reasoning
   └── Citation management
   ↓
9. Display Results
   ├── Answer with citations
   ├── Source references
   └── Reasoning steps
```

## API Integration Points

### Google Gemini API
**Used For:**
- Embeddings (`text-embedding-004`)
- Generation (`gemini-2.0-flash-exp`, `gemini-1.5-pro`, etc.)

**Configuration:**
- Set via sidebar in app
- Or via `.env` file: `GOOGLE_API_KEY=your_key`

### Vector Store (FAISS)
**Purpose:** Dense retrieval

**Operations:**
- Index creation from embeddings
- Similarity search
- Batch processing

### BM25 Index
**Purpose:** Sparse keyword retrieval

**Operations:**
- Tokenization
- TF-IDF scoring
- Keyword matching

## Extending the System

### Adding New Clause Types

**File:** `clause_extractor.py`

**Steps:**
1. Add entry to `CLAUSE_TYPES` dictionary
2. Define keywords
3. Define regex patterns
4. Test with sample documents

**Example:**
```python
'new_clause_type': {
    'keywords': ['keyword1', 'keyword2'],
    'patterns': [
        r'pattern1',
        r'pattern2',
    ]
}
```

### Adding New Retrieval Strategies

**File:** `retrieval_strategies.py`

**Steps:**
1. Add method to `HybridRetriever` class
2. Implement retrieval logic
3. Update app UI to expose new strategy
4. Test with sample queries

### Adding New Entity Types

**File:** `knowledge_graph.py`

**Steps:**
1. Add patterns to `_extract_with_rules()`
2. Define entity type constant
3. Add color mapping in `get_node_color()`
4. Test extraction

## Performance Characteristics

### Expected Processing Times
(On typical hardware: 16GB RAM, 4-core CPU)

| Operation | Small Doc (10 pages) | Medium Doc (50 pages) | Large Doc (200 pages) |
|-----------|---------------------|----------------------|----------------------|
| Upload | <1s | 1-2s | 3-5s |
| Processing | 5-10s | 20-40s | 60-120s |
| Knowledge Graph | 5-10s | 15-30s | 45-90s |
| Simple Query | 2-4s | 3-6s | 4-8s |
| Complex Query | 5-10s | 8-15s | 12-25s |

### Memory Usage

| Component | Memory Footprint |
|-----------|-----------------|
| Base App | ~200 MB |
| Per Document (10 pages) | ~5-10 MB |
| Vector Store (1000 chunks) | ~50-100 MB |
| Knowledge Graph (100 nodes) | ~5-10 MB |
| spaCy Model (if loaded) | ~500 MB |

### Scalability Limits

**Current Architecture:**
- Documents: Up to 10-20 simultaneously
- Chunks: Up to 5,000 efficiently
- Entities: Up to 1,000 in graph
- Concurrent Users: Single user (Streamlit limitation)

**For Production:**
- Use database backend (PostgreSQL + pgvector)
- Implement caching (Redis)
- Load balancing for multiple users
- Async processing for large documents

## Common Customizations

### 1. Adjust Chunk Size

**File:** `app_hierarchical.py`

**Change:**
```python
CHUNK_SIZE = 1000  # Increase for more context
CHUNK_OVERLAP = 200  # Increase to reduce fragmentation
```

### 2. Change Retrieval K

**File:** `app_hierarchical.py`

**Change:**
```python
RETRIEVER_K = 6  # Increase for more context
```

### 3. Modify Hybrid Weights

**File:** `retrieval_strategies.py`, method `hybrid_search()`

**Change:**
```python
def hybrid_search(
    self, 
    query: str, 
    k: int = None,
    dense_weight: float = 0.7,  # Adjust these
    sparse_weight: float = 0.3   # Adjust these
)
```

### 4. Add Custom Prompts

**File:** `app_hierarchical.py`, function `create_reasoning_prompt()`

**Customize the template string**

## Security Considerations

### API Keys
- ⚠️ Never commit `.env` file to version control
- ⚠️ Keep API keys secure
- ⚠️ Use environment variables in production

### User Data
- Documents are processed in memory
- No persistent storage by default
- Session data cleared on refresh

### Production Deployment
- Add authentication
- Implement rate limiting
- Use HTTPS
- Sanitize file uploads
- Validate input queries

## Troubleshooting Guide

### Import Errors
**Solution:** Ensure virtual environment is activated and dependencies installed
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### API Key Errors
**Solution:** Check key format and validity
- Should start with "AIza"
- Get new key if needed from Google AI Studio

### Processing Errors
**Solution:** Check document format
- PDF should be text-based, not scanned
- Text should be in English
- File size should be reasonable (<50 MB)

### Memory Errors
**Solution:** Reduce batch size or chunk size
- Process fewer documents
- Reduce CHUNK_SIZE
- Close other applications

### Slow Performance
**Solution:** Optimize settings
- Use smaller models (gemini-1.5-flash)
- Reduce Top K
- Disable knowledge graph for testing

## Version History

**v1.0.0** - Initial Release
- Core hierarchical RAG functionality
- Multiple retrieval strategies
- Knowledge graph construction
- Legal clause extraction
- Document comparison
- Streamlit web interface

## Future Roadmap

### Short Term
- [ ] Add PDF highlighting for citations
- [ ] Export comparison reports
- [ ] Batch document processing
- [ ] Custom clause type definitions

### Medium Term
- [ ] Fine-tuned embeddings for legal text
- [ ] Multi-language support
- [ ] Database backend
- [ ] API endpoints

### Long Term
- [ ] Contract risk analysis
- [ ] Automated clause generation
- [ ] Integration with legal databases
- [ ] Enterprise deployment options

## Credits & License

**Built For:** Educational/Academic Capstone Projects

**Technologies Used:**
- LangChain
- Google Gemini
- Streamlit
- FAISS
- NetworkX
- spaCy

**License:** MIT (or as appropriate for your project)

---

**For questions or contributions:**
- Create GitHub issue
- Submit pull request
- Contact project maintainer

**Last Updated:** January 30, 2026
