# Quick Start Guide

Get up and running with Hierarchical RAG in 5 minutes!

## Prerequisites

✅ Python 3.9 or higher installed  
✅ Google Gemini API key ([Get one free here](https://makersuite.google.com/app/apikey))  
✅ Basic knowledge of command line

## Installation (3 steps)

### Step 1: Download the Code

```bash
# Clone or download the project
cd hierarchical-rag
```

### Step 2: Run Setup Script

**On Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```bash
setup.bat
```

**Manual Installation (if scripts don't work):**
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 3: Run the Application

```bash
streamlit run app_hierarchical.py
```

The app will open in your browser at `http://localhost:8501`

## First Time Usage

### 1. Enter Your API Key

- Open the sidebar (click `>` if collapsed)
- Paste your Google Gemini API key
- Click outside the box to save

### 2. Upload Documents

- Click "📁 Upload your document"
- Select one or more PDF or TXT files
- Sample documents to try:
  - Employment contracts
  - Service agreements
  - NDAs
  - License agreements

### 3. Process Documents

- Click "🚀 Process Documents"
- Wait for processing (usually 10-30 seconds)
- You'll see document cards when ready

### 4. Start Asking Questions!

**Try these example questions:**

Simple queries:
- "What are the payment terms?"
- "When does this agreement terminate?"
- "Who are the parties involved?"

Complex queries:
- "Compare termination clauses across all documents"
- "What happens if payment is 30 days late?"
- "List all dates and amounts mentioned"

Use quick action buttons for common tasks!

## Understanding the Interface

### Tabs

**📄 Documents**
- Upload and manage documents
- View document overview
- Compare documents

**💬 Chat**
- Ask questions
- Get AI-generated answers
- View citations and sources

**🕸️ Knowledge Graph**
- Visualize entities and relationships
- See entity statistics
- Explore connections

**📊 Analytics**
- View document statistics
- See processing metrics
- Track usage

### Sidebar Settings

**⚙️ Retrieval Settings**
- **Hybrid**: Best for most cases (combines semantic + keyword)
- **Dense Only**: Pure semantic search
- **Hierarchical**: Section-aware search

**✨ Features**
- **Show Reasoning Steps**: See how the AI thinks
- **Build Knowledge Graph**: Extract entities and relationships
- **Extract Entities**: Identify people, organizations, dates, etc.

## Tips for Best Results

### Document Preparation

✅ **DO:**
- Use clear, well-structured documents
- Include section headers
- Ensure text is searchable (not scanned images)

❌ **DON'T:**
- Upload extremely large files (>50 pages) without testing first
- Use low-quality scanned PDFs
- Upload non-English documents (not optimized yet)

### Query Tips

✅ **Good Queries:**
- "What is the termination period in Section 5?"
- "Compare payment terms across both contracts"
- "Extract all monetary amounts and their purposes"

❌ **Avoid:**
- Vague questions: "Tell me about this"
- Questions about content not in documents
- Extremely complex multi-part questions (break them down)

### Performance Tips

🚀 **Speed up processing:**
- Start with fewer documents (1-2)
- Use smaller chunk sizes for faster retrieval
- Disable knowledge graph for quick tests

💡 **Improve accuracy:**
- Use Hybrid retrieval mode
- Enable reranking
- Increase Top K results for complex queries

## Troubleshooting

### Common Issues

**"API key looks too short"**
- Make sure you copied the entire key
- Keys start with "AIza"
- Get a new key if needed

**"Error processing documents"**
- Check file format (PDF/TXT only)
- Ensure file isn't corrupted
- Try with a smaller document first

**"No relevant information found"**
- Rephrase your question
- Try using keywords from the document
- Increase Top K results in settings

**Slow performance**
- Reduce number of documents
- Decrease Top K results
- Disable knowledge graph temporarily

**Import errors**
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt` again
- Check Python version (needs 3.9+)

### Getting Help

1. Check the full README.md for detailed documentation
2. Review error messages carefully
3. Try the demo functions in individual modules
4. Create an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information

## Next Steps

Once you're comfortable with basics:

1. **Explore Advanced Features**
   - Try document comparison
   - Experiment with knowledge graph queries
   - Use different retrieval strategies

2. **Test Different Document Types**
   - Legal contracts
   - Research papers
   - Technical documentation
   - Policy documents

3. **Customize the System**
   - Modify clause extraction patterns
   - Adjust retrieval parameters
   - Add custom entity types

4. **Extend Functionality**
   - Add new clause types
   - Implement custom retrievers
   - Create specialized prompts

## Example Workflow

Here's a complete workflow for analyzing legal contracts:

```
1. Upload 2-3 service agreements ✅
   ↓
2. Process documents (wait ~20 seconds) ✅
   ↓
3. Go to Documents tab → View overview ✅
   ↓
4. Click "Find Termination Clauses" quick action ✅
   ↓
5. Review AI response with citations ✅
   ↓
6. Go to Knowledge Graph tab ✅
   ↓
7. View extracted entities and relationships ✅
   ↓
8. Return to Chat tab ✅
   ↓
9. Ask: "Compare payment terms across documents" ✅
   ↓
10. Download chat history for records ✅
```

## Sample Questions by Use Case

### Contract Analysis
- "What are the key obligations of each party?"
- "Identify potential risks or unfavorable terms"
- "What happens in case of breach?"
- "Compare liability clauses across contracts"

### Due Diligence
- "List all financial obligations and amounts"
- "Extract all important dates and deadlines"
- "Identify third-party dependencies"
- "What are the termination conditions?"

### Compliance Check
- "Find all confidentiality requirements"
- "What are the governing law provisions?"
- "List all regulatory references"
- "Identify data protection clauses"

## Performance Benchmarks

Expected performance on typical hardware:

| Task | Time | Notes |
|------|------|-------|
| Upload 3 PDFs (20 pages each) | 2-5s | Network dependent |
| Process documents | 15-30s | Depends on size |
| Simple query | 2-5s | Single retrieval |
| Complex query | 5-15s | Multiple retrievals |
| Knowledge graph build | 10-20s | For 3 documents |

## Keyboard Shortcuts

- `Ctrl/Cmd + Enter`: Send chat message
- `Ctrl/Cmd + K`: Focus search/chat
- `R`: Rerun app (in development mode)

## Best Practices

1. **Start Small**: Test with 1-2 documents first
2. **Use Quick Actions**: For common queries
3. **Enable Reasoning**: To understand AI's logic
4. **Compare Systematically**: Use comparison tab
5. **Save Important Chats**: Download history
6. **Experiment**: Try different settings

## What's Next?

🎓 **For Capstone Projects:**
- Add evaluation metrics
- Create benchmark dataset
- Conduct user studies
- Write technical report

🚀 **For Production Use:**
- Add authentication
- Implement database storage
- Scale to more documents
- Add API endpoints

📚 **For Learning:**
- Study the code modules
- Modify retrieval strategies
- Add new features
- Contribute improvements

---

**Ready to analyze? Run `streamlit run app_hierarchical.py` and start exploring! 🚀**

Need more help? Check README.md or individual module documentation.
