"""
Test Script for Hierarchical RAG System
Run this to verify all modules are working correctly
"""

import sys
from typing import Dict, Any

def test_imports():
    """Test if all required packages are installed"""
    print("="*60)
    print("Testing Package Imports...")
    print("="*60)
    
    packages = [
        ('streamlit', 'Streamlit'),
        ('langchain', 'LangChain'),
        ('langchain_google_genai', 'Google Generative AI'),
        ('langchain_community', 'LangChain Community'),
        ('dotenv', 'Python Dotenv'),
        ('faiss', 'FAISS'),
        ('networkx', 'NetworkX'),
        ('pandas', 'Pandas'),
        ('rank_bm25', 'BM25'),
    ]
    
    results = {}
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name:30} - OK")
            results[name] = True
        except ImportError as e:
            print(f"❌ {name:30} - FAILED: {str(e)}")
            results[name] = False
    
    # Optional packages
    print("\nOptional Packages:")
    try:
        import spacy
        print(f"✅ {'spaCy':30} - OK")
        try:
            nlp = spacy.load("en_core_web_sm")
            print(f"✅ {'spaCy Model (en_core_web_sm)':30} - OK")
        except:
            print(f"⚠️  {'spaCy Model (en_core_web_sm)':30} - Not installed (optional)")
    except ImportError:
        print(f"⚠️  {'spaCy':30} - Not installed (optional)")
    
    print()
    return all(results.values())


def test_modules():
    """Test if custom modules can be imported"""
    print("="*60)
    print("Testing Custom Modules...")
    print("="*60)
    
    modules = [
        'document_processor',
        'retrieval_strategies',
        'knowledge_graph',
        'clause_extractor',
        'comparison_engine',
    ]
    
    results = {}
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name:30} - OK")
            results[module_name] = True
        except Exception as e:
            print(f"❌ {module_name:30} - FAILED: {str(e)}")
            results[module_name] = False
    
    print()
    return all(results.values())


def test_document_processor():
    """Test document processor module"""
    print("="*60)
    print("Testing Document Processor...")
    print("="*60)
    
    try:
        from document_processor import HierarchicalDocumentProcessor
        from langchain.schema import Document
        
        # Create sample document
        sample_text = """
ARTICLE I - DEFINITIONS

1.1 General Terms
This Agreement defines the following terms.

1.2 Specific Provisions
(a) Effective Date means January 1, 2024
(b) Party A refers to ABC Corporation

ARTICLE II - PAYMENT TERMS

2.1 Payment Amount
Party A shall pay $100,000.
"""
        
        doc = Document(page_content=sample_text, metadata={'page': 1})
        
        processor = HierarchicalDocumentProcessor()
        result = processor.process_document([doc], "test.txt")
        
        print(f"✅ Document processed successfully")
        print(f"   Sections found: {len(result['structure']['sections'])}")
        print(f"   Chunks created: {len(result['chunks'])}")
        
        if len(result['structure']['sections']) > 0:
            print(f"   Sample section: {result['structure']['sections'][0]['title']}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_knowledge_graph():
    """Test knowledge graph module"""
    print("="*60)
    print("Testing Knowledge Graph...")
    print("="*60)
    
    try:
        from knowledge_graph import KnowledgeGraphBuilder
        from langchain.schema import Document
        
        # Sample data
        doc_data = {
            'name': 'test.pdf',
            'chunks': [
                Document(
                    page_content="This agreement is between ABC Corporation and XYZ Limited, effective January 15, 2024.",
                    metadata={'section': 'Parties', 'page': 1}
                )
            ]
        }
        
        kg_builder = KnowledgeGraphBuilder(use_spacy=False)
        graph = kg_builder.build_from_documents({'test.pdf': doc_data})
        
        print(f"✅ Knowledge graph built successfully")
        print(f"   Nodes: {graph.number_of_nodes()}")
        print(f"   Edges: {graph.number_of_edges()}")
        
        stats = kg_builder.get_graph_statistics(graph)
        print(f"   Entity types: {len(stats['entity_types'])}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Knowledge graph test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_clause_extractor():
    """Test clause extractor module"""
    print("="*60)
    print("Testing Clause Extractor...")
    print("="*60)
    
    try:
        from clause_extractor import ClauseExtractor
        from langchain.schema import Document
        
        # Sample data
        doc_data = {
            'name': 'contract.pdf',
            'chunks': [
                Document(
                    page_content="Either party may terminate this agreement with 30 days written notice.",
                    metadata={'section': 'Termination', 'page': 3}
                ),
                Document(
                    page_content="Payment of $50,000 is due within 30 days of invoice date.",
                    metadata={'section': 'Payment', 'page': 2}
                )
            ]
        }
        
        extractor = ClauseExtractor()
        clauses = extractor.extract_clauses(
            {'contract.pdf': doc_data},
            clause_types=['termination', 'payment']
        )
        
        print(f"✅ Clause extraction successful")
        for clause_type, clause_list in clauses.items():
            print(f"   {clause_type}: {len(clause_list)} clauses found")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Clause extractor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_retrieval():
    """Test retrieval strategies (basic test without API)"""
    print("="*60)
    print("Testing Retrieval Strategies...")
    print("="*60)
    
    try:
        # Just test imports and basic initialization
        from retrieval_strategies import QueryExpander, ReRanker
        
        expander = QueryExpander()
        queries = expander.expand_query("What are the payment terms?")
        
        print(f"✅ Query expansion works")
        print(f"   Original query expanded to {len(queries)} variations")
        print(f"   Example: {queries[0]}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print(" HIERARCHICAL RAG SYSTEM - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_modules),
        ("Document Processor", test_document_processor),
        ("Knowledge Graph", test_knowledge_graph),
        ("Clause Extractor", test_clause_extractor),
        ("Retrieval Strategies", test_retrieval),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Print summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} - {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print()
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Get your Google Gemini API key from https://makersuite.google.com/app/apikey")
        print("2. Run: streamlit run app_hierarchical.py")
        print("3. Enter your API key in the sidebar")
        print("4. Upload documents and start analyzing!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Ensure virtual environment is activated")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check Python version (needs 3.9+)")
    
    print()
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
