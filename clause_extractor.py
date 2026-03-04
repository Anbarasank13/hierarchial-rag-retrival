"""
Clause Extractor
Specialized extraction of legal clauses from documents
"""

import re
from typing import List, Dict, Any
from langchain_core.documents import Document


class ClauseExtractor:
    """
    Extract and categorize legal clauses from documents
    """
    
    # Define common clause types and their patterns
    CLAUSE_TYPES = {
        'termination': {
            'keywords': [
                'termination', 'terminate', 'cancellation', 'cancel',
                'end this agreement', 'conclude', 'cessation'
            ],
            'patterns': [
                r'(?:termination|terminate).*?(?:with|upon|after)\s+(\d+)\s+days',
                r'either party may terminate.*?notice',
                r'this agreement (?:shall|may) be terminated',
            ]
        },
        'payment': {
            'keywords': [
                'payment', 'pay', 'fee', 'compensation', 'remuneration',
                'invoice', 'billing', 'cost', 'price'
            ],
            'patterns': [
                r'(?:pay|payment).*?(\$[\d,]+(?:\.\d{2})?)',
                r'within\s+(\d+)\s+days.*?(?:payment|invoice)',
                r'(?:fee|price|cost).*?(\$[\d,]+)',
            ]
        },
        'confidentiality': {
            'keywords': [
                'confidential', 'confidentiality', 'non-disclosure', 'NDA',
                'proprietary', 'secret', 'private information'
            ],
            'patterns': [
                r'confidential information.*?shall not',
                r'non-disclosure.*?obligation',
                r'proprietary.*?information',
            ]
        },
        'liability': {
            'keywords': [
                'liability', 'liable', 'indemnification', 'indemnify',
                'damages', 'loss', 'harm', 'injury'
            ],
            'patterns': [
                r'(?:liability|liable).*?(?:limited to|capped at|not exceed)',
                r'indemnif(?:y|ication).*?against',
                r'in no event.*?liable',
            ]
        },
        'intellectual_property': {
            'keywords': [
                'intellectual property', 'IP', 'copyright', 'patent',
                'trademark', 'trade secret', 'proprietary rights'
            ],
            'patterns': [
                r'intellectual property.*?(?:rights|ownership)',
                r'copyright.*?(?:belongs to|owned by)',
                r'(?:patent|trademark).*?rights',
            ]
        },
        'governing_law': {
            'keywords': [
                'governing law', 'jurisdiction', 'applicable law',
                'governed by', 'subject to the laws'
            ],
            'patterns': [
                r'governed by.*?laws? of\s+(\w+)',
                r'jurisdiction of.*?courts? of\s+(\w+)',
                r'subject to.*?laws? of\s+(\w+)',
            ]
        },
        'dispute_resolution': {
            'keywords': [
                'dispute', 'arbitration', 'mediation', 'litigation',
                'resolution', 'disagreement'
            ],
            'patterns': [
                r'dispute.*?(?:resolved|settled).*?(?:through|by)\s+(\w+)',
                r'arbitration.*?in accordance with',
                r'mediation.*?before.*?litigation',
            ]
        },
        'force_majeure': {
            'keywords': [
                'force majeure', 'act of god', 'unforeseen circumstances',
                'beyond control', 'unavoidable'
            ],
            'patterns': [
                r'force majeure.*?event',
                r'act of god.*?(?:including|such as)',
                r'circumstances beyond.*?control',
            ]
        },
        'warranty': {
            'keywords': [
                'warranty', 'warrantee', 'guarantee', 'representation',
                'assurance', 'covenant'
            ],
            'patterns': [
                r'warrant(?:y|ies).*?that',
                r'represent(?:s|ation).*?and warrant',
                r'guarantee.*?(?:performance|quality)',
            ]
        },
        'term_duration': {
            'keywords': [
                'term', 'duration', 'period', 'effective date',
                'commencement', 'expiration'
            ],
            'patterns': [
                r'term.*?(?:of|shall be)\s+(\d+)\s+(?:years?|months?|days?)',
                r'effective.*?(?:from|as of)\s+([A-Za-z]+\s+\d+,?\s+\d{4})',
                r'period of\s+(\d+)\s+(?:years?|months?)',
            ]
        },
    }
    
    def extract_clauses(
        self, 
        documents: Dict[str, Any],
        clause_types: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract clauses from documents
        
        Args:
            documents: Dictionary of processed documents
            clause_types: Specific clause types to extract (None = all types)
            
        Returns:
            Dictionary mapping clause types to extracted clauses
        """
        
        if clause_types is None:
            clause_types = list(self.CLAUSE_TYPES.keys())
        
        results = {clause_type: [] for clause_type in clause_types}
        
        for doc_name, doc_data in documents.items():
            for clause_type in clause_types:
                clauses = self._extract_clause_type(
                    doc_data, 
                    clause_type,
                    doc_name
                )
                results[clause_type].extend(clauses)
        
        return results
    
    def _extract_clause_type(
        self, 
        doc_data: Dict[str, Any], 
        clause_type: str,
        doc_name: str
    ) -> List[Dict[str, Any]]:
        """Extract specific clause type from a document"""
        
        if clause_type not in self.CLAUSE_TYPES:
            return []
        
        clause_config = self.CLAUSE_TYPES[clause_type]
        keywords = clause_config['keywords']
        patterns = clause_config['patterns']
        
        extracted_clauses = []
        
        for chunk in doc_data['chunks']:
            content = chunk.page_content
            content_lower = content.lower()
            
            # Check if any keyword appears in chunk
            has_keyword = any(kw in content_lower for kw in keywords)
            
            if has_keyword:
                # Extract using patterns
                matched_patterns = []
                extracted_values = []
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                    
                    for match in matches:
                        matched_patterns.append(pattern)
                        # Extract specific values if pattern has groups
                        if match.groups():
                            extracted_values.extend([g for g in match.groups() if g])
                
                # Create clause entry
                clause = {
                    'type': clause_type,
                    'content': content,
                    'source': doc_name,
                    'section': chunk.metadata.get('section', 'Unknown'),
                    'page': chunk.metadata.get('page', 'N/A'),
                    'keywords_found': [kw for kw in keywords if kw in content_lower],
                    'patterns_matched': len(matched_patterns),
                    'extracted_values': extracted_values,
                    'confidence': self._calculate_confidence(
                        len([kw for kw in keywords if kw in content_lower]),
                        len(matched_patterns)
                    )
                }
                
                extracted_clauses.append(clause)
        
        # Deduplicate and sort by confidence
        extracted_clauses = self._deduplicate_clauses(extracted_clauses)
        extracted_clauses.sort(key=lambda x: x['confidence'], reverse=True)
        
        return extracted_clauses
    
    def _calculate_confidence(self, keyword_count: int, pattern_count: int) -> float:
        """
        Calculate confidence score for clause extraction
        
        Score is based on:
        - Number of keywords found
        - Number of patterns matched
        """
        keyword_score = min(keyword_count * 0.2, 0.5)
        pattern_score = min(pattern_count * 0.3, 0.5)
        
        return min(keyword_score + pattern_score, 1.0)
    
    def _deduplicate_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate clauses based on content similarity"""
        
        if not clauses:
            return []
        
        unique_clauses = []
        seen_contents = set()
        
        for clause in clauses:
            # Use first 100 characters as fingerprint
            fingerprint = clause['content'][:100].lower().strip()
            
            if fingerprint not in seen_contents:
                unique_clauses.append(clause)
                seen_contents.add(fingerprint)
        
        return unique_clauses
    
    def compare_clauses(
        self,
        clauses1: List[Dict[str, Any]],
        clauses2: List[Dict[str, Any]],
        clause_type: str
    ) -> Dict[str, Any]:
        """
        Compare clauses of the same type from two documents
        
        Returns:
            Comparison results showing similarities and differences
        """
        
        comparison = {
            'clause_type': clause_type,
            'doc1_clauses': len(clauses1),
            'doc2_clauses': len(clauses2),
            'similarities': [],
            'differences': [],
            'unique_to_doc1': [],
            'unique_to_doc2': []
        }
        
        # Simple text-based comparison
        doc1_texts = {c['content'][:200] for c in clauses1}
        doc2_texts = {c['content'][:200] for c in clauses2}
        
        # Find similarities (clauses appearing in both)
        similar_texts = doc1_texts & doc2_texts
        comparison['similarities'] = [
            {'text': text[:100] + '...'} for text in similar_texts
        ]
        
        # Find unique clauses
        unique_doc1 = doc1_texts - doc2_texts
        unique_doc2 = doc2_texts - doc1_texts
        
        comparison['unique_to_doc1'] = [
            {'text': text[:100] + '...'} for text in unique_doc1
        ]
        
        comparison['unique_to_doc2'] = [
            {'text': text[:100] + '...'} for text in unique_doc2
        ]
        
        return comparison
    
    def extract_key_terms(
        self, 
        clause: Dict[str, Any],
        term_types: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        Extract key terms from a clause
        
        Term types:
        - dates
        - amounts
        - durations
        - parties
        - obligations
        """
        
        if term_types is None:
            term_types = ['dates', 'amounts', 'durations', 'parties']
        
        content = clause['content']
        terms = {}
        
        if 'dates' in term_types:
            date_patterns = [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
            ]
            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, content, re.IGNORECASE))
            terms['dates'] = list(set(dates))
        
        if 'amounts' in term_types:
            amount_pattern = r'(\$[\d,]+(?:\.\d{2})?)'
            amounts = re.findall(amount_pattern, content)
            terms['amounts'] = list(set(amounts))
        
        if 'durations' in term_types:
            duration_pattern = r'(\d+)\s+(days?|weeks?|months?|years?)'
            durations = re.findall(duration_pattern, content, re.IGNORECASE)
            terms['durations'] = [f"{num} {unit}" for num, unit in durations]
        
        if 'parties' in term_types:
            party_pattern = r'\b(Party\s+[A-Z]|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:LLC|Inc\.|Corp\.|Corporation|Company|Ltd\.))\b'
            parties = re.findall(party_pattern, content)
            terms['parties'] = list(set(parties))
        
        return terms
    
    def generate_clause_summary(self, clause: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of a clause
        """
        
        clause_type = clause['type'].replace('_', ' ').title()
        source = clause['source']
        section = clause['section']
        
        # Extract key terms
        key_terms = self.extract_key_terms(clause)
        
        summary_parts = [
            f"**{clause_type} Clause**",
            f"Source: {source}, Section: {section}",
        ]
        
        if key_terms.get('parties'):
            summary_parts.append(f"Parties: {', '.join(key_terms['parties'][:3])}")
        
        if key_terms.get('amounts'):
            summary_parts.append(f"Amounts: {', '.join(key_terms['amounts'][:3])}")
        
        if key_terms.get('durations'):
            summary_parts.append(f"Durations: {', '.join(key_terms['durations'][:3])}")
        
        if key_terms.get('dates'):
            summary_parts.append(f"Dates: {', '.join(key_terms['dates'][:3])}")
        
        # Add snippet of content
        content_snippet = clause['content'][:150].strip()
        summary_parts.append(f"\nContent: {content_snippet}...")
        
        return "\n".join(summary_parts)


# Example usage
def demo_clause_extraction():
    """Demonstrate clause extraction"""
    from langchain.schema import Document
    
    # Sample document
    sample_doc = {
        'name': 'sample_contract.pdf',
        'chunks': [
            Document(
                page_content="""
                TERMINATION CLAUSE
                Either party may terminate this Agreement upon thirty (30) days 
                written notice to the other party. In the event of termination, 
                Party A shall pay Party B $5,000 as termination fee.
                """,
                metadata={'section': '5. Termination', 'page': 3}
            ),
            Document(
                page_content="""
                PAYMENT TERMS
                Party A agrees to pay Party B the sum of $100,000 within 
                thirty (30) days of invoice date. Late payments shall incur 
                interest at 5% per annum.
                """,
                metadata={'section': '2. Payment', 'page': 1}
            ),
            Document(
                page_content="""
                CONFIDENTIALITY
                Each party agrees to maintain the confidentiality of all 
                proprietary information disclosed by the other party during 
                the term of this Agreement and for three (3) years thereafter.
                """,
                metadata={'section': '7. Confidentiality', 'page': 4}
            ),
        ],
        'structure': {},
        'metadata': {}
    }
    
    # Extract clauses
    extractor = ClauseExtractor()
    documents = {'sample_contract.pdf': sample_doc}
    
    clauses = extractor.extract_clauses(
        documents,
        clause_types=['termination', 'payment', 'confidentiality']
    )
    
    # Display results
    for clause_type, clause_list in clauses.items():
        print(f"\n{'='*60}")
        print(f"{clause_type.upper()} CLAUSES ({len(clause_list)} found)")
        print('='*60)
        
        for clause in clause_list:
            print(extractor.generate_clause_summary(clause))
            print(f"Confidence: {clause['confidence']:.2f}")
            print()


if __name__ == "__main__":
    demo_clause_extraction()
