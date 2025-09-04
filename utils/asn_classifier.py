#!/usr/bin/env python3
"""
ASN Classification with Elasticsearch Integration

This module provides comprehensive ASN classification using dynamic data
from Elasticsearch ps_asns index containing BGP NSRC REN data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, Optional
import re
from utils.helpers import ConnectES


class ASNClassifier:
    """ASN classifier with Elasticsearch integration"""
    
    def __init__(self):
        self.ren_asns = {}  # ASN -> network name mapping
        self.es_ren_asns = {}
        self._load_ren_asns_from_es()
    
    def _load_ren_asns_from_es(self):
        """Load REN ASNs from Elasticsearch ps_asns index"""
        try:
            es = ConnectES()
            
            # Query to get all ASNs from ps_asns index
            query = {
                "query": {"match_all": {}},
                "size": 10000,  # Get more results
                "_source": ["owner"]
            }
            
            print("🔍 Fetching ASN data from Elasticsearch ps_asns index...")
            
            # Use scroll to get all results
            response = es.search(index="ps_asns", body=query, scroll='5m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            
            all_asns = []
            while hits:
                all_asns.extend(hits)
                # Continue scrolling
                response = es.scroll(scroll_id=scroll_id, scroll='5m')
                hits = response['hits']['hits']
            
            print(f"   📊 Retrieved {len(all_asns)} ASN records from Elasticsearch")
            
            # Process ASN records and identify REN networks
            ren_patterns = [
                # Major REN networks
                r'(?i)(GEANT|JANET|CERN|Internet2|ESnet|AARNET|GARR|RENATER|SWITCH|UNINETT|SURFNET|BELNET|RESTENA|CESNET|PIONIER|GRNET|SANET|ARNES|AMRES|CARNET|SINET|REANNZ|RNP|REUNA|FNAL|DESY|NORDUNET)',
                # Academic institutions
                r'(?i)(University|Universit|Research|Academic|Education|School|College)',
                # REN-specific terms
                r'(?i)(NREN|R&E|REN|-REN)',
                # Educational domains (though might not be in owner field)
                r'(?i)(\.edu|\.ac\.|\.uni\.|\.univ)',
                # Research networks
                r'(?i)(National.*Research|Research.*Network|Education.*Network)',
                # Grid/WLCG
                r'(?i)(WLCG|Grid|LCG)',
                # Major labs and facilities
                r'(?i)(CERN|FERMI|SLAC|KEK|TRIUMF|DESY|FNAL)',
                # Additional patterns for academic/research
                r'(?i)(Institute|Institut|Laboratory|National.*Laboratory)',
                # Country-specific academic patterns
                r'(?i)(Jisc|SUNET|CERNET)',
                # DOE/National lab patterns
                r'(?i)(DOE|National.*Lab|Energy.*Research)',
            ]
            
            ren_count = 0
            for hit in all_asns:
                # ASN is in the document ID
                asn_str = hit['_id']
                source = hit['_source']
                owner = source.get('owner', '')
                
                if not asn_str or not owner:
                    continue
                
                try:
                    asn_num = int(asn_str)
                except (ValueError, TypeError):
                    continue
                
                # Skip reserved/invalid ASNs
                if 'RESERVED' in owner.upper() or 'UNALLOCATED' in owner.upper():
                    continue
                
                # Check if owner matches REN patterns
                is_ren = False
                for pattern in ren_patterns:
                    if re.search(pattern, owner):
                        is_ren = True
                        break
                
                if is_ren:
                    # Clean up owner description for display
                    clean_desc = owner.replace(',', '').strip()
                    if len(clean_desc) > 60:
                        clean_desc = clean_desc[:60] + '...'
                    
                    self.es_ren_asns[asn_num] = clean_desc
                    ren_count += 1
            
            print(f"   ✅ Identified {ren_count} REN/WLCG ASNs from Elasticsearch")
            print(f"   🔗 Total REN database: {len(self.es_ren_asns)} ASNs")
            
            # Use only ES data
            self.ren_asns = self.es_ren_asns
            
        except Exception as e:
            print(f"   ⚠️ Failed to load REN ASNs from Elasticsearch: {e}")
            print(f"   📋 Using empty REN database - classification will be basic only")
            self.ren_asns = {}
            import traceback
            traceback.print_exc()
    
    def classify_asn_type(self, asn) -> str:
        """
        Classify ASN as Research/Education Network (REN/WLCG) or Commodity
        
        Returns:
        --------
        str: 'REN/WLCG (<network_name>)', 'LIKELY_REN', 'COMMODITY', 'PRIVATE', or 'UNKNOWN'
        """
        try:
            asn = int(asn)
        except:
            return 'UNKNOWN'
        
        # Exclude obvious invalid/reserved values
        if asn == 0 or asn < 0 or asn == 4294967295:
            return 'UNKNOWN'
        
        # Check against comprehensive REN database from ES
        if asn in self.ren_asns:
            return f'REN/WLCG ({self.ren_asns[asn]})'
        
        # Check for common university/research patterns by ASN range
        # Many old academic/research ASNs are in lower ranges (1-7000)
        if 100 <= asn <= 7000:
            return 'LIKELY_REN'
        
        # Private ASN ranges per RFC 6996
        if (64512 <= asn <= 65534) or (4200000000 <= asn <= 4294967294):
            return 'PRIVATE'
        
        # Reserved/unassigned ranges
        if asn > 4294967294:
            return 'UNKNOWN'
        
        # Otherwise classify as commodity
        return 'COMMODITY'
    
    def get_ren_asn_stats(self) -> Dict[str, int]:
        """Get statistics about REN ASN database"""
        return {
            'total_ren_asns': len(self.ren_asns),
            'es_ren_asns': len(self.es_ren_asns)
        }
    
    def search_ren_asns(self, pattern: str) -> Dict[int, str]:
        """Search REN ASNs by pattern"""
        results = {}
        pattern_re = re.compile(pattern, re.IGNORECASE)
        
        for asn, name in self.ren_asns.items():
            if pattern_re.search(name):
                results[asn] = name
        
        return results
    
    def get_sample_ren_asns(self, limit: int = 10) -> Dict[int, str]:
        """Get a sample of REN ASNs for display"""
        items = list(self.ren_asns.items())[:limit]
        return dict(items)


# Global classifier instance
_global_classifier = None

def get_asn_classifier() -> ASNClassifier:
    """Get global ASN classifier instance (singleton pattern)"""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = ASNClassifier()
    return _global_classifier

def classify_asn_type(asn) -> str:
    """
    Convenience function for ASN classification
    
    Parameters:
    -----------
    asn : int or str
        ASN number to classify
        
    Returns:
    --------
    str: Classification result
    """
    classifier = get_asn_classifier()
    return classifier.classify_asn_type(asn)


if __name__ == "__main__":
    # Test the classifier
    classifier = ASNClassifier()
    
    # Test some known ASNs
    test_asns = [513, 786, 11537, 7660, 1103, 137, 15169, 32934, 64512, 2000, 5000]
    
    print("🧪 Testing ASN Classification:")
    print("=" * 60)
    
    for asn in test_asns:
        classification = classifier.classify_asn_type(asn)
        print(f"   AS{asn:5d}: {classification}")
    
    stats = classifier.get_ren_asn_stats()
    print(f"\n📊 REN ASN Database Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Show some examples
    sample_asns = classifier.get_sample_ren_asns(5)
    print(f"\n🔍 Sample REN ASNs found:")
    for asn, name in sample_asns.items():
        print(f"   AS{asn}: {name}")
    
    # Search example
    if len(classifier.ren_asns) > 0:
        # Try different search patterns
        for search_term in ["GEANT", "University", "Research", "CERN", "Education"]:
            results = classifier.search_ren_asns(search_term)
            if results:
                print(f"\n🔍 '{search_term}' search results: {len(results)} matches")
                for asn, name in list(results.items())[:3]:
                    print(f"   AS{asn}: {name}")
                if len(results) > 3:
                    print(f"   ... and {len(results)-3} more")
                break