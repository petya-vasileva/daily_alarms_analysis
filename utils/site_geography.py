#!/usr/bin/env python3
"""
Site Geography Module

Fetches and maps site country information from CRIC API for any DataFrame
with src_site and dest_site columns.
"""

import pandas as pd
import requests
import json
import os
from difflib import get_close_matches
import ssl
import urllib3
from . import helpers as hp

def get_known_sites_fallback():
    """
    Fallback dictionary of known sites when CRIC API is unavailable
    
    Returns:
    --------
    dict
        Dictionary mapping site names to country information
    """
    return {
        # CERN and European sites
        'CERN-PROD': {'country': 'Switzerland', 'country_code': 'CH', 'federation': 'CERN', 'tier_level': 0},
        'CERN': {'country': 'Switzerland', 'country_code': 'CH', 'federation': 'CERN', 'tier_level': 0},
        
        # US sites
        'BNL-ATLAS': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 1},
        'FNAL_FERMIGRID': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 1},
        'SLAC': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 1},
        'ANL-LCRC': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 2},
        'UC_SAN_DIEGO-T2': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 2},
        'MIT-T2': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 2},
        'CALTECH-T2': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 2},
        'WISCONSIN-T2': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 2},
        'NEBRASKA-T2': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 2},
        'PURDUE-T2': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 2},
        'FLORIDA-T2': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 2},
        
        # Canadian sites
        'CA-EAST-T2': {'country': 'Canada', 'country_code': 'CA', 'federation': 'Canada-East Federation', 'tier_level': 2},
        'CA-WATERLOO-T2': {'country': 'Canada', 'country_code': 'CA', 'federation': 'Canada-East Federation', 'tier_level': 2},
        'TRIUMF-T1': {'country': 'Canada', 'country_code': 'CA', 'federation': 'Canada-West Federation', 'tier_level': 1},
        'SFU-T2': {'country': 'Canada', 'country_code': 'CA', 'federation': 'Canada-West Federation', 'tier_level': 2},
        
        # French sites
        'IN2P3-CC': {'country': 'France', 'country_code': 'FR', 'federation': 'France-GRILLES', 'tier_level': 1},
        'IN2P3-LAPP-LHCONE': {'country': 'France', 'country_code': 'FR', 'federation': 'France-GRILLES', 'tier_level': 2},
        'IN2P3-CPPM': {'country': 'France', 'country_code': 'FR', 'federation': 'France-GRILLES', 'tier_level': 2},
        'IN2P3-LPC': {'country': 'France', 'country_code': 'FR', 'federation': 'France-GRILLES', 'tier_level': 2},
        'IN2P3-LPSC': {'country': 'France', 'country_code': 'FR', 'federation': 'France-GRILLES', 'tier_level': 2},
        'FR-GRIF_LPNHE': {'country': 'France', 'country_code': 'FR', 'federation': 'France-GRILLES', 'tier_level': 2},
        
        # German sites
        'FZK-LCG2': {'country': 'Germany', 'country_code': 'DE', 'federation': 'Germany', 'tier_level': 1},
        'DESY-HH': {'country': 'Germany', 'country_code': 'DE', 'federation': 'Germany', 'tier_level': 2},
        'RWTH-AACHEN': {'country': 'Germany', 'country_code': 'DE', 'federation': 'Germany', 'tier_level': 2},
        'GOEGRID': {'country': 'Germany', 'country_code': 'DE', 'federation': 'Germany', 'tier_level': 2},
        
        # Italian sites
        'INFN-T1': {'country': 'Italy', 'country_code': 'IT', 'federation': 'Italy', 'tier_level': 1},
        'INFN-ROMA1': {'country': 'Italy', 'country_code': 'IT', 'federation': 'Italy', 'tier_level': 2},
        'INFN-MILANO': {'country': 'Italy', 'country_code': 'IT', 'federation': 'Italy', 'tier_level': 2},
        'INFN-NAPOLI': {'country': 'Italy', 'country_code': 'IT', 'federation': 'Italy', 'tier_level': 2},
        'INFN-PISA': {'country': 'Italy', 'country_code': 'IT', 'federation': 'Italy', 'tier_level': 2},
        
        # Spanish sites
        'PIC': {'country': 'Spain', 'country_code': 'ES', 'federation': 'Spain', 'tier_level': 1},
        'PIC-LHCOPNE': {'country': 'Spain', 'country_code': 'ES', 'federation': 'Spain', 'tier_level': 1},
        'IFCA-LCG2': {'country': 'Spain', 'country_code': 'ES', 'federation': 'Spain', 'tier_level': 2},
        'UAM-LCG2': {'country': 'Spain', 'country_code': 'ES', 'federation': 'Spain', 'tier_level': 2},
        
        # UK sites
        'RAL-LCG2': {'country': 'United Kingdom', 'country_code': 'GB', 'federation': 'UK', 'tier_level': 1},
        'UKI-LT2-IC-HEP': {'country': 'United Kingdom', 'country_code': 'GB', 'federation': 'UK', 'tier_level': 2},
        'UKI-NORTHGRID-MAN-HEP': {'country': 'United Kingdom', 'country_code': 'GB', 'federation': 'UK', 'tier_level': 2},
        'UKI-SCOTGRID-GLASGOW': {'country': 'United Kingdom', 'country_code': 'GB', 'federation': 'UK', 'tier_level': 2},
        
        # Netherlands sites
        'NIKHEF-ELPROD': {'country': 'Netherlands', 'country_code': 'NL', 'federation': 'Netherlands', 'tier_level': 1},
        'SARA-MATRIX': {'country': 'Netherlands', 'country_code': 'NL', 'federation': 'Netherlands', 'tier_level': 2},
        
        # Nordic sites
        'NDGF-T1': {'country': 'Denmark', 'country_code': 'DK', 'federation': 'Nordic', 'tier_level': 1},
        'HELSINKITECH': {'country': 'Finland', 'country_code': 'FI', 'federation': 'Nordic', 'tier_level': 2},
        
        # Eastern European sites
        'PRAGUELCG2': {'country': 'Czech Republic', 'country_code': 'CZ', 'federation': 'Czech Republic', 'tier_level': 2},
        'RRC-KI-T1': {'country': 'Russia', 'country_code': 'RU', 'federation': 'Russia', 'tier_level': 1},
        'JINR-T1': {'country': 'Russia', 'country_code': 'RU', 'federation': 'Russia', 'tier_level': 1},
        
        # Asian sites
        'KEK': {'country': 'Japan', 'country_code': 'JP', 'federation': 'Japan', 'tier_level': 1},
        'TOKYO-LCG2': {'country': 'Japan', 'country_code': 'JP', 'federation': 'Japan', 'tier_level': 2},
        'ASGC': {'country': 'Taiwan', 'country_code': 'TW', 'federation': 'Taiwan', 'tier_level': 2},
        
        # Australian sites
        'AUSTRALIA-ATLAS': {'country': 'Australia', 'country_code': 'AU', 'federation': 'Australia', 'tier_level': 2},
        
        # South American sites
        'SPRACE': {'country': 'Brazil', 'country_code': 'BR', 'federation': 'Brazil', 'tier_level': 2},
        
        # African sites
        'ZA-UJ': {'country': 'South Africa', 'country_code': 'ZA', 'federation': 'South Africa', 'tier_level': 2},
        
        # Additional common variations
        'CERN_PROD': {'country': 'Switzerland', 'country_code': 'CH', 'federation': 'CERN', 'tier_level': 0},
        'BNL_ATLAS': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 1},
        'FNAL-FERMIGRID': {'country': 'United States', 'country_code': 'US', 'federation': 'OSG', 'tier_level': 1},
    }

def save_cric_cache(site_countries, cache_file='site_countries_cache.json'):
    """Save CRIC data to local cache file"""
    try:
        with open(cache_file, 'w') as f:
            json.dump(site_countries, f, indent=2)
        print(f"   💾 Saved {len(site_countries)} sites to cache: {cache_file}")
    except Exception as e:
        print(f"   ⚠️ Failed to save cache: {e}")

def load_cric_cache(cache_file='site_countries_cache.json'):
    """Load CRIC data from local cache file"""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                site_countries = json.load(f)
            print(f"   📂 Loaded {len(site_countries)} sites from cache: {cache_file}")
            return site_countries
    except Exception as e:
        print(f"   ⚠️ Failed to load cache: {e}")
    return None

def get_site_countries(use_cache=True, cache_file='site_countries_cache.json'):
    """
    Fetch site country information from CRIC API with fallback options
    
    Parameters:
    -----------
    use_cache : bool
        Whether to use local cache (default: True)
    cache_file : str
        Cache file name (default: 'site_countries_cache.json')
    
    Returns:
    --------
    dict
        Dictionary mapping site names to country information
    """
    site_countries = {}
    
    # Try to load from cache first
    if use_cache:
        cached_data = load_cric_cache(cache_file)
        if cached_data:
            return cached_data
    
    # Try to fetch from CRIC API
    try:
        print("🌍 Fetching site country information from CRIC API...")
        
        # Disable SSL warnings for self-signed certificates
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        url = "https://wlcg-cric.cern.ch/api/core/federation/query/?json"
        
        # Try with SSL verification disabled
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        
        cric_data = response.json()
        
        for site_name, site_info in cric_data.items():
            site_countries[site_name.upper()] = {
                'country': site_info.get('country', 'Unknown'),
                'country_code': site_info.get('country_code', 'XX'),
                'federation': site_info.get('accounting_name', 'Unknown'),
                'tier_level': site_info.get('tier_level', 0)
            }
        
        print(f"   📋 Loaded {len(site_countries)} sites from CRIC API")
        
        # Save to cache for future use
        if use_cache:
            save_cric_cache(site_countries, cache_file)
        
        return site_countries
        
    except Exception as e:
        print(f"   ⚠️ Error fetching CRIC data: {e}")
    
    # Fallback to known sites
    print("   🔄 Using fallback database of known sites...")
    known_sites = get_known_sites_fallback()
    print(f"   📋 Loaded {len(known_sites)} known sites from fallback")
    
    return known_sites

# Global cache for ps_alarms_meta data
_ps_alarms_cache = None

def load_all_ps_alarms_meta():
    """
    Load all site data from ps_alarms_meta at once and cache it.
    
    Returns:
    --------
    dict
        Dictionary mapping site names to country information
    """
    global _ps_alarms_cache
    
    if _ps_alarms_cache is not None:
        return _ps_alarms_cache
    
    _ps_alarms_cache = {}
    
    try:
        # Query all documents from ps_alarms_meta
        query = {
            "size": 10000,  # Should be more than enough for all sites
            "query": {"match_all": {}},
            "_source": ["site", "netsite", "country"]
        }
        
        response = hp.es.search(index='ps_alarms_meta', body=query)
        hits = response.get('hits', {}).get('hits', [])
        
        print(f"   📥 Loading {len(hits)} entries from ps_alarms_meta...")
        
        for hit in hits:
            source = hit['_source']
            country = source.get('country')
            
            if country and country.strip():
                # Map to our standard format
                country_info = {
                    'country': country.strip(),
                    'country_code': 'XX',
                    'federation': 'Unknown',
                    'tier_level': 0
                }
                
                # Extract country code from common patterns
                if 'United States' in country:
                    country_info['country_code'] = 'US'
                elif 'United Kingdom' in country:
                    country_info['country_code'] = 'GB'
                elif 'Germany' in country:
                    country_info['country_code'] = 'DE'
                elif 'France' in country:
                    country_info['country_code'] = 'FR'
                elif 'Italy' in country:
                    country_info['country_code'] = 'IT'
                elif 'Canada' in country:
                    country_info['country_code'] = 'CA'
                elif 'Brazil' in country:
                    country_info['country_code'] = 'BR'
                elif 'Netherlands' in country:
                    country_info['country_code'] = 'NL'
                elif 'Spain' in country:
                    country_info['country_code'] = 'ES'
                elif 'Poland' in country:
                    country_info['country_code'] = 'PL'
                
                # Store under both site and netsite names (uppercase)
                site = source.get('site')
                netsite = source.get('netsite')
                
                if site:
                    _ps_alarms_cache[site.upper().strip()] = country_info
                if netsite and netsite != site:
                    _ps_alarms_cache[netsite.upper().strip()] = country_info
        
        print(f"   ✅ Cached {len(_ps_alarms_cache)} unique sites from ps_alarms_meta")
        
    except Exception as e:
        print(f"   ⚠️ Error loading ps_alarms_meta: {e}")
        _ps_alarms_cache = {}
    
    return _ps_alarms_cache

def query_ps_alarms_meta_for_site(site_name):
    """
    Query cached ps_alarms_meta data for site information.
    
    Parameters:
    -----------
    site_name : str
        Site name to lookup
        
    Returns:
    --------
    dict or None
        Country information from ps_alarms_meta or None if not found
    """
    if not site_name or not isinstance(site_name, str):
        return None
    
    # Load cache if not already loaded
    ps_cache = load_all_ps_alarms_meta()
    
    # Convert to uppercase for matching
    site_upper = site_name.upper().strip()
    
    return ps_cache.get(site_upper)

def guess_site_country(site_name, site_countries, threshold=0.6):
    """
    Try to guess site country if exact match not found
    
    Parameters:
    -----------
    site_name : str
        Site name to lookup
    site_countries : dict
        Dictionary of known site countries
    threshold : float
        Similarity threshold for fuzzy matching
    
    Returns:
    --------
    dict
        Country information or None if no good match
    """
    # Handle None or empty site names
    if not site_name or site_name is None:
        return None
        
    # Ensure site_name is a string
    if not isinstance(site_name, str):
        site_name = str(site_name)
    
    site_upper = site_name.upper().strip()
    
    # Handle empty strings after processing
    if not site_upper:
        return None
    
    # Direct match in CRIC data
    if site_upper in site_countries:
        return site_countries[site_upper]
    
    # STEP 1: Try ps_alarms_meta first (direct site info with country)
    ps_result = query_ps_alarms_meta_for_site(site_name)
    if ps_result:
        return ps_result
    
    # STEP 2: Try fuzzy matching against CRIC data
    cric_sites = list(site_countries.keys())
    close_matches = get_close_matches(site_upper, cric_sites, n=1, cutoff=threshold)
    
    if close_matches:
        matched_site = close_matches[0]
        # print(f"   🔍 Fuzzy matched '{site_name}' → '{matched_site}'")
        return site_countries[matched_site]
    
    # Try common variations
    variations = [
        site_upper.replace('-', '_'),
        site_upper.replace('_', '-'),
        site_upper + '-T2',
        site_upper + '-T1',
        site_upper.replace('T2', '').replace('T1', '').rstrip('-_')
    ]
    
    for variation in variations:
        if variation in site_countries:
            print(f"   🔄 Variation matched '{site_name}' → '{variation}'")
            return site_countries[variation]
    
    # Substring matching - extract meaningful parts
    # Remove common suffixes/prefixes that aren't site identifiers
    cleanup_patterns = ['LHCONE', 'LHCOPN', 'LCG2', 'T1', 'T2', '-', '_']
    
    # Extract meaningful tokens from the site name
    site_tokens = []
    temp_site = site_upper
    
    # Remove common patterns and split by delimiters
    for pattern in cleanup_patterns:
        temp_site = temp_site.replace(pattern, ' ')
    
    # Split and filter out empty/short tokens
    tokens = [token.strip() for token in temp_site.split() if len(token.strip()) >= 2]
    site_tokens.extend(tokens)
    
    # Also try original site name parts split by delimiters
    original_tokens = []
    for delimiter in ['-', '_']:
        parts = site_upper.split(delimiter)
        for part in parts:
            clean_part = part.strip()
            # Skip common LHC/Grid patterns but keep meaningful identifiers
            if clean_part and clean_part not in ['LHCONE', 'LHCOPN', 'LCG2', 'T1', 'T2']:
                if len(clean_part) >= 2:  # Keep tokens with at least 2 characters
                    original_tokens.append(clean_part)
    
    all_tokens = list(set(site_tokens + original_tokens))  # Remove duplicates
    
    if all_tokens:
        print(f"   🔍 Trying substring matching for '{site_name}' with tokens: {all_tokens}")
        
        # Try to find CRIC sites that contain any of our meaningful tokens
        best_matches = []
        
        for token in all_tokens:
            if len(token) >= 3:  # Focus on longer tokens for better matching
                for cric_site in cric_sites:
                    # Check if token appears in CRIC site name
                    if token in cric_site:
                        confidence = len(token) / len(cric_site)  # Longer matches are better
                        best_matches.append((cric_site, token, confidence))
        
        # Sort by confidence and take best match
        if best_matches:
            best_matches.sort(key=lambda x: x[2], reverse=True)
            best_site, _, confidence = best_matches[0]  # Don't need matched_token
            
            # Only accept matches with reasonable confidence
            if confidence > 0.1:  # At least 10% of the CRIC site name
                return site_countries[best_site]
    
    return None

def test_site_matching(test_sites=None):
    """Test site matching with problem cases."""
    if test_sites is None:
        test_sites = [
            # Sites that should be found in ps_alarms_meta
            'NEBRASKA', 'NEBRASKA-LHCONE', 'UTAH-LHCONE',
            # Sites that need CRIC substring matching
            'NCBJ-LHCOPN', 'praguelcg2-LHCONE', 'KHARKOV-KIPT-LCG2-LHCONE', 
            'CA-SFU-T2-LHCONE', 'RAL-LCG2-LHCOPN', 'CBPF-LHCONE',
            'CA-UVIC-CLOUD-LHCONE', 'CIT_CMS_T2-LHCONE', 'FMPHI-UNIMIB-LHCONE'
        ]
    
    site_countries = get_site_countries()  # Use the correct function name
    print(f"🧪 Testing site matching on {len(test_sites)} problematic sites:")
    print("="*80)
    
    matched = 0
    ps_alarms_matches = 0
    cric_matches = 0
    unmatched_sites = []
    
    for site in test_sites:
        # First test ps_alarms_meta directly
        ps_result = query_ps_alarms_meta_for_site(site)
        if ps_result:
            ps_alarms_matches += 1
            matched += 1
            continue
            
        # Then test full CRIC matching
        result = guess_site_country(site, site_countries)
        if result and result != 'Unknown':
            cric_matches += 1
            matched += 1
        else:
            unmatched_sites.append(site)
    
    print(f"\n📊 Results: {matched}/{len(test_sites)} sites matched ({matched/len(test_sites)*100:.1f}%)")
    print(f"   • ps_alarms_meta: {ps_alarms_matches} sites")
    print(f"   • CRIC (fuzzy/substring): {cric_matches} sites")
    
    if unmatched_sites:
        print(f"\n❌ Unmatched sites ({len(unmatched_sites)}):")
        for site in unmatched_sites:
            print(f"   • {site}")
    else:
        print(f"\n✅ All sites matched!")
    
    return matched, len(test_sites)

def add_geography_to_dataframe(df, src_col='src_site', dest_col='dest_site', site_countries=None):
    """
    Add country information to any DataFrame with source and destination site columns
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with source and destination site columns
    src_col : str
        Name of source site column (default: 'src_site')
    dest_col : str
        Name of destination site column (default: 'dest_site')
    site_countries : dict, optional
        Pre-loaded site country mapping. If None, will fetch from CRIC API
    
    Returns:
    --------
    DataFrame
        DataFrame with added country columns
    """
    # print(f"🗺️  Adding geography to DataFrame with {len(df)} rows...")
    
    # Fetch site countries if not provided
    if site_countries is None:
        site_countries = get_site_countries()
        if not site_countries:
            print("   ❌ Failed to fetch site countries")
            return df
    
    # Validate input columns
    if src_col not in df.columns:
        raise ValueError(f"Source column '{src_col}' not found in DataFrame")
    if dest_col not in df.columns:
        raise ValueError(f"Destination column '{dest_col}' not found in DataFrame")
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Add country columns
    result_df['src_country'] = 'Unknown'
    result_df['src_country_code'] = 'XX'
    result_df['src_federation'] = 'Unknown'
    result_df['src_tier_level'] = 0
    result_df['dest_country'] = 'Unknown'  
    result_df['dest_country_code'] = 'XX'
    result_df['dest_federation'] = 'Unknown'
    result_df['dest_tier_level'] = 0
    result_df['is_international'] = False
    
    # Get unique sites to avoid repeated lookups, filtering out None/NaN values
    src_sites = df[src_col].dropna().unique()
    dest_sites = df[dest_col].dropna().unique()
    unique_sites = set(src_sites) | set(dest_sites)
    
    # Remove any remaining None values and empty strings
    unique_sites = {site for site in unique_sites if site and str(site).strip()}
    
    site_lookup = {}
    unmatched_sites = []
    
    print(f"   🔍 Looking up {len(unique_sites)} unique sites...")
    
    for site in unique_sites:
        country_info = guess_site_country(site, site_countries)
        if country_info:
            site_lookup[site] = country_info
        else:
            site_lookup[site] = {
                'country': 'Unknown',
                'country_code': 'XX', 
                'federation': 'Unknown',
                'tier_level': 0
            }
            unmatched_sites.append(site)
    
    # Fill in country information efficiently using vectorized operations where possible
    for site, info in site_lookup.items():
        if site and info:  # Additional safety check
            # Update source site info
            src_mask = result_df[src_col] == site
            if src_mask.any():
                result_df.loc[src_mask, 'src_country'] = info['country']
                result_df.loc[src_mask, 'src_country_code'] = info['country_code']
                result_df.loc[src_mask, 'src_federation'] = info['federation']
                result_df.loc[src_mask, 'src_tier_level'] = info['tier_level']
            
            # Update destination site info
            dest_mask = result_df[dest_col] == site
            if dest_mask.any():
                result_df.loc[dest_mask, 'dest_country'] = info['country']
                result_df.loc[dest_mask, 'dest_country_code'] = info['country_code']
                result_df.loc[dest_mask, 'dest_federation'] = info['federation']
                result_df.loc[dest_mask, 'dest_tier_level'] = info['tier_level']
    
    # Calculate international flag
    result_df['is_international'] = (result_df['src_country'] != result_df['dest_country']) & \
                                   (result_df['src_country'] != 'Unknown') & \
                                   (result_df['dest_country'] != 'Unknown')
    
    # Report statistics
    matched_sites = len(unique_sites) - len(unmatched_sites)
    print(f"   ✅ Matched {matched_sites}/{len(unique_sites)} sites to countries")
    
    if unmatched_sites:
        print(f"   ❌ Unmatched sites ({len(unmatched_sites)}):")
        for site in unmatched_sites:
            print(f"      • {site}")
    
    # Geography statistics
    international_count = result_df['is_international'].sum()
    print(f"   🌐 International connections: {international_count}/{len(result_df)} ({international_count/len(result_df)*100:.1f}%)")
    
    # Show top country pairs
    if international_count > 0:
        top_country_pairs = result_df[result_df['is_international']].groupby(['src_country', 'dest_country']).size().nlargest(5)
        print(f"   🔗 Top international routes:")
        for (src_country, dest_country), count in top_country_pairs.items():
            print(f"      • {src_country} → {dest_country}: {count}")
    
    return result_df

def get_country_summary(df):
    """
    Generate a summary of country distribution in the DataFrame
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with geography columns added
    
    Returns:
    --------
    dict
        Summary statistics about countries and geography
    """
    if 'src_country' not in df.columns:
        raise ValueError("DataFrame must have geography columns. Run add_geography_to_dataframe() first.")
    
    summary = {
        'total_connections': len(df),
        'international_connections': df['is_international'].sum(),
        'international_percentage': df['is_international'].mean() * 100,
        'unique_countries': len(set(df['src_country'].unique()) | set(df['dest_country'].unique())) - 1,  # -1 for 'Unknown'
        'countries_as_source': df['src_country'].value_counts().to_dict(),
        'countries_as_destination': df['dest_country'].value_counts().to_dict(),
        'top_international_routes': df[df['is_international']].groupby(['src_country', 'dest_country']).size().nlargest(10).to_dict()
    }
    
    return summary