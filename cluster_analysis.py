#!/usr/bin/env python3
"""
Cluster analysis module for investigating anomalous routing patterns
"""

from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Optional ASN classifier
try:
    from utils.asn_classifier import classify_asn_type
except ImportError:
    def classify_asn_type(asn) -> str:
        """Basic fallback ASN classification"""
        try:
            asn = int(asn)
        except:
            return 'UNKNOWN'
            
        basic_ren_asns = {
            513: 'CERN', 786: 'JANET-UK', 1103: 'SURFNET-Netherlands',
            137: 'GARR-Italy', 11537: 'Internet2', 7660: 'AARNET-Australia'
        }
        
        if asn in basic_ren_asns:
            return f'REN/WLCG ({basic_ren_asns[asn]})'
        if 1000 <= asn <= 7000:
            return 'LIKELY_REN'
        return 'COMMODITY'


def is_public_asn(asn) -> bool:
    """Return True if ASN is a public (non-private, non-reserved) ASN.

    Considers both 16-bit and 32-bit ASN private ranges per IANA/RFC:
      - Public 16-bit: 1..64511
      - Private 16-bit: 64512..65534
      - 65535 is reserved
      - Public 32-bit: 65536..4199999999
      - Private 32-bit: 4200000000..4294967294
      - 0 and 4294967295 are reserved/invalid
    """
    try:
        if asn is None:
            return False
        asn = int(asn)
    except Exception:
        return False

    # Exclude obvious invalid/reserved values
    if asn == 0 or asn < 0 or asn == 4294967295:
        return False

    # 16-bit public
    if 1 <= asn <= 64511:
        return True

    # 16-bit private or reserved
    if 64512 <= asn <= 65535:
        return False

    # 32-bit public range (but exclude the 32-bit private block)
    if 65536 <= asn <= 4294967294:
        if 4200000000 <= asn <= 4294967294:
            return False
        return True

    return False

def analyze_cluster_paths(cluster_results, analysis_traces, baseline_traces):
    """
    Extract and analyze ASN paths from clusters, comparing against baseline
    
    Parameters:
    -----------
    cluster_results : dict
        Results from extract_anomaly_clusters()
    analysis_traces : DataFrame  
        Analysis period trace data
    baseline_traces : DataFrame
        Baseline period trace data for comparison
    
    Returns:
    --------
    dict with detailed cluster path analysis including baseline comparison
    """
    
    print(f"\n🔍 ANALYZING ASN PATHS IN CLUSTERS (vs BASELINE)")
    print("=" * 60)
    
    # First, build baseline routing patterns by site pair
    print("📊 Building baseline routing patterns...")
    baseline_patterns = {}
    
    for _, trace in baseline_traces.iterrows():
        if 'asns' in trace and trace['asns'] is not None:
            site_pair = (trace['src_site'], trace['dest_site'])
            # remove private and invalid ASNs
            # TODO: fillin the 0 ASNs by mapping to IPs (as in ps_asn_anomalies.py)
            clean_asns = [asn for asn in trace['asns'] if is_public_asn(asn)]
            
            if clean_asns and site_pair:
                if site_pair not in baseline_patterns:
                    baseline_patterns[site_pair] = Counter()
                baseline_patterns[site_pair][tuple(clean_asns)] += 1
    
    print(f"   ✅ Found baseline patterns for {len(baseline_patterns)} site pairs")
    
    cluster_paths = {}
    
    # Process top anomalous clusters
    for i, (cluster_id, info) in enumerate(cluster_results['sorted_clusters'][:3]):
        print(f"\n" + "="*80)
        print(f"🚨 CLUSTER {cluster_id} (Rank #{i+1}) - {info['anomaly_rate']:.1%} anomalous")
        print(f"   Size: {info['size']} paths")
        print("="*80)
        
        # Get actual trace indices for this cluster
        sample_indices = info['sample_indices']
        cluster_traces = analysis_traces.iloc[sample_indices]
        
        # Group cluster traces by site pair
        cluster_by_site_pair = {}
        for _, trace in cluster_traces.iterrows():
                if 'asns' in trace and trace['asns'] is not None:
                    site_pair = (trace['src_site'], trace['dest_site'])
                    clean_asns = [asn for asn in trace['asns'] if is_public_asn(asn)]
                
                if clean_asns and site_pair:
                    if site_pair not in cluster_by_site_pair:
                        cluster_by_site_pair[site_pair] = Counter()
                    cluster_by_site_pair[site_pair][tuple(clean_asns)] += 1
        
        # Analyze each site pair in the cluster
        site_pair_analysis = {}
        
        for site_pair, anomaly_paths in cluster_by_site_pair.items():
            print(f"\n📍 SITE PAIR: {site_pair[0]} → {site_pair[1]}")
            print("-" * 60)
            
            # Get baseline paths for this site pair
            baseline_paths = baseline_patterns.get(site_pair, Counter())
            
            if baseline_paths:
                print(f"🟢 NORMAL PATHS (baseline):")
                total_baseline = sum(baseline_paths.values())
                for j, (path, count) in enumerate(baseline_paths.most_common(3)):
                    pct = count / total_baseline * 100
                    path_str = ' → '.join(map(str, path))
                    print(f"   #{j+1} {path_str} ({count} traces, {pct:.1f}%)")
            else:
                print(f"   ⚠️ No baseline data for this site pair")
            
            print(f"\n🔴 ANOMALOUS PATHS (cluster):")
            total_anomaly = sum(anomaly_paths.values())
            for j, (path, count) in enumerate(anomaly_paths.most_common(3)):
                pct = count / total_anomaly * 100
                path_str = ' → '.join(map(str, path))
                print(f"   #{j+1} {path_str} ({count} traces, {pct:.1f}%)")
            
            # Identify routing changes
            if baseline_paths:
                baseline_set = set(baseline_paths.keys())
                anomaly_set = set(anomaly_paths.keys())
                
                new_paths = anomaly_set - baseline_set
                disappeared_paths = baseline_set - anomaly_set
                common_paths = baseline_set & anomaly_set
                
                print(f"\n🚨 ROUTING CHANGES:")
                if new_paths:
                    print(f"   🆕 NEW PATHS ({len(new_paths)}):")
                    for path in list(new_paths)[:2]:
                        print(f"      → {' → '.join(map(str, path))}")
                        
                if disappeared_paths:
                    print(f"   🚫 DISAPPEARED PATHS ({len(disappeared_paths)}):")
                    for path in list(disappeared_paths)[:2]:
                        print(f"      → {' → '.join(map(str, path))}")
                        
                if common_paths:
                    print(f"   🔄 COMMON PATHS: {len(common_paths)} (frequency may have changed)")
                
                # Detect specific patterns
                patterns = detect_routing_patterns(baseline_paths, anomaly_paths)
                if patterns:
                    print(f"\n🔍 DETECTED PATTERNS:")
                    for pattern in patterns:
                        print(f"   • {pattern}")
            
            site_pair_analysis[site_pair] = {
                'baseline_paths': baseline_paths,
                'anomaly_paths': anomaly_paths,
                'total_anomaly_traces': total_anomaly
            }
        
        cluster_paths[cluster_id] = {
            'rank': i + 1,
            'size': info['size'],
            'anomaly_rate': info['anomaly_rate'],
            'site_pairs': site_pair_analysis,
            'traces': cluster_traces
        }
    
    return cluster_paths

def detect_routing_patterns(baseline_paths, anomaly_paths):
    """
    Detect common routing change patterns
    """
    patterns = []
    
    # Check for path inflation (ASN repetition)
    for path in anomaly_paths.keys():
        if len(path) != len(set(path)):  # Duplicates exist
            duplicates = [asn for asn in set(path) if path.count(asn) > 1]
            patterns.append(f"Path inflation: AS{duplicates[0]} repeated {path.count(duplicates[0])} times")
    
    # Check for path length changes
    if baseline_paths:
        avg_baseline_len = sum(len(path) * count for path, count in baseline_paths.items()) / sum(baseline_paths.values())
        avg_anomaly_len = sum(len(path) * count for path, count in anomaly_paths.items()) / sum(anomaly_paths.values())
        
        if avg_anomaly_len > avg_baseline_len + 0.5:
            patterns.append(f"Path lengthening: {avg_baseline_len:.1f} → {avg_anomaly_len:.1f} hops avg")
        elif avg_anomaly_len < avg_baseline_len - 0.5:
            patterns.append(f"Path shortening: {avg_baseline_len:.1f} → {avg_anomaly_len:.1f} hops avg")
    
    return patterns

def identify_routing_providers(cluster_paths):
    """
    Identify which ASNs/providers are most involved in anomalous clusters
    Include classification of REN/WLCG vs Commodity networks
    """
    
    print(f"\n🏢 ROUTING PROVIDER ANALYSIS")
    print("=" * 50)
    
    # Count ASN involvement across all clusters
    asn_involvement = defaultdict(int)
    asn_cluster_count = defaultdict(int)
    asn_classifications = {}
    
    for cluster_id, data in cluster_paths.items():
        cluster_asns = set()
        # Iterate through site pairs and their anomaly paths
        for site_pair, site_data in data['site_pairs'].items():
            for path, count in site_data['anomaly_paths'].items():
                for asn in path:
                    asn_involvement[asn] += count
                    cluster_asns.add(asn)
                    if asn not in asn_classifications:
                        asn_classifications[asn] = classify_asn_type(asn)
        
        for asn in cluster_asns:
            asn_cluster_count[asn] += 1
    
    # Sort by involvement
    top_asns = sorted(asn_involvement.items(), key=lambda x: x[1], reverse=True)
    
    # Analyze by network type
    ren_asns = []
    commodity_asns = []
    
    for asn, count in top_asns:
        classification = asn_classifications[asn]
        if 'REN/WLCG' in classification or 'LIKELY_REN' in classification:
            ren_asns.append((asn, count, classification))
        else:
            commodity_asns.append((asn, count, classification))
    
    print(f"🚨 TOP ASNs IN ANOMALOUS CLUSTERS:")
    for i, (asn, count) in enumerate(top_asns[:15]):
        clusters = asn_cluster_count[asn]
        classification = asn_classifications[asn]
        icon = "🎓" if 'REN' in classification else "💼"
        print(f"   #{i+1:2d} {icon} AS{asn:5d}: {count:3d} anomalous paths in {clusters} clusters - {classification}")
    
    print(f"\n📊 NETWORK TYPE BREAKDOWN:")
    print(f"   🎓 Research/Education Networks (REN/WLCG): {len(ren_asns)}")
    print(f"   💼 Commodity Networks: {len(commodity_asns)}")
    
    if ren_asns:
        print(f"\n🎓 TOP REN/WLCG ASNs IN ANOMALIES:")
        for i, (asn, count, classification) in enumerate(ren_asns[:5]):
            clusters = asn_cluster_count[asn]
            print(f"   #{i+1} AS{asn}: {count} paths in {clusters} clusters - {classification}")
    
    if commodity_asns:
        print(f"\n💼 TOP COMMODITY ASNs IN ANOMALIES:")
        for i, (asn, count, classification) in enumerate(commodity_asns[:5]):
            clusters = asn_cluster_count[asn]
            print(f"   #{i+1} AS{asn}: {count} paths in {clusters} clusters")
    
    return {
        'asn_involvement': dict(asn_involvement),
        'asn_cluster_count': dict(asn_cluster_count),
        'asn_classifications': asn_classifications,
        'top_asns': top_asns,
        'ren_asns': ren_asns,
        'commodity_asns': commodity_asns
    }

def cluster_summary_report(cluster_results, cluster_paths, provider_analysis):
    """
    Generate a comprehensive summary report
    """
    
    print(f"\n📋 CLUSTER ANALYSIS SUMMARY REPORT")
    print("=" * 60)
    
    print(f"🔍 CLUSTERING OVERVIEW:")
    print(f"   • Total clusters found: {cluster_results['n_clusters']}")
    print(f"   • Noise points: {cluster_results['n_noise']}")
    print(f"   • Analyzed top clusters: {len(cluster_paths)}")
    
    print(f"\n🚨 MOST PROBLEMATIC AREAS:")
    
    # Site pairs with highest anomaly concentration
    all_site_pairs = {}
    for cluster_id, data in cluster_paths.items():
        for (src, dest), site_data in data['site_pairs'].items():
            key = f"{src} → {dest}"
            if key not in all_site_pairs:
                all_site_pairs[key] = {'count': 0, 'clusters': set()}
            all_site_pairs[key]['count'] += site_data['total_anomaly_traces']
            all_site_pairs[key]['clusters'].add(cluster_id)
    
    # Sort by involvement
    problematic_pairs = sorted(
        all_site_pairs.items(), 
        key=lambda x: (len(x[1]['clusters']), x[1]['count']), 
        reverse=True
    )
    
    print(f"   TOP PROBLEMATIC SITE PAIRS:")
    for i, (pair, info) in enumerate(problematic_pairs[:5]):
        clusters_str = ', '.join(map(str, sorted(info['clusters'])))
        print(f"   #{i+1} {pair}: {info['count']} anomalous paths in clusters {clusters_str}")
    
    # ASN analysis
    print(f"\n   TOP PROBLEMATIC ASNs:")
    for i, (asn, count) in enumerate(provider_analysis['top_asns'][:5]):
        clusters = provider_analysis['asn_cluster_count'][asn]
        print(f"   #{i+1} AS{asn}: {count} anomalous paths in {clusters} clusters")
    
    print(f"\n💡 NETWORK TYPE INSIGHTS:")
    ren_count = len(provider_analysis['ren_asns'])
    commodity_count = len(provider_analysis['commodity_asns'])
    total_asns = ren_count + commodity_count
    
    if total_asns > 0:
        ren_percentage = (ren_count / total_asns) * 100
        print(f"   🎓 REN/WLCG networks: {ren_count}/{total_asns} ({ren_percentage:.1f}%) of problematic ASNs")
        print(f"   💼 Commodity networks: {commodity_count}/{total_asns} ({100-ren_percentage:.1f}%) of problematic ASNs")
        
        if commodity_count > ren_count:
            print(f"   ⚠️ Commodity networks dominate anomalies - expected behavior")
        elif ren_count > commodity_count:
            print(f"   🚨 Unusual: REN networks showing more anomalies than commodity")

    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Investigate AS{provider_analysis['top_asns'][0][0]} routing policies")
    print(f"   2. Check {problematic_pairs[0][0]} performance correlation")
    print(f"   3. Monitor clusters {list(cluster_paths.keys())[:3]} for future anomalies")
    
    if provider_analysis['commodity_asns']:
        top_commodity = provider_analysis['commodity_asns'][0][0]
        print(f"   4. Focus on commodity AS{top_commodity} for performance optimization")
    
    if provider_analysis['ren_asns']:
        top_ren = provider_analysis['ren_asns'][0][0]
        print(f"   5. Investigate REN AS{top_ren} for potential infrastructure issues")
    
    return {
        'problematic_pairs': problematic_pairs,
        'summary': {
            'total_clusters': cluster_results['n_clusters'],
            'analyzed_clusters': len(cluster_paths),
            'top_asn': provider_analysis['top_asns'][0][0] if provider_analysis['top_asns'] else None,
            'top_site_pair': problematic_pairs[0][0] if problematic_pairs else None
        }
    }

def run_full_cluster_analysis(cluster_results, analysis_traces, baseline_traces):
    """
    Run complete cluster analysis pipeline
    
    Parameters:
    -----------
    cluster_results : dict
        Results from extract_anomaly_clusters()
    analysis_traces : DataFrame  
        Analysis period trace data
    baseline_traces : DataFrame
        Baseline period trace data for comparison
    """
    
    print(f"\n🎯 STARTING COMPREHENSIVE CLUSTER ANALYSIS")
    print("=" * 70)
    
    # Step 1: Analyze cluster paths with baseline comparison
    cluster_paths = analyze_cluster_paths(cluster_results, analysis_traces, baseline_traces)
    
    # Step 2: Identify routing providers  
    provider_analysis = identify_routing_providers(cluster_paths)
    
    # Step 3: Generate summary report
    summary = cluster_summary_report(cluster_results, cluster_paths, provider_analysis)
    
    return {
        'cluster_paths': cluster_paths,
        'provider_analysis': provider_analysis,
        'summary': summary
    }