#!/usr/bin/env python3
"""
Topology-Performance Correlation Analysis

Correlates routing topology anomalies with performance degradations.
"""

from collections import defaultdict, Counter
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def align_routing_performance_events(analysis_results, performance_df, time_window_minutes=30):
    """
    Align routing anomalies with performance events in time
    
    Parameters:
    -----------
    analysis_results : DataFrame
        Output from graph_based_anomaly_detector with routing anomalies
    performance_df : DataFrame  
        Performance data with flags (owd_anomaly, loss_anomaly, thr_anomaly)
    time_window_minutes : int
        Time window for considering events as co-occurring
        
    Returns:
    --------
    DataFrame with aligned routing and performance events
    """
    print(f"🔗 ALIGNING ROUTING ANOMALIES WITH PERFORMANCE EVENTS")
    print("=" * 60)
    
    # Filter to routing anomalies only
    routing_anomalies = analysis_results[analysis_results['graph_is_anomaly'] == True].copy()
    
    # Filter to performance events only
    perf_events = performance_df[
        (performance_df.get('owd_anomaly', False) == True) |
        (performance_df.get('loss_anomaly', False) == True) |  
        (performance_df.get('thr_anomaly', False) == True)
    ].copy()
    
    print(f"   📊 Data summary:")
    print(f"      • Routing anomalies: {len(routing_anomalies)}")
    print(f"      • Performance events: {len(perf_events)}")
    
    if len(routing_anomalies) == 0 or len(perf_events) == 0:
        print("   ⚠️ No data to correlate")
        return pd.DataFrame()
    
    # Create time window for alignment
    time_delta = timedelta(minutes=time_window_minutes)
    
    aligned_events = []
    
    for _, routing_event in routing_anomalies.iterrows():
        routing_time = routing_event['dt']
        routing_pair = (routing_event['src_site'], routing_event['dest_site'])
        
        # Find performance events within time window for the same site pair
        time_mask = (
            (perf_events['dt'] >= routing_time - time_delta) & 
            (perf_events['dt'] <= routing_time + time_delta)
        )
        pair_mask = (
            (perf_events['src_site'] == routing_pair[0]) & 
            (perf_events['dest_site'] == routing_pair[1])
        )
        
        matching_perf = perf_events[time_mask & pair_mask]
        
        if len(matching_perf) > 0:
            for _, perf_event in matching_perf.iterrows():
                time_diff_minutes = abs((perf_event['dt'] - routing_time).total_seconds()) / 60
                
                aligned_event = {
                    'src_site': routing_pair[0],
                    'dest_site': routing_pair[1], 
                    'routing_time': routing_time,
                    'performance_time': perf_event['dt'],
                    'time_diff_minutes': time_diff_minutes,
                    'routing_anomaly_score': routing_event['graph_anomaly_score'],
                    'owd_anomaly': perf_event.get('owd_anomaly', False),
                    'loss_anomaly': perf_event.get('loss_anomaly', False),
                    'thr_anomaly': perf_event.get('thr_anomaly', False),
                    'owd_ratio': perf_event.get('owd_ratio', np.nan),
                    'loss_rate': perf_event.get('loss', np.nan),
                    'thr_ratio': perf_event.get('thr_ratio', np.nan),
                    'routing_first': routing_time < perf_event['dt']
                }
                aligned_events.append(aligned_event)
    
    if not aligned_events:
        print("   ⚠️ No aligned events found within time window")
        return pd.DataFrame()
        
    aligned_df = pd.DataFrame(aligned_events)
    
    print(f"   ✅ Found {len(aligned_df)} aligned routing-performance events")
    print(f"      • Average time difference: {aligned_df['time_diff_minutes'].mean():.1f} minutes")
    print(f"      • Routing-first events: {aligned_df['routing_first'].sum()} ({aligned_df['routing_first'].mean()*100:.1f}%)")
    
    return aligned_df


def analyze_site_pair_correlation(aligned_events, analysis_results, performance_df):
    """
    Analyze which site pairs show the strongest routing-performance correlation
    """
    print(f"\n📊 SITE PAIR CORRELATION ANALYSIS")
    print("=" * 50)
    
    if len(aligned_events) == 0:
        print("   ⚠️ No aligned events to analyze")
        return {}
    
    # Count events by site pair
    pair_stats = defaultdict(lambda: {
        'routing_anomalies': 0,
        'performance_events': 0, 
        'aligned_events': 0,
        'correlation_rate': 0,
        'avg_routing_score': 0,
        'performance_types': Counter()
    })
    
    # Count routing anomalies by pair
    routing_anomalies = analysis_results[analysis_results['graph_is_anomaly'] == True]
    for _, event in routing_anomalies.iterrows():
        pair = (event['src_site'], event['dest_site'])
        pair_stats[pair]['routing_anomalies'] += 1
        pair_stats[pair]['avg_routing_score'] += event['graph_anomaly_score']
    
    # Count performance events by pair
    perf_events = performance_df[
        (performance_df.get('owd_anomaly', False) == True) |
        (performance_df.get('loss_anomaly', False) == True) |  
        (performance_df.get('thr_anomaly', False) == True)
    ]
    for _, event in perf_events.iterrows():
        pair = (event['src_site'], event['dest_site'])
        pair_stats[pair]['performance_events'] += 1
    
    # Count aligned events and performance types
    for _, event in aligned_events.iterrows():
        pair = (event['src_site'], event['dest_site'])
        pair_stats[pair]['aligned_events'] += 1
        
        if event['owd_anomaly']:
            pair_stats[pair]['performance_types']['owd'] += 1
        if event['loss_anomaly']:
            pair_stats[pair]['performance_types']['loss'] += 1  
        if event['thr_anomaly']:
            pair_stats[pair]['performance_types']['throughput'] += 1
    
    # Calculate correlation rates and averages
    for pair in pair_stats:
        stats = pair_stats[pair]
        if stats['routing_anomalies'] > 0:
            stats['avg_routing_score'] /= stats['routing_anomalies']
            stats['correlation_rate'] = stats['aligned_events'] / stats['routing_anomalies']
    
    # Convert to DataFrame for analysis
    correlation_data = []
    for pair, stats in pair_stats.items():
        if stats['routing_anomalies'] > 0:  # Only pairs with routing anomalies
            correlation_data.append({
                'src_site': pair[0],
                'dest_site': pair[1],
                'pair_name': f"{pair[0]} → {pair[1]}",
                **stats
            })
    
    if not correlation_data:
        print("   ⚠️ No correlation data available")
        return {}
        
    correlation_df = pd.DataFrame(correlation_data)
    correlation_df = correlation_df.sort_values('correlation_rate', ascending=False)
    
    print(f"   📈 Top correlated site pairs:")
    for i, row in correlation_df.head(10).iterrows():
        perf_types = ', '.join([f"{k}:{v}" for k, v in row['performance_types'].items() if v > 0])
        print(f"      {row['pair_name']}: {row['correlation_rate']:.2f} "
              f"({row['aligned_events']}/{row['routing_anomalies']}) - {perf_types}")
    
    return {
        'correlation_df': correlation_df,
        'pair_stats': dict(pair_stats)
    }


def analyze_asn_involvement(aligned_events, analysis_results, cluster_analysis=None):
    """
    Analyze which ASNs are most involved in correlated routing-performance events
    """
    print(f"\n🏢 ASN INVOLVEMENT ANALYSIS")
    print("=" * 40)
    
    if len(aligned_events) == 0:
        print("   ⚠️ No aligned events to analyze")
        return {}
    
    # Get ASN paths for correlated events
    asn_involvement = defaultdict(lambda: {
        'routing_events': 0,
        'performance_correlation': 0,
        'site_pairs': set(),
        'performance_types': Counter()
    })
    
    # Analyze ASNs in routing anomalies that correlate with performance
    correlated_traces = []
    for _, event in aligned_events.iterrows():
        # Find the corresponding routing trace
        routing_mask = (
            (analysis_results['src_site'] == event['src_site']) &
            (analysis_results['dest_site'] == event['dest_site']) &
            (abs((analysis_results['dt'] - event['routing_time']).dt.total_seconds()) < 300)  # 5 min window
        )
        
        matching_traces = analysis_results[routing_mask]
        if len(matching_traces) > 0:
            trace = matching_traces.iloc[0]
            correlated_traces.append(trace)
            
            # Analyze ASNs in this trace
            asn_path = trace.get('asns', [])
            if asn_path:
                clean_asns = [int(asn) for asn in asn_path if asn and asn != 0 and asn < 64512]
                
                for asn in clean_asns:
                    asn_involvement[asn]['performance_correlation'] += 1
                    asn_involvement[asn]['site_pairs'].add((event['src_site'], event['dest_site']))
                    
                    if event['owd_anomaly']:
                        asn_involvement[asn]['performance_types']['owd'] += 1
                    if event['loss_anomaly']:
                        asn_involvement[asn]['performance_types']['loss'] += 1
                    if event['thr_anomaly']:
                        asn_involvement[asn]['performance_types']['throughput'] += 1
    
    # Count total routing events per ASN
    routing_anomalies = analysis_results[analysis_results['graph_is_anomaly'] == True]
    for _, trace in routing_anomalies.iterrows():
        asn_path = trace.get('asns', [])
        if asn_path:
            clean_asns = [int(asn) for asn in asn_path if asn and asn != 0 and asn < 64512]
            for asn in clean_asns:
                asn_involvement[asn]['routing_events'] += 1
    
    # Calculate correlation rates for ASNs
    asn_correlation = []
    for asn, stats in asn_involvement.items():
        if stats['routing_events'] > 0 and stats['performance_correlation'] > 0:
            correlation_rate = stats['performance_correlation'] / stats['routing_events']
            asn_correlation.append({
                'asn': asn,
                'routing_events': stats['routing_events'],
                'performance_correlation': stats['performance_correlation'],
                'correlation_rate': correlation_rate,
                'site_pairs_count': len(stats['site_pairs']),
                'performance_types': dict(stats['performance_types'])
            })
    
    if not asn_correlation:
        print("   ⚠️ No ASN correlation data available")
        return {}
    
    asn_df = pd.DataFrame(asn_correlation)
    asn_df = asn_df.sort_values('performance_correlation', ascending=False)
    
    print(f"   🚨 Top ASNs involved in routing-performance correlations:")
    for i, row in asn_df.head(10).iterrows():
        perf_types = ', '.join([f"{k}:{v}" for k, v in row['performance_types'].items() if v > 0])
        print(f"      AS{row['asn']}: {row['performance_correlation']} events, "
              f"{row['correlation_rate']:.2f} rate, {row['site_pairs_count']} pairs - {perf_types}")
    
    return {
        'asn_correlation_df': asn_df,
        'correlated_traces': correlated_traces
    }


def analyze_causality_patterns(aligned_events):
    """
    Analyze temporal patterns to infer causality between routing and performance
    """
    print(f"\n⏰ CAUSALITY PATTERN ANALYSIS")
    print("=" * 40)
    
    if len(aligned_events) == 0:
        print("   ⚠️ No aligned events to analyze")
        return {}
    
    # Analyze timing patterns
    routing_first = aligned_events[aligned_events['routing_first'] == True]
    performance_first = aligned_events[aligned_events['routing_first'] == False]
    
    print(f"   📊 Temporal patterns:")
    print(f"      • Routing → Performance: {len(routing_first)} events ({len(routing_first)/len(aligned_events)*100:.1f}%)")
    print(f"      • Performance → Routing: {len(performance_first)} events ({len(performance_first)/len(aligned_events)*100:.1f}%)")
    
    # Analyze time differences
    if len(routing_first) > 0:
        routing_first_times = routing_first['time_diff_minutes']
        print(f"      • Avg time: Routing leads by {routing_first_times.mean():.1f} ± {routing_first_times.std():.1f} min")
    
    if len(performance_first) > 0:
        perf_first_times = performance_first['time_diff_minutes'] 
        print(f"      • Avg time: Performance leads by {perf_first_times.mean():.1f} ± {perf_first_times.std():.1f} min")
    
    # Analyze by performance type
    causality_by_type = {}
    for perf_type in ['owd_anomaly', 'loss_anomaly', 'thr_anomaly']:
        type_events = aligned_events[aligned_events[perf_type] == True]
        if len(type_events) > 0:
            routing_leads = (type_events['routing_first'] == True).sum()
            causality_by_type[perf_type] = {
                'total_events': len(type_events),
                'routing_leads': routing_leads,
                'routing_leads_pct': routing_leads / len(type_events) * 100
            }
            
            perf_name = perf_type.replace('_anomaly', '').upper()
            print(f"      • {perf_name}: {routing_leads}/{len(type_events)} routing-first ({routing_leads/len(type_events)*100:.1f}%)")
    
    return {
        'routing_first_events': routing_first,
        'performance_first_events': performance_first,
        'causality_by_type': causality_by_type,
        'overall_routing_leads_pct': len(routing_first) / len(aligned_events) * 100
    }


def create_correlation_visualizations(aligned_events, site_correlation, asn_correlation, causality_analysis):
    """
    Create visualizations for routing-performance correlations
    """
    if len(aligned_events) == 0:
        print("   ⚠️ No data for visualization")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Routing-Performance Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Time difference distribution
    ax1 = axes[0, 0]
    time_diffs = aligned_events['time_diff_minutes']
    ax1.hist(time_diffs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(time_diffs.mean(), color='red', linestyle='--', label=f'Mean: {time_diffs.mean():.1f} min')
    ax1.set_xlabel('Time Difference (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Temporal Alignment of Routing and Performance Events')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Causality patterns
    ax2 = axes[0, 1]
    if 'causality_by_type' in causality_analysis:
        types = ['owd_anomaly', 'loss_anomaly', 'thr_anomaly']
        type_names = ['OWD', 'Loss', 'Throughput']
        routing_leads_pcts = [causality_analysis['causality_by_type'].get(t, {}).get('routing_leads_pct', 0) for t in types]
        
        bars = ax2.bar(type_names, routing_leads_pcts, color=['orange', 'red', 'green'], alpha=0.7)
        ax2.set_ylabel('Routing Leads (%)')
        ax2.set_title('Causality by Performance Type')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, pct in zip(bars, routing_leads_pcts):
            if pct > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{pct:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Site pair correlation heatmap
    ax3 = axes[1, 0]
    if 'correlation_df' in site_correlation and len(site_correlation['correlation_df']) > 0:
        corr_df = site_correlation['correlation_df'].head(15)  # Top 15 pairs
        
        y_pos = np.arange(len(corr_df))
        bars = ax3.barh(y_pos, corr_df['correlation_rate'], color='lightcoral', alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{row['src_site'][:8]}→{row['dest_site'][:8]}" for _, row in corr_df.iterrows()], fontsize=8)
        ax3.set_xlabel('Correlation Rate')
        ax3.set_title('Top Site Pair Correlations')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: ASN involvement
    ax4 = axes[1, 1]
    if 'asn_correlation_df' in asn_correlation and len(asn_correlation['asn_correlation_df']) > 0:
        asn_df = asn_correlation['asn_correlation_df'].head(10)
        
        scatter = ax4.scatter(asn_df['routing_events'], asn_df['performance_correlation'], 
                            s=asn_df['site_pairs_count']*20, alpha=0.6, c=asn_df['correlation_rate'], 
                            cmap='Reds')
        
        for _, row in asn_df.iterrows():
            ax4.annotate(f"AS{row['asn']}", (row['routing_events'], row['performance_correlation']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Total Routing Events')
        ax4.set_ylabel('Performance Correlations')
        ax4.set_title('ASN Involvement (size=site pairs, color=correlation rate)')
        plt.colorbar(scatter, ax=ax4, label='Correlation Rate')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def run_full_topology_performance_analysis(analysis_results, performance_df, cluster_analysis=None, time_window_minutes=30):
    """
    Run complete topology-performance correlation analysis
    
    Parameters:
    -----------
    analysis_results : DataFrame
        Output from graph_based_anomaly_detector
    performance_df : DataFrame
        Performance data with anomaly flags
    cluster_analysis : dict, optional
        Results from cluster analysis
    time_window_minutes : int
        Time window for event alignment
    """
    print(f"\n🔗 COMPREHENSIVE TOPOLOGY-PERFORMANCE CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Step 1: Align routing and performance events in time
    aligned_events = align_routing_performance_events(analysis_results, performance_df, time_window_minutes)
    
    if len(aligned_events) == 0:
        print("❌ No correlated events found - analysis cannot proceed")
        return {}
    
    # Step 2: Analyze site pair correlations
    site_correlation = analyze_site_pair_correlation(aligned_events, analysis_results, performance_df)
    
    # Step 3: Analyze ASN involvement
    asn_correlation = analyze_asn_involvement(aligned_events, analysis_results, cluster_analysis)
    
    # Step 4: Analyze causality patterns
    causality_analysis = analyze_causality_patterns(aligned_events)
    
    # Step 5: Create visualizations
    print(f"\n🎨 Creating correlation visualizations...")
    fig = create_correlation_visualizations(aligned_events, site_correlation, asn_correlation, causality_analysis)
    
    # Summary insights
    print(f"\n💡 KEY INSIGHTS:")
    if causality_analysis.get('overall_routing_leads_pct', 0) > 60:
        print("   • Routing changes typically PRECEDE performance degradation")
        print("   • Hypothesis: Routing instability causes performance issues")
    elif causality_analysis.get('overall_routing_leads_pct', 0) < 40:
        print("   • Performance issues typically PRECEDE routing changes") 
        print("   • Hypothesis: Performance degradation triggers rerouting")
    else:
        print("   • Mixed causality - both patterns present")
        print("   • Hypothesis: Complex feedback loop between routing and performance")
    
    if len(aligned_events) > 0:
        correlation_rate = len(aligned_events) / len(analysis_results[analysis_results['graph_is_anomaly'] == True])
        print(f"   • Overall correlation rate: {correlation_rate:.2f} ({len(aligned_events)} aligned events)")
    
    return {
        'aligned_events': aligned_events,
        'site_correlation': site_correlation,
        'asn_correlation': asn_correlation,
        'causality_analysis': causality_analysis,
        'visualization': fig,
        'summary': {
            'total_aligned_events': len(aligned_events),
            'routing_leads_percentage': causality_analysis.get('overall_routing_leads_pct', 0),
            'top_correlated_asn': asn_correlation.get('asn_correlation_df', pd.DataFrame()).iloc[0]['asn'] if 'asn_correlation_df' in asn_correlation and len(asn_correlation['asn_correlation_df']) > 0 else None
        }
    }


if __name__ == "__main__":
    print("Topology-Performance Correlation Analyzer")
    print("========================================")
    print()
    print("This module correlates routing topology anomalies with performance degradations:")
    print("1. 🔗 Temporal alignment of routing and performance events")
    print("2. 📊 Site pair correlation analysis") 
    print("3. 🏢 ASN involvement analysis")
    print("4. ⏰ Causality pattern detection")
    print("5. 🎨 Comprehensive visualizations")
    print()
    print("Usage:")
    print("  from topology_performance_correlator import run_full_topology_performance_analysis")
    print("  results = run_full_topology_performance_analysis(analysis_results, performance_df)")