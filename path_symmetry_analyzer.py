#!/usr/bin/env python3
"""
Path symmetry analysis for throughput data validation
"""

import warnings
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

def comprehensive_path_analysis(throughput_df, trace_df):
    """
    Comprehensive analysis of path symmetry with different validation modes
    """
    print("🔍 COMPREHENSIVE PATH SYMMETRY ANALYSIS")
    print("=" * 80)
    
    # Get throughput pairs that need reverse filling
    thr_pairs = set(zip(throughput_df['src_site'], throughput_df['dest_site']))
    missing_reverse_pairs = []
    
    for src, dest in thr_pairs:
        reverse_pair = (dest, src)
        if reverse_pair not in thr_pairs:
            missing_reverse_pairs.append((src, dest))
    
    print(f"📊 Throughput pairs needing reverse filling: {len(missing_reverse_pairs)}")
    
    # Analyze each validation mode
    validation_results = {}
    
    for mode in ['strict', 'relaxed', 'lenient']:
        print(f"\n🔄 Testing {mode.upper()} validation:")
        
        symmetric_count = 0
        asymmetric_count = 0
        no_trace_count = 0
        results_detail = []
        
        for src, dest in missing_reverse_pairs[:20]:  # Analyze first 20 for detailed output
            result = analyze_single_pair(src, dest, trace_df, mode)
            results_detail.append(result)
            
            if result['status'] == 'symmetric':
                symmetric_count += 1
            elif result['status'] == 'asymmetric':
                asymmetric_count += 1
            else:
                no_trace_count += 1
        
        print(f"   ✅ Symmetric: {symmetric_count}")
        print(f"   ❌ Asymmetric: {asymmetric_count}") 
        print(f"   ⚠️ No trace data: {no_trace_count}")
        
        validation_results[mode] = {
            'symmetric': symmetric_count,
            'asymmetric': asymmetric_count,
            'no_traces': no_trace_count,
            'details': results_detail
        }
    
    # Show detailed examples for relaxed mode
    print(f"\n🔍 DETAILED ANALYSIS (RELAXED MODE)")
    print("=" * 60)
    
    relaxed_results = validation_results['relaxed']['details']
    
    # Show symmetric examples
    symmetric_examples = [r for r in relaxed_results if r['status'] == 'symmetric'][:3]
    if symmetric_examples:
        print(f"\n✅ SYMMETRIC PATH EXAMPLES:")
        for i, ex in enumerate(symmetric_examples, 1):
            print(f"{i}. {ex['pair'][0]} → {ex['pair'][1]}")
            print(f"   Forward:  {ex['forward_path']} ({ex['forward_count']} traces)")
            print(f"   Reverse:  {ex['reverse_path']} ({ex['reverse_count']} traces)")
            print(f"   Expected: {ex['expected_reverse']}")
            print(f"   Overlap:  {ex.get('overlap_ratio', 'N/A'):.1%}")
    
    # Show asymmetric examples  
    asymmetric_examples = [r for r in relaxed_results if r['status'] == 'asymmetric'][:5]
    if asymmetric_examples:
        print(f"\n❌ ASYMMETRIC PATH EXAMPLES:")
        for i, ex in enumerate(asymmetric_examples, 1):
            print(f"{i}. {ex['pair'][0]} → {ex['pair'][1]}")
            print(f"   Forward:  {ex['forward_path']} ({ex['forward_count']} traces)")
            print(f"   Reverse:  {ex['reverse_path']} ({ex['reverse_count']} traces)")
            print(f"   Expected: {ex['expected_reverse']}")
            if 'differences' in ex:
                print(f"   Issues:   {', '.join(ex['differences'])}")
            if 'overlap_ratio' in ex:
                print(f"   Overlap:  {ex['overlap_ratio']:.1%}")
    
    # Analyze asymmetry patterns
    print(f"\n📊 ASYMMETRY PATTERN ANALYSIS")
    print("-" * 40)
    
    asymmetry_patterns = Counter()
    for result in relaxed_results:
        if result['status'] == 'asymmetric' and 'differences' in result:
            for diff in result['differences']:
                asymmetry_patterns[diff] += 1
    
    for pattern, count in asymmetry_patterns.most_common():
        print(f"   • {pattern}: {count} pairs")
    
    # Create distribution plots
    create_validation_mode_plots(validation_results)
    
    # Create thr_ratio distribution plots for filled data by mode
    create_thr_ratio_plots_by_mode(throughput_df, validation_results)
    
    return validation_results

def analyze_single_pair(src, dest, trace_df, validation_mode):
    """Analyze symmetry for a single site pair"""
    
    # Get traces
    forward_traces = trace_df[
        (trace_df['src_site'] == src) & 
        (trace_df['dest_site'] == dest)
    ]
    reverse_traces = trace_df[
        (trace_df['src_site'] == dest) & 
        (trace_df['dest_site'] == src)
    ]
    
    if forward_traces.empty or reverse_traces.empty:
        return {
            'pair': (src, dest),
            'status': 'no_traces',
            'forward_traces': len(forward_traces),
            'reverse_traces': len(reverse_traces)
        }
    
    # Get ASN path distributions
    forward_asn_paths = forward_traces['asns'].apply(
        lambda x: tuple(asn for asn in x if asn and asn != 0)
    ).value_counts()
    
    reverse_asn_paths = reverse_traces['asns'].apply(
        lambda x: tuple(asn for asn in x if asn and asn != 0)
    ).value_counts()
    
    if forward_asn_paths.empty or reverse_asn_paths.empty:
        return {
            'pair': (src, dest),
            'status': 'no_paths',
            'forward_traces': len(forward_traces),
            'reverse_traces': len(reverse_traces)
        }
    
    # Get most common paths
    forward_common = forward_asn_paths.index[0]
    reverse_common = reverse_asn_paths.index[0]
    forward_count = forward_asn_paths.iloc[0]
    reverse_count = reverse_asn_paths.iloc[0]
    expected_reverse = forward_common[::-1]
    
    # Base result
    result = {
        'pair': (src, dest),
        'forward_path': forward_common,
        'reverse_path': reverse_common,
        'expected_reverse': expected_reverse,
        'forward_count': forward_count,
        'reverse_count': reverse_count,
        'forward_traces': len(forward_traces),
        'reverse_traces': len(reverse_traces)
    }
    
    # Check symmetry based on mode
    if validation_mode == 'strict':
        is_symmetric = forward_common == reverse_common[::-1]
        result['status'] = 'symmetric' if is_symmetric else 'asymmetric'
        if not is_symmetric:
            result['differences'] = compare_paths(forward_common, reverse_common[::-1])
    
    elif validation_mode == 'relaxed':
        # Length check
        len_diff = abs(len(reverse_common) - len(expected_reverse))
        
        # ASN overlap check
        forward_asns = set(forward_common)
        reverse_asns = set(reverse_common)
        overlap = len(forward_asns.intersection(reverse_asns))
        total_unique = len(forward_asns.union(reverse_asns))
        overlap_ratio = overlap / max(total_unique, 1)
        
        is_symmetric = len_diff <= 1 and overlap_ratio >= 0.8
        
        result['status'] = 'symmetric' if is_symmetric else 'asymmetric'
        result['length_diff'] = len_diff
        result['overlap_ratio'] = overlap_ratio
        
        if not is_symmetric:
            differences = []
            if len_diff > 1:
                differences.append(f"Length diff: {len_diff}")
            if overlap_ratio < 0.8:
                differences.append(f"Low overlap: {overlap_ratio:.1%}")
            result['differences'] = differences
    
    elif validation_mode == 'lenient':
        # Path stability check
        forward_stability = forward_count / len(forward_traces)
        reverse_stability = reverse_count / len(reverse_traces)
        
        is_symmetric = forward_stability > 0.5 and reverse_stability > 0.5
        
        result['status'] = 'symmetric' if is_symmetric else 'asymmetric'
        result['forward_stability'] = forward_stability
        result['reverse_stability'] = reverse_stability
        
        if not is_symmetric:
            differences = []
            if forward_stability <= 0.5:
                differences.append(f"Forward unstable: {forward_stability:.1%}")
            if reverse_stability <= 0.5:
                differences.append(f"Reverse unstable: {reverse_stability:.1%}")
            result['differences'] = differences
    
    return result

def compare_paths(path1, path2):
    """Compare two ASN paths and return difference descriptions"""
    differences = []
    
    if len(path1) != len(path2):
        differences.append(f"Length: {len(path1)} vs {len(path2)}")
    
    min_len = min(len(path1), len(path2))
    different_positions = 0
    
    for i in range(min_len):
        if path1[i] != path2[i]:
            different_positions += 1
    
    if different_positions > 0:
        differences.append(f"{different_positions} ASN differences")
    
    if len(path1) > min_len:
        differences.append(f"{len(path1) - min_len} extra ASNs in forward")
    elif len(path2) > min_len:
        differences.append(f"{len(path2) - min_len} extra ASNs in reverse")
    
    return differences

def show_path_diversity(trace_df, top_pairs=10):
    """Show path diversity for top site pairs"""
    print(f"\n🛤️ PATH DIVERSITY ANALYSIS")
    print("=" * 50)
    
    # Get pairs with most traces
    pair_counts = trace_df.groupby(['src_site', 'dest_site']).size().sort_values(ascending=False)
    
    for i, ((src, dest), count) in enumerate(pair_counts.head(top_pairs).items()):
        print(f"\n{i+1}. {src} → {dest} ({count} traces)")
        
        # Get path diversity
        pair_traces = trace_df[
            (trace_df['src_site'] == src) & 
            (trace_df['dest_site'] == dest)
        ]
        
        asn_paths = pair_traces['asns'].apply(
            lambda x: tuple(asn for asn in x if asn and asn != 0)
        ).value_counts()
        
        total_paths = len(asn_paths)
        dominant_path_pct = asn_paths.iloc[0] / count * 100
        
        print(f"   Path variations: {total_paths}")
        print(f"   Dominant path: {dominant_path_pct:.1f}% of traces")
        print(f"   Most common: {asn_paths.index[0]}")
        
        if total_paths > 1:
            print(f"   2nd most common: {asn_paths.index[1]} ({asn_paths.iloc[1]} traces)")

def create_validation_mode_plots(validation_results):
    """Create distribution plots for each validation mode"""
    print(f"\n📊 VALIDATION MODE DISTRIBUTION PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    modes = ['strict', 'relaxed', 'lenient']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    for i, mode in enumerate(modes):
        data = validation_results[mode]
        
        # Create pie chart for each mode
        labels = ['Symmetric', 'Asymmetric', 'No traces']
        sizes = [data['symmetric'], data['asymmetric'], data['no_traces']]
        colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        # Only show non-zero slices
        non_zero_idx = [j for j, size in enumerate(sizes) if size > 0]
        if non_zero_idx:
            labels_filtered = [labels[j] for j in non_zero_idx]
            sizes_filtered = [sizes[j] for j in non_zero_idx] 
            colors_filtered = [colors_pie[j] for j in non_zero_idx]
            
            wedges, texts, autotexts = axes[i].pie(
                sizes_filtered, 
                labels=labels_filtered,
                colors=colors_filtered,
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        axes[i].set_title(f'{mode.title()} Mode\n({sum(sizes)} pairs)', 
                         fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_thr_ratio_plots_by_mode(throughput_df, validation_results):
    """Create thr_ratio distribution plots for filled data, broken down by validation mode"""
    print(f"\n📊 THR_RATIO DISTRIBUTIONS BY VALIDATION MODE")
    print("=" * 70)
    
    # Get filled throughput data
    filled_data = throughput_df[throughput_df.get('throughput_filled', False) == True].copy()
    
    if filled_data.empty:
        print("⚠️ No filled throughput data found")
        return
    
    print(f"Analyzing {len(filled_data)} filled throughput measurements")
    
    # Ensure thr_ratio is numeric
    filled_data['thr_ratio'] = pd.to_numeric(filled_data['thr_ratio'], errors='coerce')
    filled_data = filled_data.dropna(subset=['thr_ratio'])
    
    if filled_data.empty:
        print("⚠️ No valid thr_ratio data found")
        return
    
    # Create plots for each validation mode
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    modes = ['strict', 'relaxed', 'lenient']
    colors = ['#e74c3c', '#4ecdc4', '#45b7d1']
    
    # Get pairs that were considered symmetric in each mode (from the sample analyzed)
    mode_symmetric_pairs = {}
    for mode in modes:
        if mode in validation_results:
            details = validation_results[mode].get('details', [])
            symmetric_pairs = {result['pair'] for result in details if result['status'] == 'symmetric'}
            mode_symmetric_pairs[mode] = symmetric_pairs
        else:
            mode_symmetric_pairs[mode] = set()
    
    all_stats = {}
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        
        # Filter filled data to only pairs that would be considered symmetric in this mode
        symmetric_pairs = mode_symmetric_pairs[mode]
        
        if symmetric_pairs:
            # Create pair column for filtering
            filled_data['pair'] = list(zip(filled_data['src_site'], filled_data['dest_site']))
            mode_data = filled_data[filled_data['pair'].isin(symmetric_pairs)]
        else:
            # If no symmetric pairs identified in sample, use all data
            mode_data = filled_data
        
        if not mode_data.empty:
            series = mode_data['thr_ratio']
            
            # Plot histogram
            ax.hist(series, bins=25, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add statistical lines
            median_val = series.median()
            mean_val = series.mean()
            
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.axvline(mean_val, color='purple', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(1.0, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Perfect (1.0)')
            
            # Statistics
            stats = {
                'count': len(series),
                'median': median_val,
                'mean': mean_val,
                'std': series.std(),
                'near_perfect': len(series[(series >= 0.8) & (series <= 1.2)]) / len(series),
                'realistic': len(series[(series >= 0.5) & (series <= 2.0)]) / len(series)
            }
            all_stats[mode] = stats
            
            # Labels and formatting
            ax.set_xlabel('thr_ratio', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{mode.title()} Mode Filled Data\n({len(series)} measurements)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add text box with key stats
            textstr = f'Near Perfect (0.8-1.2): {stats["near_perfect"]:.1%}\nRealistic (0.5-2.0): {stats["realistic"]:.1%}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
        else:
            ax.text(0.5, 0.5, f'No filled data\nfor {mode} mode', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{mode.title()} Mode - No Data', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 THR_RATIO VALIDATION SUMMARY:")
    print("-" * 50)
    for mode, stats in all_stats.items():
        print(f"{mode.title()} Mode ({stats['count']} measurements):")
        print(f"   Median: {stats['median']:.3f}, Mean: {stats['mean']:.3f} (±{stats['std']:.3f})")
        print(f"   Near Perfect (0.8-1.2): {stats['near_perfect']:.1%}")
        print(f"   Realistic (0.5-2.0): {stats['realistic']:.1%}")
        print()

def analyze_all_path_pairs(trace_df, baseline_df=None):
    """
    Analyze path symmetry for ALL pairs in the dataset regardless of throughput data
    Separate baseline and analysis data if baseline_df is provided
    """
    print(f"\n🌐 COMPREHENSIVE PATH SYMMETRY ANALYSIS (ALL PAIRS)")
    print("=" * 80)
    
    datasets = {}
    if baseline_df is not None:
        datasets['baseline'] = baseline_df
        datasets['analysis'] = trace_df
        print(f"📊 Analyzing baseline ({len(baseline_df)} traces) and analysis ({len(trace_df)} traces) data separately")
    else:
        datasets['all_data'] = trace_df
        print(f"📊 Analyzing all trace data ({len(trace_df)} traces)")
    
    overall_results = {}
    
    for dataset_name, df in datasets.items():
        print(f"\n🔍 DATASET: {dataset_name.upper()}")
        print("-" * 50)
        
        # Get all unique site pairs
        all_pairs = set()
        for _, row in df.iterrows():
            src, dest = row['src_site'], row['dest_site']
            if src != dest:  # Skip self-loops
                all_pairs.add((src, dest))
        
        print(f"Total unique pairs: {len(all_pairs)}")
        
        # Find pairs that exist in both directions
        bidirectional_pairs = []
        unidirectional_pairs = []
        
        processed_pairs = set()
        
        for src, dest in all_pairs:
            if (src, dest) in processed_pairs or (dest, src) in processed_pairs:
                continue
                
            reverse_pair = (dest, src)
            if reverse_pair in all_pairs:
                bidirectional_pairs.append((src, dest))
                processed_pairs.add((src, dest))
                processed_pairs.add(reverse_pair)
            else:
                unidirectional_pairs.append((src, dest))
                processed_pairs.add((src, dest))
        
        print(f"Bidirectional pairs: {len(bidirectional_pairs)}")
        print(f"Unidirectional pairs: {len(unidirectional_pairs)}")
        
        # Analyze bidirectional pairs for symmetry
        if bidirectional_pairs:
            symmetry_analysis = analyze_bidirectional_symmetry(df, bidirectional_pairs)
            overall_results[dataset_name] = {
                'total_pairs': len(all_pairs),
                'bidirectional': len(bidirectional_pairs),
                'unidirectional': len(unidirectional_pairs),
                'symmetry_analysis': symmetry_analysis
            }
            
            # Show symmetry distribution
            create_symmetry_distribution_plot(symmetry_analysis, dataset_name)
        else:
            overall_results[dataset_name] = {
                'total_pairs': len(all_pairs),
                'bidirectional': 0,
                'unidirectional': len(unidirectional_pairs),
                'symmetry_analysis': None
            }
    
    return overall_results

def analyze_bidirectional_symmetry(trace_df, bidirectional_pairs):
    """Analyze symmetry for bidirectional pairs"""
    
    symmetry_stats = {
        'strict': {'symmetric': 0, 'asymmetric': 0, 'no_data': 0},
        'relaxed': {'symmetric': 0, 'asymmetric': 0, 'no_data': 0}, 
        'lenient': {'symmetric': 0, 'asymmetric': 0, 'no_data': 0}
    }
    
    path_length_diffs = []
    overlap_ratios = []
    stability_ratios = []
    
    sample_size = min(len(bidirectional_pairs), 1000)  # Limit for performance
    sample_pairs = bidirectional_pairs[:sample_size]
    
    print(f"   Analyzing {sample_size} bidirectional pairs...")
    
    for src, dest in sample_pairs:
        # Analyze this pair for each mode
        for mode in ['strict', 'relaxed', 'lenient']:
            result = analyze_single_pair(src, dest, trace_df, mode)
            
            if result['status'] == 'symmetric':
                symmetry_stats[mode]['symmetric'] += 1
            elif result['status'] == 'asymmetric':
                symmetry_stats[mode]['asymmetric'] += 1
            else:
                symmetry_stats[mode]['no_data'] += 1
            
            # Collect metrics for distribution analysis
            if mode == 'relaxed' and result['status'] in ['symmetric', 'asymmetric']:
                if 'length_diff' in result:
                    path_length_diffs.append(result['length_diff'])
                if 'overlap_ratio' in result:
                    overlap_ratios.append(result['overlap_ratio'])
            
            if mode == 'lenient' and result['status'] in ['symmetric', 'asymmetric']:
                if 'forward_stability' in result and 'reverse_stability' in result:
                    stability_ratios.append(result['forward_stability'])
                    stability_ratios.append(result['reverse_stability'])
    
    # Print summary
    for mode in ['strict', 'relaxed', 'lenient']:
        stats = symmetry_stats[mode]
        total = sum(stats.values())
        if total > 0:
            print(f"   {mode.title()}: {stats['symmetric']}/{total} ({stats['symmetric']/total:.1%}) symmetric")
    
    return {
        'symmetry_stats': symmetry_stats,
        'path_length_diffs': path_length_diffs,
        'overlap_ratios': overlap_ratios,
        'stability_ratios': stability_ratios,
        'sample_size': sample_size
    }

def create_symmetry_distribution_plot(symmetry_analysis, dataset_name):
    """Create distribution plots for symmetry metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Path Symmetry Distributions - {dataset_name.title()}', fontsize=14, fontweight='bold')
    
    # 1. Validation mode comparison (pie chart)
    ax = axes[0, 0]
    modes = ['Strict', 'Relaxed', 'Lenient']
    symmetric_ratios = []
    
    for mode in ['strict', 'relaxed', 'lenient']:
        stats = symmetry_analysis['symmetry_stats'][mode]
        total = sum(stats.values())
        if total > 0:
            ratio = stats['symmetric'] / total
            symmetric_ratios.append(ratio)
        else:
            symmetric_ratios.append(0)
    
    bars = ax.bar(modes, symmetric_ratios, color=['#e74c3c', '#4ecdc4', '#45b7d1'])
    ax.set_ylabel('Symmetric Ratio')
    ax.set_title('Symmetry by Validation Mode')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, symmetric_ratios):
        if ratio > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{ratio:.1%}', ha='center', fontweight='bold')
    
    # 2. Path length differences
    if symmetry_analysis['path_length_diffs']:
        ax = axes[0, 1]
        ax.hist(symmetry_analysis['path_length_diffs'], bins=20, color='#f39c12', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Path Length Difference')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Path Length Differences')
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No length diff data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Path Length Differences - No Data')
    
    # 3. ASN overlap ratios
    if symmetry_analysis['overlap_ratios']:
        ax = axes[1, 0]
        ax.hist(symmetry_analysis['overlap_ratios'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('ASN Overlap Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of ASN Overlap Ratios')
        ax.axvline(0.8, color='red', linestyle='--', alpha=0.8, label='Threshold (0.8)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No overlap data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('ASN Overlap Ratios - No Data')
    
    # 4. Path stability ratios
    if symmetry_analysis['stability_ratios']:
        ax = axes[1, 1]
        ax.hist(symmetry_analysis['stability_ratios'], bins=20, color='#1abc9c', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Path Stability Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Path Stability Ratios')
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.8, label='Threshold (0.5)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No stability data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Path Stability Ratios - No Data')
    
    plt.tight_layout()
    plt.show()

# Usage for notebook
notebook_usage = '''
# Comprehensive path symmetry analysis
from path_symmetry_analyzer import (
    comprehensive_path_analysis, 
    show_path_diversity, 
    analyze_all_path_pairs
)

# Show path diversity first
show_path_diversity(datasets['trace_df'])

# Original analysis (throughput-based)
analysis = comprehensive_path_analysis(
    datasets['throughput_df'], 
    datasets['trace_df']
)

print("\\n📊 SUMMARY ACROSS VALIDATION MODES:")
for mode, results in analysis.items():
    print(f"{mode.title()}: {results['symmetric']} symmetric, {results['asymmetric']} asymmetric")

# NEW: Analyze ALL path pairs regardless of throughput data
# If you have separate baseline and analysis datasets:
all_pairs_analysis = analyze_all_path_pairs(
    datasets['trace_df'],          # analysis data
    baseline_df=datasets.get('baseline_trace_df', None)  # baseline data (optional)
)

# Or analyze all data together:
# all_pairs_analysis = analyze_all_path_pairs(datasets['trace_df'])
'''

if __name__ == "__main__":
    print("Path Symmetry Analyzer")
    print("=====================")
    print("Analyzes why throughput filling is not finding symmetric paths.")
    print()
    print("Usage in notebook:")
    print(notebook_usage)