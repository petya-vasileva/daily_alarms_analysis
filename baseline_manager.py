import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple
import warnings
from utils.helpers import ConnectES
from data_queries import query_owd_baseline, query_throughput_baseline, query_trace_baseline

warnings.filterwarnings('ignore')

class BaselineManager:
    """
    Manages baselines for network performance analysis:
    1. Expected path lengths (based on historical successful traces)
    2. Destination reachability flags (never reached in time window)
    3. Expected one-way delays (minimum, percentiles, and statistical baselines)
    4. Throughput baselines (simple and bulk processing)
    """
    
    def __init__(self, cache_enabled=True):
        self.es = ConnectES()
        self.cache_enabled = cache_enabled
        self.cache = {} if cache_enabled else None
        self.query_count = 0
        self.cache_hits = 0
        
    def _get_cache_key(self, query_type: str, params: dict) -> Optional[str]:
        """Generate cache key for queries"""
        if not self.cache_enabled:
            return None
        key_data = f"{query_type}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _execute_cached_query(self, index: str, query: dict, cache_key: Optional[str] = None):
        """Execute query with caching support"""
        if cache_key and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        result = self.es.search(index=index, body=query)
        self.query_count += 1
        
        if cache_key:
            self.cache[cache_key] = result
            
        return result
    
    def get_expected_path_length(self, src: str, dest: str, time_window: int = 7, 
                                field_type: str = "netsite") -> Dict:
        """
        Get expected path length baseline for a source-destination pair
        
        Parameters:
        -----------
        src : str
            Source identifier (netsite or host)
        dest : str
            Destination identifier (netsite or host)
        time_window : int
            Historical data window in days (default: 7)
        field_type : str
            Field type to use: "netsite" or "host" (default: "netsite")
            
        Returns:
        --------
        dict
            Expected path length baseline data including:
            - expected_path_length: median successful path length
            - path_length_stats: comprehensive statistics
            - baseline_quality: quality indicator
            - measurements_count: number of measurements
        """
        print(f"📏 Calculating expected path length: {src} → {dest} ({field_type})")
        
        # Validate field_type
        if field_type not in ["netsite", "host"]:
            raise ValueError(f"field_type must be 'netsite' or 'host', got '{field_type}'")
        
        # Calculate time range
        current_time = datetime.utcnow()
        start_time = current_time - timedelta(days=time_window)
        
        cache_key = self._get_cache_key("path_length", {
            "src": src, "dest": dest, "field_type": field_type,
            "start_time": start_time.isoformat()
        })
        
        try:
            # Use query from data_queries module  
            start_time_iso = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            end_time_iso = current_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            
            result = query_trace_baseline(src, dest, start_time_iso, end_time_iso, field_type, "path_length")
            if cache_key:
                self.cache[cache_key] = result
                self.query_count += 1
            
            if result['hits']['total']['value'] == 0:
                return {
                    'expected_path_length': None,
                    'path_length_stats': None,
                    'baseline_quality': 'no_data',
                    'measurements_count': 0,
                    'timespan_days': time_window,
                    'successful_measurements': 0,
                    'reachability_rate': 0.0
                }
            
            # Process trace data
            path_lengths = []
            successful_paths = []
            total_traces = 0
            successful_traces = 0
            
            for hit in result['hits']['hits']:
                source = hit['_source']
                total_traces += 1
                
                # Get path length
                hops = source.get('hops', [])
                n_hops = source.get('n_hops', len(hops) if isinstance(hops, list) else 0)
                destination_reached = source.get('destination_reached', False)
                
                # Use n_hops if available, otherwise count hops
                path_length = n_hops if n_hops > 0 else (len(hops) if isinstance(hops, list) else 0)
                
                if path_length > 0:
                    path_lengths.append(path_length)
                    
                    # Prioritize successful traces for baseline
                    if destination_reached:
                        successful_paths.append(path_length)
                        successful_traces += 1
            
            if not path_lengths:
                return {
                    'expected_path_length': None,
                    'path_length_stats': None,
                    'baseline_quality': 'invalid_data',
                    'measurements_count': total_traces,
                    'timespan_days': time_window,
                    'successful_measurements': 0,
                    'reachability_rate': 0.0
                }
            
            # Calculate statistics
            path_stats = {
                'min': np.min(path_lengths),
                'max': np.max(path_lengths),
                'mean': np.mean(path_lengths),
                'median': np.median(path_lengths),
                'std': np.std(path_lengths),
                'p25': np.percentile(path_lengths, 25),
                'p75': np.percentile(path_lengths, 75),
                'p95': np.percentile(path_lengths, 95)
            }
            
            # Determine expected path length (prefer successful paths)
            if successful_paths:
                expected_length = np.median(successful_paths)
                baseline_quality = 'good' if len(successful_paths) >= 4 else 'fair'
            else:
                expected_length = np.median(path_lengths)
                baseline_quality = 'poor_reachability'
            
            reachability_rate = successful_traces / total_traces if total_traces > 0 else 0.0
            
            return {
                'expected_path_length': float(expected_length),
                'path_length_stats': path_stats,
                'baseline_quality': baseline_quality,
                'measurements_count': total_traces,
                'timespan_days': time_window,
                'successful_measurements': successful_traces,
                'reachability_rate': reachability_rate
            }
            
        except Exception as e:
            print(f"   ⚠ Error calculating path length baseline: {e}")
            return {
                'expected_path_length': None,
                'path_length_stats': None,
                'baseline_quality': 'error',
                'measurements_count': 0,
                'timespan_days': time_window,
                'successful_measurements': 0,
                'reachability_rate': 0.0,
                'error': str(e)
            }
    
    def get_reachability_flag(self, src: str, dest: str, time_window: int = 7,
                             field_type: str = "netsite") -> Dict:
        """
        Check if destination was never reached in the time window
        
        Parameters:
        -----------
        src : str
            Source identifier
        dest : str
            Destination identifier
        time_window : int
            Historical data window in days (default: 7)
        field_type : str
            Field type to use: "netsite" or "host" (default: "netsite")
            
        Returns:
        --------
        dict
            Reachability analysis including:
            - never_reached: True if destination was never reached
            - reachability_stats: detailed reachability statistics
            - total_attempts: total trace attempts
            - successful_attempts: successful trace attempts
        """
        print(f"🎯 Checking reachability: {src} → {dest} ({field_type})")
        
        if field_type not in ["netsite", "host"]:
            raise ValueError(f"field_type must be 'netsite' or 'host', got '{field_type}'")
        
        current_time = datetime.utcnow()
        start_time = current_time - timedelta(days=time_window)
        
        cache_key = self._get_cache_key("reachability", {
            "src": src, "dest": dest, "field_type": field_type,
            "start_time": start_time.isoformat()
        })
        
        try:
            # Use query from data_queries module
            start_time_iso = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            end_time_iso = current_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            
            result = query_trace_baseline(src, dest, start_time_iso, end_time_iso, field_type, "reachability")
            if cache_key:
                self.cache[cache_key] = result
                self.query_count += 1
            
            if result['hits']['total']['value'] == 0:
                return {
                    'never_reached': True,  # No data means never reached
                    'reachability_stats': None,
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'reachability_rate': 0.0,
                    'baseline_quality': 'no_data'
                }
            
            # Process reachability data
            total_attempts = 0
            successful_attempts = 0
            complete_attempts = 0
            timestamps = []
            max_rtts = []
            
            for hit in result['hits']['hits']:
                source = hit['_source']
                total_attempts += 1
                timestamps.append(source.get('timestamp'))
                
                destination_reached = source.get('destination_reached', False)
                path_complete = source.get('path_complete', False)
                max_rtt = source.get('max_rtt')
                
                if destination_reached:
                    successful_attempts += 1
                
                if path_complete:
                    complete_attempts += 1
                    
                if max_rtt is not None and max_rtt > 0:
                    max_rtts.append(max_rtt)
            
            # Calculate statistics
            reachability_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
            completion_rate = complete_attempts / total_attempts if total_attempts > 0 else 0.0
            
            reachability_stats = {
                'reachability_rate': reachability_rate,
                'completion_rate': completion_rate,
                'avg_rtt': np.mean(max_rtts) if max_rtts else None,
                'median_rtt': np.median(max_rtts) if max_rtts else None,
                'rtt_std': np.std(max_rtts) if len(max_rtts) > 1 else None,
                'first_attempt': min(timestamps) if timestamps else None,
                'last_attempt': max(timestamps) if timestamps else None
            }
            
            # Determine if never reached
            never_reached = successful_attempts == 0
            
            # Determine baseline quality
            if total_attempts == 0:
                baseline_quality = 'no_data'
            elif total_attempts < 3:
                baseline_quality = 'insufficient_data'
            elif reachability_rate == 0:
                baseline_quality = 'unreachable'
            elif reachability_rate < 0.5:
                baseline_quality = 'poor'
            elif reachability_rate < 0.9:
                baseline_quality = 'fair'
            else:
                baseline_quality = 'good'
            
            return {
                'never_reached': never_reached,
                'reachability_stats': reachability_stats,
                'total_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'reachability_rate': reachability_rate,
                'baseline_quality': baseline_quality
            }
            
        except Exception as e:
            print(f"   ⚠ Error checking reachability: {e}")
            return {
                'never_reached': True,  # Assume unreachable on error
                'reachability_stats': None,
                'total_attempts': 0,
                'successful_attempts': 0,
                'reachability_rate': 0.0,
                'baseline_quality': 'error',
                'error': str(e)
            }
    
    def get_simple_throughput_baseline(self, src: str, dest: str, reference_date: datetime, baseline_days: int = 21) -> Dict:
        """
        Get simple throughput baseline for a site pair (sparse data approach)
        
        Parameters:
        -----------
        src : str
            Source site name
        dest : str
            Destination site name
        reference_date : datetime
            Reference date (baseline calculated before this date)
        baseline_days : int
            Days to look back (default: 21)
        
        Returns:
        --------
        dict
            Simple baseline statistics
        """
        # Calculate baseline period
        baseline_end = reference_date
        baseline_start = baseline_end - timedelta(days=baseline_days)
        
        baseline_start_str = baseline_start.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        baseline_end_str = baseline_end.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        # Get baseline data
        try:
            baseline_data = query_throughput_baseline(baseline_start_str, baseline_end_str)
            baseline_df = pd.DataFrame(baseline_data)
            
            if baseline_df.empty:
                return {
                    'baseline_mean_mbps': None,
                    'baseline_median_mbps': None,
                    'baseline_std_mbps': None,
                    'baseline_count': 0,
                    'baseline_quality': 'no_data'
                }
            
            # Process and filter for this pair
            baseline_df['src_site'] = baseline_df['src_site'].str.upper()
            baseline_df['dest_site'] = baseline_df['dest_site'].str.upper()
            baseline_df['value_mbps'] = baseline_df['value'] * 1e-6
            
            pair_data = baseline_df[
                (baseline_df['src_site'] == src.upper()) &
                (baseline_df['dest_site'] == dest.upper())
            ]
            
            if len(pair_data) == 0:
                return {
                    'baseline_mean_mbps': None,
                    'baseline_median_mbps': None,
                    'baseline_std_mbps': None,
                    'baseline_count': 0,
                    'baseline_quality': 'no_data'
                }
            
            # Calculate simple statistics
            values = pair_data['value_mbps'].values
            
            # Determine quality based on data points
            if len(values) >= 15:
                quality = 'good'
            elif len(values) >= 10:
                quality = 'fair'  
            elif len(values) >= 5:
                quality = 'poor'
            else:
                quality = 'insufficient'
            
            return {
                'baseline_mean_mbps': float(np.mean(values)),
                'baseline_median_mbps': float(np.median(values)),
                'baseline_std_mbps': float(np.std(values)),
                'baseline_count': len(values),
                'baseline_quality': quality,
                'baseline_period': f"{baseline_start_str} to {baseline_end_str}"
            }
            
        except Exception as e:
            print(f"   ⚠️ Error getting baseline for {src} → {dest}: {e}")
            return {
                'baseline_mean_mbps': None,
                'baseline_median_mbps': None, 
                'baseline_std_mbps': None,
                'baseline_count': 0,
                'baseline_quality': 'error'
            }
    
    def get_bulk_simple_throughput_baselines(self, pairs: List[Tuple[str, str]], reference_date: datetime, baseline_days: int = 21) -> Dict:
        """
        Get simple throughput baselines for multiple pairs efficiently (single query)
        
        Parameters:
        -----------
        pairs : list of tuples
            List of (src, dest) pairs
        reference_date : datetime
            Reference date (baseline calculated before this date)
        baseline_days : int
            Days to look back (default: 21)
        
        Returns:
        --------
        dict
            Dictionary with (src, dest) keys and baseline statistics as values
        """
        print(f"🚀 Bulk simple throughput baselines for {len(pairs)} pairs")
        
        # Calculate baseline period
        baseline_end = reference_date
        baseline_start = baseline_end - timedelta(days=baseline_days)
        
        baseline_start_str = baseline_start.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        baseline_end_str = baseline_end.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        # Get baseline data ONCE for all pairs
        try:
            baseline_data = query_throughput_baseline(baseline_start_str, baseline_end_str)
            baseline_df = pd.DataFrame(baseline_data)
            
            if baseline_df.empty:
                print("   ⚠️ No baseline data found")
                return {pair: {
                    'baseline_mean_mbps': None,
                    'baseline_median_mbps': None,
                    'baseline_std_mbps': None,
                    'baseline_count': 0,
                    'baseline_quality': 'no_data'
                } for pair in pairs}
            
            # Process baseline data once
            baseline_df['src_site'] = baseline_df['src_site'].str.upper()
            baseline_df['dest_site'] = baseline_df['dest_site'].str.upper()
            baseline_df['value_mbps'] = baseline_df['value'] * 1e-6
            
            print(f"   📈 Processing baselines from {len(baseline_df)} measurements")
            
            # Calculate baselines for each pair
            results = {}
            for src, dest in pairs:
                pair_data = baseline_df[
                    (baseline_df['src_site'] == src.upper()) &
                    (baseline_df['dest_site'] == dest.upper())
                ]
                
                if len(pair_data) == 0:
                    results[(src, dest)] = {
                        'baseline_mean_mbps': None,
                        'baseline_median_mbps': None,
                        'baseline_std_mbps': None,
                        'baseline_count': 0,
                        'baseline_quality': 'no_data'
                    }
                    continue
                
                # Calculate simple statistics
                values = pair_data['value_mbps'].values
                
                # Determine quality based on data points
                if len(values) >= 15:
                    quality = 'good'
                elif len(values) >= 10:
                    quality = 'fair'  
                elif len(values) >= 5:
                    quality = 'poor'
                else:
                    quality = 'insufficient'
                
                results[(src, dest)] = {
                    'baseline_mean_mbps': float(np.mean(values)),
                    'baseline_median_mbps': float(np.median(values)),
                    'baseline_std_mbps': float(np.std(values)),
                    'baseline_count': len(values),
                    'baseline_quality': quality,
                    'baseline_period': f"{baseline_start_str} to {baseline_end_str}"
                }
            
            return results
            
        except Exception as e:
            print(f"   ⚠️ Error getting bulk baselines: {e}")
            return {pair: {
                'baseline_mean_mbps': None,
                'baseline_median_mbps': None, 
                'baseline_std_mbps': None,
                'baseline_count': 0,
                'baseline_quality': 'error'
            } for pair in pairs}
    

    def get_expected_owd(self, src: str, dest: str, time_window: int = 7,
                        field_type: str = "netsite", reference_date: Optional[datetime] = None) -> Dict:
        """
        Get expected one-way delay baseline for a source-destination pair
        
        Parameters:
        -----------
        src : str
            Source identifier
        dest : str
            Destination identifier
        time_window : int
            Historical data window in days (default: 7)
        field_type : str
            Field type to use: "netsite" or "host" (default: "netsite")
        reference_date : datetime, optional
            Reference date for baseline calculation. If None, uses current time.
            Baseline will be calculated from (reference_date - time_window) to reference_date.
            
        Returns:
        --------
        dict
            Expected OWD baseline data including:
            - min_owd: minimum observed delay
            - expected_owd: statistical baseline delay
            - owd_stats: comprehensive delay statistics
            - baseline_quality: quality indicator
        """
        if field_type not in ["netsite", "host"]:
            raise ValueError(f"field_type must be 'netsite' or 'host', got '{field_type}'")
        
        # Use reference_date if provided, otherwise use current time
        if reference_date is not None:
            end_time = reference_date
        else:
            end_time = datetime.utcnow()
        
        start_time = end_time - timedelta(days=time_window)
        
        # Convert to ISO format for query
        start_time_iso = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        end_time_iso = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        cache_key = self._get_cache_key("owd", {
            "src": src, "dest": dest, "field_type": field_type,
            "start_time": start_time_iso
        })
        
        try:
            # Use query from data_queries module
            result = query_owd_baseline(src, dest, start_time_iso, end_time_iso, field_type)
            if cache_key:
                self.cache[cache_key] = result
                self.query_count += 1
            
            if result['aggregations']['delay_stats']['count'] == 0:
                return {
                    'min_owd': None,
                    'expected_owd': None,
                    'owd_stats': None,
                    'baseline_quality': 'no_data',
                    'measurements_count': 0,
                    'timespan_days': time_window
                }
            
            # Extract aggregation results
            stats = result['aggregations']['delay_stats']
            percentiles = result['aggregations']['delay_percentiles']['values']
            histogram = result['aggregations']['delay_histogram']['buckets']
            
            # Build comprehensive statistics
            owd_stats = {
                'count': int(stats['count']),
                'min': stats['min'],
                'max': stats['max'],
                'avg': stats['avg'],
                'sum': stats['sum'],
                'p5': percentiles['5.0'],
                'p10': percentiles['10.0'],
                'p25': percentiles['25.0'],
                'p50': percentiles['50.0'],  # median
                'p75': percentiles['75.0'],
                'p90': percentiles['90.0'],
                'p95': percentiles['95.0'],
                'p99': percentiles['99.0']
            }
            
            # Calculate mode from histogram (most common delay)
            if histogram:
                mode_bucket = max(histogram, key=lambda x: x['doc_count'])
                mode_delay = mode_bucket['key']
            else:
                mode_delay = owd_stats['p50']  # fallback to median
            
            # Determine expected OWD using multiple approaches
            min_owd = stats['min']
            
            # Method 1: Statistical approach (5th percentile as baseline)
            statistical_baseline = percentiles['5.0']
            
            # Method 3: Minimum with buffer (min + 10% for noise tolerance)
            buffered_min = min_owd * 1.1

            if owd_stats['count'] >= 100:
                # Sufficient data: use 5th percentile
                expected_owd = statistical_baseline
                baseline_method = 'p5_statistical'
            elif owd_stats['count'] >= 20:
                # Moderate data: use buffered minimum
                expected_owd = buffered_min
                baseline_method = 'buffered_minimum'
            else:
                # Limited data: use minimum
                expected_owd = min_owd
                baseline_method = 'minimum'

            if owd_stats['count'] >= 100:
                baseline_quality = 'excellent'
            elif owd_stats['count'] >= 50:
                baseline_quality = 'good'
            elif owd_stats['count'] >= 20:
                baseline_quality = 'fair'
            elif owd_stats['count'] >= 5:
                baseline_quality = 'poor'
            else:
                baseline_quality = 'insufficient'
            
            # derived metrics
            owd_stats.update({
                'mode': mode_delay,
                'baseline_method': baseline_method,
                'coefficient_of_variation': (np.sqrt((owd_stats['p75'] - owd_stats['p25'])**2) / owd_stats['avg']) if owd_stats['avg'] > 0 else None,
                'iqr': owd_stats['p75'] - owd_stats['p25'],
                'outlier_threshold_high': owd_stats['p75'] + 1.5 * (owd_stats['p75'] - owd_stats['p25']),
                'outlier_threshold_low': max(0, owd_stats['p25'] - 1.5 * (owd_stats['p75'] - owd_stats['p25']))
            })
            
            return {
                'min_owd': float(min_owd),
                'expected_owd': float(expected_owd),
                'owd_stats': owd_stats,
                'baseline_quality': baseline_quality,
                'measurements_count': int(owd_stats['count']),
                'timespan_days': time_window
            }
            
        except Exception as e:
            print(f"   ⚠ Error calculating OWD baseline: {e}")
            return {
                'min_owd': None,
                'expected_owd': None,
                'owd_stats': None,
                'baseline_quality': 'error',
                'measurements_count': 0,
                'timespan_days': time_window,
                'error': str(e)
            }
    
  
    def get_bulk_baselines(self, pairs: List[Tuple[str, str]], time_window: int = 7,
                          field_type: str = "netsite", max_workers: int = 10, 
                          reference_date: Optional[datetime] = None, 
                          include_throughput: bool = True, throughput_window: int = 21) -> pd.DataFrame:
        """
        Get baselines for multiple pairs efficiently using parallel processing
        
        Parameters:
        -----------
        pairs : list of tuples
            List of (src, dest) pairs
        time_window : int
            Historical data window in days (default: 7)
        field_type : str
            Field type to use: "netsite" or "host" (default: "netsite")
        max_workers : int
            Maximum concurrent threads (default: 10)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with comprehensive baselines for all pairs
        """
        import concurrent.futures
        from tqdm import tqdm
        
        print(f"📊 Getting bulk baselines for {len(pairs)} pairs")
        print(f"   Time window: {time_window} days")
        print(f"   Field type: {field_type}")
        print(f"   Max workers: {max_workers}")
        
        def process_pair(pair):
            src, dest = pair
            return self.get_comprehensive_baseline(src, dest, time_window, field_type, reference_date, 
                                                include_throughput, throughput_window)
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(pairs), desc="Processing pairs") as pbar:
                future_to_pair = {executor.submit(process_pair, pair): pair for pair in pairs}
                
                for future in concurrent.futures.as_completed(future_to_pair):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        pair = future_to_pair[future]
                        print(f"   ⚠ Error processing {pair}: {e}")
                        results.append({
                            'src': pair[0], 'dest': pair[1], 'error': str(e),
                            'overall_baseline_quality': 'error'
                        })
                    finally:
                        pbar.update(1)
        
        df = pd.DataFrame(results)
        
        # Summary statistics
        quality_summary = df['overall_baseline_quality'].value_counts()
        print(f"\n✅ Baseline processing complete:")
        for quality, count in quality_summary.items():
            print(f"   • {quality}: {count} pairs")
        
        print(f"   • Cache efficiency: {self.cache_hits}/{self.query_count} queries cached")
        
        return df
    

def get_throughput_baseline_for_pairs(unique_pairs, reference_date, baseline_days=21):
    """
    Get throughput baselines for all pairs efficiently
    
    Parameters:
    -----------
    unique_pairs : DataFrame
        DataFrame with src_site and dest_site columns
    reference_date : datetime
        Reference date (baseline calculated before this date)
    baseline_days : int
        Days to look back (default: 21)
    
    Returns:
    --------
    dict
        Dictionary with (src_site, dest_site) keys and baseline statistics as values
    """
    print(f"📊 Getting baselines for {len(unique_pairs)} pairs efficiently...")
    
    pair_list = [(row['src_site'], row['dest_site']) for _, row in unique_pairs.iterrows()]
    
    manager = BaselineManager()
    baseline_results = manager.get_bulk_simple_throughput_baselines(pair_list, reference_date, baseline_days)
    
    return baseline_results

def get_throughput_with_baselines(date_from_str, date_to_str, baseline_days=21, max_workers=10):
    """
    Get throughput data with baselines using parallel processing
    
    Parameters:
    -----------
    date_from_str : str
        Start date for throughput analysis (ISO format)
    date_to_str : str
        End date for throughput analysis (ISO format)  
    baseline_days : int
        Days to look back for baseline (default: 21)
    max_workers : int
        Maximum concurrent threads for baseline calculation (default: 10)
    Returns:
    --------
    pandas.DataFrame
        DataFrame with throughput data and baseline columns
    """
    
    print(f"🚀 Getting throughput data with baselines")
    print(f"   Analysis period: {date_from_str} to {date_to_str}")
    print(f"   Baseline lookback: {baseline_days} days")
    print(f"   Max workers: {max_workers}")
    
    # Get current throughput data
    print("\n📊 Querying current throughput data...")
    current_data = query_throughput_baseline(date_from_str, date_to_str)
    current_df = pd.DataFrame(current_data)
    
    if current_df.empty:
        print("   ⚠️ No current throughput data found")
        return pd.DataFrame()
    
    # Process current data
    current_df['dt'] = pd.to_datetime(current_df['from'], unit='ms')
    current_df['dt_str'] = current_df['dt'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    current_df['src_site'] = current_df['src_site'].str.upper()
    current_df['dest_site'] = current_df['dest_site'].str.upper()
    current_df['value_mbps'] = current_df['value'] * 1e-6
    
    booleanDictionary = {True: 'ipv6', False: 'ipv4'}
    current_df['ipv'] = current_df['ipv6'].map(booleanDictionary)
    
    print(f"   📈 Found {len(current_df)} current measurements")
    
    # Get unique pairs
    unique_pairs = current_df[['src_site', 'dest_site']].drop_duplicates()
    print(f"   📈 Unique site pairs: {len(unique_pairs)}")
    
    # Parse reference date
    ref_date = datetime.strptime(date_from_str, '%Y-%m-%dT%H:%M:%S.000Z')
    
    # Calculate baselines for all pairs efficiently
    print(f"\n🎯 Calculating baselines for {len(unique_pairs)} pairs efficiently...")
    baseline_results = get_throughput_baseline_for_pairs(unique_pairs, ref_date, baseline_days)
    
    # Add baseline columns to current data
    print("\n🔗 Adding baselines to current data...")
    
    baseline_columns = ['baseline_mean_mbps', 'baseline_median_mbps', 'baseline_std_mbps', 
                       'baseline_count', 'baseline_quality']
    
    # Initialize columns with proper data types
    current_df['baseline_mean_mbps'] = pd.Series(dtype='float64')
    current_df['baseline_median_mbps'] = pd.Series(dtype='float64')
    current_df['baseline_std_mbps'] = pd.Series(dtype='float64')
    current_df['baseline_count'] = pd.Series(dtype='int64')
    current_df['baseline_quality'] = pd.Series(dtype='object')
    
    # Fill baseline values with proper type conversion
    for idx, row in current_df.iterrows():
        pair_key = (row['src_site'], row['dest_site'])
        if pair_key in baseline_results:
            baseline = baseline_results[pair_key]
            for col in baseline_columns:
                value = baseline.get(col)
                if col in ['baseline_mean_mbps', 'baseline_median_mbps', 'baseline_std_mbps'] and value is not None:
                    current_df.at[idx, col] = float(value)
                elif col == 'baseline_count' and value is not None:
                    current_df.at[idx, col] = int(value)
                else:
                    current_df.at[idx, col] = value
    
    # Calculate comparison metrics
    print("\n📈 Calculating comparison metrics...")
    
    # Convert baseline columns to proper numeric types
    numeric_baseline_cols = ['baseline_mean_mbps', 'baseline_median_mbps', 'baseline_std_mbps']
    for col in numeric_baseline_cols:
        current_df[col] = pd.to_numeric(current_df[col], errors='coerce')
    current_df['baseline_count'] = pd.to_numeric(current_df['baseline_count'], errors='coerce').astype('Int64')
    
    # Only for pairs with valid baselines
    valid_baseline_mask = (current_df['baseline_mean_mbps'].notna()) & (current_df['baseline_mean_mbps'] > 0)
    
    # Initialize comparison columns with proper types
    current_df['vs_baseline_mean_pct'] = pd.Series(dtype='float64')
    current_df['vs_baseline_median_pct'] = pd.Series(dtype='float64')
    current_df['vs_baseline_zscore'] = pd.Series(dtype='float64')
    current_df['is_anomaly'] = False
    
    if valid_baseline_mask.any():
        # Percentage change from baseline mean
        current_df.loc[valid_baseline_mask, 'vs_baseline_mean_pct'] = (
            (current_df.loc[valid_baseline_mask, 'value_mbps'] - 
             current_df.loc[valid_baseline_mask, 'baseline_mean_mbps']) / 
            current_df.loc[valid_baseline_mask, 'baseline_mean_mbps'] * 100
        ).round(2)
        
        # Percentage change from baseline median  
        current_df.loc[valid_baseline_mask, 'vs_baseline_median_pct'] = (
            (current_df.loc[valid_baseline_mask, 'value_mbps'] - 
             current_df.loc[valid_baseline_mask, 'baseline_median_mbps']) / 
            current_df.loc[valid_baseline_mask, 'baseline_median_mbps'] * 100
        ).round(2)
        
        # Z-score (only if std > 0)
        std_mask = valid_baseline_mask & (current_df['baseline_std_mbps'] > 0)
        if std_mask.any():
            current_df.loc[std_mask, 'vs_baseline_zscore'] = (
                (current_df.loc[std_mask, 'value_mbps'] - 
                 current_df.loc[std_mask, 'baseline_mean_mbps']) / 
                current_df.loc[std_mask, 'baseline_std_mbps']
            ).round(2)
        
        # Flag anomalies (>50% change from baseline mean for good quality baselines)
        anomaly_mask = (
            valid_baseline_mask & 
            (current_df['vs_baseline_mean_pct'].abs() >= 50) & 
            (current_df['baseline_quality'].isin(['good', 'fair']))
        )
        current_df.loc[anomaly_mask, 'is_anomaly'] = True
    
    # Summary
    print(f"\n✅ Analysis complete:")
    print(f"   • Total measurements: {len(current_df)}")
    print(f"   • Pairs with baselines: {valid_baseline_mask.sum()}")
    print(f"   • Anomalies detected: {current_df['is_anomaly'].sum()}")
    
    if valid_baseline_mask.any():
        quality_summary = current_df[valid_baseline_mask]['baseline_quality'].value_counts()
        print(f"\n📊 Baseline quality distribution:")
        for quality, count in quality_summary.items():
            print(f"   • {quality}: {count} measurements")
        
        # Show some anomalies
        anomalies = current_df[current_df['is_anomaly'] == True]
        if not anomalies.empty:
            print(f"\n🚨 Sample anomalies (top 5 by deviation):")
            sample_anomalies = anomalies.nlargest(5, 'vs_baseline_mean_pct', keep='first')[
                ['src_site', 'dest_site', 'value_mbps', 'baseline_mean_mbps', 'vs_baseline_mean_pct', 'baseline_quality']
            ]
            for _, row in sample_anomalies.iterrows():
                print(f"   • {row['src_site']} → {row['dest_site']}: {row['value_mbps']:.1f} Mbps vs {row['baseline_mean_mbps']:.1f} Mbps baseline ({row['vs_baseline_mean_pct']:+.1f}%)")
    
    return current_df

def test_throughput_baselines(date_from=None, date_to=None, baseline_days=21, max_workers=10):
    """Test the throughput baseline analysis with timing"""
    
    # Use provided dates or default test period
    if date_from is None:
        date_from = '2025-08-08T03:00:00.000Z'
    if date_to is None:
        date_to = '2025-08-08T06:00:00.000Z'
    
    print("="*60)
    print("TESTING THROUGHPUT BASELINE ANALYSIS")
    print("="*60)
    
    # Test with efficient processing
    start_time = time.time()
    throughput_df = get_throughput_with_baselines(date_from, date_to, baseline_days, max_workers)
    end_time = time.time()
    
    print(f"\n✅ Test completed in {end_time - start_time:.2f} seconds")
    
    if not throughput_df.empty:
        print(f"📋 Result: {len(throughput_df)} measurements with baselines")
        
        # Show sample results
        sample_cols = ['src_site', 'dest_site', 'value_mbps', 'baseline_mean_mbps', 
                      'baseline_quality', 'vs_baseline_mean_pct', 'is_anomaly']
        available_cols = [col for col in sample_cols if col in throughput_df.columns]
        sample_df = throughput_df[available_cols].head(10)
        print(f"\n📊 Sample results:")
        print(sample_df.to_string(index=False))
        
        # Show summary stats
        if 'baseline_quality' in throughput_df.columns:
            quality_counts = throughput_df['baseline_quality'].value_counts()
            print(f"\n🎯 Baseline quality distribution:")
            for quality, count in quality_counts.items():
                print(f"   • {quality}: {count}")
        
        if 'is_anomaly' in throughput_df.columns:
            anomaly_count = throughput_df['is_anomaly'].sum()
            print(f"\n🚨 Anomalies detected: {anomaly_count}")
        
    else:
        print("❌ No data found in test period")
        
    return throughput_df