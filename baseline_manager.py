# baseline_manager.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from helpers import ConnectES
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class BaselineManager:
    """
    This class manages baselines for:
    1. Expected path lengths (based on historical successful traces)
    2. Destination reachability flags (never reached in time window)
    3. Expected one-way delays (minimum, percentiles, and statistical baselines)
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
        
        # Set field names based on type
        src_field = f"src_{field_type}"
        dest_field = f"dest_{field_type}"
        
        # Calculate time range
        current_time = datetime.utcnow()
        start_time = current_time - timedelta(days=time_window)
        
        # Query for historical traceroute data
        path_query = {
            "size": 5000,  # Limit to recent measurements
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": current_time.isoformat()
                                }
                            }
                        },
                        {"term": {src_field: src}},
                        {"term": {dest_field: dest}},
                        {"term": {"ipv6": True}},
                        {
                            "exists": {"field": "hops"}
                        }
                    ]
                }
            },
            "_source": [
                "timestamp", "hops", "destination_reached", "path_complete", 
                "n_hops", "ttls", "max_rtt"
            ],
            "sort": [{"timestamp": {"order": "desc"}}]
        }
        
        cache_key = self._get_cache_key("path_length", {
            "src": src, "dest": dest, "field_type": field_type,
            "start_time": start_time.isoformat()
        })
        
        try:
            result = self._execute_cached_query("ps_trace*", path_query, cache_key)
            
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
        
        src_field = f"src_{field_type}"
        dest_field = f"dest_{field_type}"
        
        current_time = datetime.utcnow()
        start_time = current_time - timedelta(days=time_window)
        
        # Query for reachability data
        reachability_query = {
            "size": 1000,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": current_time.isoformat()
                                }
                            }
                        },
                        {"term": {src_field: src}},
                        {"term": {dest_field: dest}},
                        {"term": {"ipv6": True}}
                    ]
                }
            },
            "_source": [
                "timestamp", "destination_reached", "path_complete", 
                "hops", "max_rtt", "n_hops"
            ],
            "sort": [{"timestamp": {"order": "desc"}}]
        }
        
        cache_key = self._get_cache_key("reachability", {
            "src": src, "dest": dest, "field_type": field_type,
            "start_time": start_time.isoformat()
        })
        
        try:
            result = self._execute_cached_query("ps_trace*", reachability_query, cache_key)
            
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
            # Import queryData function
            import sys
            sys.path.append('/Users/petyav/workspace/current/throughput_on_paths')
            from ps_throughput import queryData
            
            baseline_data = queryData(baseline_start_str, baseline_end_str)
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
            # Import queryData function
            import sys
            sys.path.append('/Users/petyav/workspace/current/throughput_on_paths')
            from ps_throughput import queryData
            
            print(f"   📊 Querying baseline data: {baseline_start_str} to {baseline_end_str}")
            baseline_data = queryData(baseline_start_str, baseline_end_str)
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
        print(f"⏱️  Calculating expected OWD: {src} → {dest} ({field_type})")
        
        if field_type not in ["netsite", "host"]:
            raise ValueError(f"field_type must be 'netsite' or 'host', got '{field_type}'")
        
        src_field = f"src_{field_type}"
        dest_field = f"dest_{field_type}"
        
        # Use reference_date if provided, otherwise use current time
        if reference_date is not None:
            end_time = reference_date
        else:
            end_time = datetime.utcnow()
        
        start_time = end_time - timedelta(days=time_window)
        
        # Query for OWD data with aggregations for efficiency
        owd_query = {
            "size": 0,  # Only need aggregations
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        },
                        {"term": {src_field: src}},
                        {"term": {dest_field: dest}},
                        {"term": {"ipv6": True}},
                        {
                            "range": {
                                "delay_median": {
                                    "gt": 0,
                                    "lt": 10000  # Reasonable delay range
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "delay_stats": {
                    "stats": {"field": "delay_median"}
                },
                "delay_percentiles": {
                    "percentiles": {
                        "field": "delay_median",
                        "percents": [5, 10, 25, 50, 75, 90, 95, 99]
                    }
                },
                "delay_histogram": {
                    "histogram": {
                        "field": "delay_median",
                        "interval": 1.0
                    }
                }
            }
        }
        
        cache_key = self._get_cache_key("owd", {
            "src": src, "dest": dest, "field_type": field_type,
            "start_time": start_time.isoformat()
        })
        
        try:
            result = self._execute_cached_query("ps_owd*", owd_query, cache_key)
            
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
            
            # # Method 2: Mode-based approach (most common delay)
            # mode_baseline = mode_delay
            
            # Method 3: Minimum with buffer (min + 10% for noise tolerance)
            buffered_min = min_owd * 1.1
            
            # Choose the best baseline based on data characteristics
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
            
            # Determine baseline quality
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
            
            # Add derived metrics
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