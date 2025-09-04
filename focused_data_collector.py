#!/usr/bin/env python3
"""
Focused Network Data Collector

Collects network performance and routing data for specific site pairs only.
Filters queries at the database level to minimize data transfer and processing.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from baseline_manager import BaselineManager
import utils.helpers as hp

class FocusedDataCollector:
    def __init__(self, date_from_str, date_to_str, focus_pairs, baseline_days=21, include_reverse_pairs=False):
        """
        Initialize focused data collector
        
        Parameters:
        -----------
        date_from_str : str
            Start date in ISO format
        date_to_str : str  
            End date in ISO format
        focus_pairs : list of tuples
            List of (src_site, dest_site) pairs to collect data for (UPPERCASE from alarms)
        baseline_days : int
            Days to look back for baseline calculation (default: 21)
        include_reverse_pairs : bool
            Whether to also collect data for reverse direction pairs (default: False)
            WARNING: This doubles the number of queries and data volume
        """
        self.date_from = date_from_str
        self.date_to = date_to_str
        self.focus_pairs = focus_pairs
        self.baseline_days = baseline_days
        self.include_reverse_pairs = include_reverse_pairs
        self.baseline_manager = BaselineManager()
        
        # Parse dates for calculations (ensure UTC timezone)
        self.date_from_dt = pd.to_datetime(date_from_str, utc=True).to_pydatetime()
        self.date_to_dt = pd.to_datetime(date_to_str, utc=True).to_pydatetime()
        
        # Add reverse pairs if requested
        if self.include_reverse_pairs:
            original_pairs = list(focus_pairs)  # Make a copy
            reverse_pairs = [(dest, src) for src, dest in original_pairs if (dest, src) not in original_pairs]
            self.focus_pairs = original_pairs + reverse_pairs
            print(f"🔄 Added {len(reverse_pairs)} reverse pairs (total: {len(self.focus_pairs)})")
        
        print(f"🎯 Focused data collector initialized")
        print(f"   Analysis period: {date_from_str} to {date_to_str}")
        print(f"   Focus pairs: {len(self.focus_pairs)} site pairs ({'bidirectional' if include_reverse_pairs else 'unidirectional'})")
        print(f"   Baseline lookback: {baseline_days} days")
        
        # Map uppercase focus pairs to actual database site names
        # print(f"\n📍 Mapping site names from alarms to database...")
        self.site_name_mapping = self._build_site_name_mapping()
        self.db_focus_pairs = self._map_focus_pairs_to_db_names()
        
        print(f"   ✅ Mapped {len(self.db_focus_pairs)} pairs to database site names")
        
        # # Show focus pairs
        # for i, (src, dest) in enumerate(self.db_focus_pairs[:5]):
        #     alarm_src, alarm_dest = focus_pairs[i]
        #     print(f"      {i+1}. {alarm_src} ↔ {alarm_dest} → {src} ↔ {dest}")
        # if len(focus_pairs) > 5:
        #     print(f"      ... and {len(focus_pairs) - 5} more pairs")

    def _build_site_name_mapping(self):
        all_alarm_sites = set()
        for src, dest in self.focus_pairs:
            all_alarm_sites.add(src)
            all_alarm_sites.add(dest)
        
        site_mapping = {}
        
        # print(f"   🔍 Looking up {len(all_alarm_sites)} unique sites in database...")
        
        # Query each site to find its actual database name
        for alarm_site in all_alarm_sites:
            if not alarm_site:  # Skip empty sites
                continue
                
            try:
                # Case-insensitive search in throughput index first
                query = {
                    "size": 1,
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "term": {
                                        "src_netsite": {
                                        "value": alarm_site,
                                        "case_insensitive": True
                                        }
                                    }
                                },
                                {
                                    "term": {
                                        "dest_netsite": {
                                            "value": alarm_site,
                                            "case_insensitive": True
                                        }
                                    }
                                }

                            ]
                        }
                    },
                    "_source": ["src_netsite", "dest_netsite"]
                }
                
                result = hp.es.search(index='ps_owd', body=query)
                
                if result['hits']['hits']:
                    hit = result['hits']['hits'][0]['_source']
                    # Find which field matched
                    if hit.get('src_netsite', '').upper() == alarm_site:
                        db_name = hit['src_netsite']
                    elif hit.get('dest_netsite', '').upper() == alarm_site:
                        db_name = hit['dest_netsite']
                    else:
                        db_name = alarm_site  # Fallback
                    
                    site_mapping[alarm_site] = db_name
                    # print(f"      • {alarm_site} → {db_name}")
                else:
                    # No match found, use as-is
                    site_mapping[alarm_site] = alarm_site
                    print(f"      • {alarm_site} → {alarm_site} (no DB match)")
                    
            except Exception as e:
                print(f"      ❌ Error mapping {alarm_site}: {e}")
                site_mapping[alarm_site] = alarm_site  # Fallback
        
        return site_mapping
    
    def _map_focus_pairs_to_db_names(self):
        """
        Convert uppercase focus pairs to database site names using the mapping
        """
        db_pairs = []
        
        for src_alarm, dest_alarm in self.focus_pairs:
            src_db = self.site_name_mapping.get(src_alarm, src_alarm)
            dest_db = self.site_name_mapping.get(dest_alarm, dest_alarm)
            db_pairs.append((src_db, dest_db))
        
        return db_pairs

    def collect_throughput_data_focused(self):
        """Collect throughput data with baselines for focus pairs only (query-level filtering)"""
        print(f"\n📊 Collecting throughput data for {len(self.db_focus_pairs)} focus pairs...")
        
        all_throughput_data = []
        
        # Query each focus pair individually to minimize data transfer (using database site names)
        for src_site, dest_site in self.db_focus_pairs:
            # print(f"   📡 Querying throughput: {src_site} ↔ {dest_site}...")
            
            try:                
                # Query throughput data for this specific pair
                pair_data = self._query_throughput_for_pair(src_site, dest_site)
                
                if pair_data:
                    all_throughput_data.extend(pair_data)
                    print(f"      → Found {len(pair_data)} measurements")
                else:
                    print(f"      → No data")
                    
            except Exception as e:
                print(f"      ❌ Error querying {src_site} ↔ {dest_site}: {e}")
                continue
        
        if not all_throughput_data:
            print("   ⚠️ No throughput data found for focus pairs")
            return pd.DataFrame()
        
        # Process throughput data
        current_df = pd.DataFrame(all_throughput_data)
        
        # Process current data (same as original function)
        current_df['dt'] = pd.to_datetime(current_df['from'], unit='ms', utc=True)
        current_df['src_site'] = current_df['src_site'].str.upper()
        current_df['dest_site'] = current_df['dest_site'].str.upper()
        current_df['value_mbps'] = current_df['value'] * 1e-6
        
        
        print(f"   ✅ Collected {len(current_df)} throughput measurements for focus pairs")
        
        # Get baselines for focus pairs only
        print("   📊 Calculating baselines for focus pairs...")
        
        # Use bulk baseline calculation for efficiency
        unique_pairs = current_df[['src_site', 'dest_site']].drop_duplicates()
        pair_list = [(row['src_site'], row['dest_site']) for _, row in unique_pairs.iterrows()]
        
        ref_date = pd.to_datetime(self.date_from, utc=True).to_pydatetime()
        baseline_results = self.baseline_manager.get_bulk_simple_throughput_baselines(
            pair_list, ref_date, self.baseline_days
        )
        
        # Add baseline columns
        baseline_columns = ['baseline_mean_mbps', 'baseline_median_mbps', 'baseline_std_mbps', 
                           'baseline_count', 'baseline_quality']
        
        for col in baseline_columns:
            current_df[col] = None
        
        # Fill baseline values
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
        
        # Calculate performance metrics
        valid_baseline_mask = (current_df['baseline_median_mbps'].notna()) & (current_df['baseline_median_mbps'] > 0)
        
        current_df['thr_ratio'] = None
        current_df['thr_perf_flag'] = False
        
        if valid_baseline_mask.any():
            current_df.loc[valid_baseline_mask, 'thr_ratio'] = (
                current_df.loc[valid_baseline_mask, 'value_mbps'] / 
                current_df.loc[valid_baseline_mask, 'baseline_median_mbps']
            ).clip(upper=10.0)
            
            current_df.loc[valid_baseline_mask, 'thr_perf_flag'] = (
                current_df.loc[valid_baseline_mask, 'thr_ratio'] <= 0.6
            )
        
        print(f"   🚨 Performance flags: {current_df['thr_perf_flag'].sum()} degraded throughput events")
        
        # Throughput filling will be done later after trace data is collected
        # (needed for symmetric path validation)
        
        return current_df
    
    def _query_throughput_for_pair(self, src_site, dest_site):
        date_from_ms = int(self.date_from_dt.timestamp() * 1000)
        date_to_ms = int(self.date_to_dt.timestamp() * 1000)
        
        query = {
            "bool": {
                "must": [
                    {
                        "range": {
                            "timestamp": {
                                "gt": date_from_ms,
                                "lte": date_to_ms
                            }
                        }
                    },
                    {"term": {"src_production": True}},
                    {"term": {"dest_production": True}},
                    {"term": {"ipv6": True}},
                    {"term": {"src_netsite": src_site}},
                    {"term": {"dest_netsite": dest_site}}
                ]
            }
        }
        
        aggregations = {
            "groupby": {
                "composite": {
                    "size": 9999,
                    "sources": [
                        {"ipv6": {"terms": {"field": "ipv6"}}},
                        {"src": {"terms": {"field": "src"}}},
                        {"dest": {"terms": {"field": "dest"}}},
                        {"src_host": {"terms": {"field": "src_host"}}},
                        {"dest_host": {"terms": {"field": "dest_host"}}},
                        {"src_site": {"terms": {"field": "src_netsite"}}},
                        {"dest_site": {"terms": {"field": "dest_netsite"}}}
                    ]
                },
                "aggs": {
                    "throughput": {
                        "avg": {"field": "throughput"}
                    }
                }
            }
        }
        
        try:
            aggdata = hp.es.search(index='ps_throughput', query=query, aggregations=aggregations)
            
            aggrs = []
            if 'aggregations' in aggdata and 'groupby' in aggdata['aggregations']:
                for item in aggdata['aggregations']['groupby']['buckets']:
                    aggrs.append({
                        'hash': str(item['key']['src'] + '-' + item['key']['dest']),
                        'from': date_from_ms, 
                        'to': date_to_ms,
                        'ipv6': item['key']['ipv6'],
                        'src': item['key']['src'], 
                        'dest': item['key']['dest'],
                        'src_host': item['key']['src_host'], 
                        'dest_host': item['key']['dest_host'],
                        'src_site': item['key']['src_site'], 
                        'dest_site': item['key']['dest_site'],
                        'value': item['throughput']['value'],
                        'doc_count': item['doc_count']
                    })
            
            return aggrs
            
        except Exception as e:
            print(f"         ❌ ES query failed: {e}")
            return []
    
    def _fill_symmetric_throughput(self, throughput_df, trace_df, validation_mode='strict'):
        """
        Fill missing throughput measurements using reverse direction when available
        and ASN paths are symmetric
        
        This helps with the low frequency of throughput tests - if we have A→B but not B→A,
        and the ASN paths are symmetric, we can use A→B value for B→A.
        """
        print(f"   🔄 Analyzing bidirectional throughput coverage with path symmetry validation...")
        
        original_count = len(throughput_df)
        
        # Group by site pairs to find missing reverse directions
        pairs_with_data = set(zip(throughput_df['src_site'], throughput_df['dest_site']))
        
        missing_reverse = []
        filled_count = 0
        symmetric_validated = 0
        asymmetric_skipped = 0
        
        for src, dest in pairs_with_data:
            reverse_pair = (dest, src)
            
            if reverse_pair not in pairs_with_data:
                # We have A→B but not B→A - check if paths are symmetric
                is_symmetric = self._check_path_symmetry(src, dest, trace_df, validation_mode)
                
                if is_symmetric:
                    forward_data = throughput_df[
                        (throughput_df['src_site'] == src) & 
                        (throughput_df['dest_site'] == dest)
                    ].copy()
                    
                    if not forward_data.empty:
                        # Create reverse direction entries
                        reverse_data = forward_data.copy()
                        reverse_data['src_site'] = dest
                        reverse_data['dest_site'] = src
                        reverse_data['hash'] = reverse_data['dest'] + '-' + reverse_data['src']  # Swap hash
                        
                        # Swap IP addresses and hosts if available
                        if 'src' in reverse_data.columns and 'dest' in reverse_data.columns:
                            reverse_data['src'], reverse_data['dest'] = reverse_data['dest'], reverse_data['src']
                        if 'src_host' in reverse_data.columns and 'dest_host' in reverse_data.columns:
                            reverse_data['src_host'], reverse_data['dest_host'] = reverse_data['dest_host'], reverse_data['src_host']
                        
                        # Add flag to indicate this is filled data
                        reverse_data['throughput_filled'] = True
                        
                        # Add to missing reverse list
                        missing_reverse.append({
                            'original_pair': (src, dest),
                            'filled_pair': (dest, src),
                            'measurements': len(reverse_data),
                            'symmetric': True
                        })
                        
                        # Append to throughput_df
                        throughput_df = pd.concat([throughput_df, reverse_data], ignore_index=True)
                        filled_count += len(reverse_data)
                        symmetric_validated += 1
                else:
                    # Paths are not symmetric - skip filling
                    asymmetric_skipped += 1
        
        # Add throughput_filled flag to original data
        if 'throughput_filled' not in throughput_df.columns:
            throughput_df['throughput_filled'] = False
            throughput_df.loc[throughput_df.index < original_count, 'throughput_filled'] = False
        
        print(f"      📊 Path symmetry validation:")
        print(f"         • Symmetric pairs (filled): {symmetric_validated}")
        print(f"         • Asymmetric pairs (skipped): {asymmetric_skipped}")
        
        if missing_reverse:
            print(f"      ✅ Filled {len(missing_reverse)} symmetric pairs ({filled_count} measurements)")
            print(f"      📊 Total throughput measurements: {original_count} → {len(throughput_df)}")
            
            # Show examples
            for i, fill_info in enumerate(missing_reverse[:3]):
                orig = fill_info['original_pair']
                filled = fill_info['filled_pair']
                count = fill_info['measurements']
                print(f"         {i+1}. {orig[0]} → {orig[1]} ({count} measurements) → filled {filled[0]} → {filled[1]}")
            
            if len(missing_reverse) > 3:
                print(f"         ... and {len(missing_reverse) - 3} more symmetric pairs")
        else:
            print(f"      ℹ️ No fillable symmetric pairs found")
        
        return throughput_df
    
    def _check_path_symmetry(self, src_site, dest_site, trace_df, validation_mode='strict'):
        """
        Check if ASN paths between two sites are symmetric by comparing ASN sequences
        
        Parameters:
        -----------
        validation_mode : str
            'strict' - exact reverse path match required
            'relaxed' - allow minor differences (length ±1, 80% ASN overlap)
            'lenient' - just check that both directions have stable paths
        
        Returns True if paths are considered symmetric based on validation mode
        """
        # Get traces for both directions
        forward_traces = trace_df[
            (trace_df['src_site'] == src_site) & 
            (trace_df['dest_site'] == dest_site)
        ]
        reverse_traces = trace_df[
            (trace_df['src_site'] == dest_site) & 
            (trace_df['dest_site'] == src_site)
        ]
        
        if forward_traces.empty or reverse_traces.empty:
            return False  # Can't validate symmetry without both directions
        
        # Get most common ASN paths for each direction
        forward_asn_paths = forward_traces['asns'].apply(
            lambda x: tuple(asn for asn in x if asn and asn != 0)  # Clean and convert to tuple
        ).value_counts()
        
        reverse_asn_paths = reverse_traces['asns'].apply(
            lambda x: tuple(asn for asn in x if asn and asn != 0)  # Clean and convert to tuple
        ).value_counts()
        
        if forward_asn_paths.empty or reverse_asn_paths.empty:
            return False
        
        # Get the most common paths
        forward_common_path = forward_asn_paths.index[0]
        reverse_common_path = reverse_asn_paths.index[0]
        
        if validation_mode == 'strict':
            # Exact reverse match required
            return forward_common_path == reverse_common_path[::-1]
        
        elif validation_mode == 'relaxed':
            # Allow minor differences: length ±1, 80% ASN overlap
            expected_reverse = forward_common_path[::-1]
            
            # Check length difference
            len_diff = abs(len(reverse_common_path) - len(expected_reverse))
            if len_diff > 1:
                return False
            
            # Check ASN overlap
            forward_asns = set(forward_common_path)
            reverse_asns = set(reverse_common_path)
            overlap = len(forward_asns.intersection(reverse_asns))
            total_unique = len(forward_asns.union(reverse_asns))
            
            overlap_ratio = overlap / max(total_unique, 1)
            return overlap_ratio >= 0.8
        
        elif validation_mode == 'lenient':
            # Just check that both directions have stable dominant paths
            forward_stability = forward_asn_paths.iloc[0] / len(forward_traces)
            reverse_stability = reverse_asn_paths.iloc[0] / len(reverse_traces)
            
            # Both directions should have >50% consistency
            return forward_stability > 0.5 and reverse_stability > 0.5
        
        else:
            # Default to strict
            return forward_common_path == reverse_common_path[::-1]

    def collect_owd_data_focused(self, aggregate_minutes=10):
        """
        Collect OWD data for focus pairs only using efficient ES queries
        """
        print(f"\n⏱️ Collecting OWD data for focus pairs (aggregated to {aggregate_minutes}min intervals)...")
        
        # Convert dates to milliseconds for ES query
        date_from_ms = int(self.date_from_dt.timestamp() * 1000)
        date_to_ms = int(self.date_to_dt.timestamp() * 1000)
        
        all_owd_records = []
        
        # Query each focus pair individually to limit data at source (using database site names)
        for src_site, dest_site in self.db_focus_pairs:
            # print(f"   📡 Querying {src_site} → {dest_site}...")
            
            query = {
                "size": 0,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "timestamp": {
                                        "gt": date_from_ms,
                                        "lte": date_to_ms
                                    }
                                }
                            },
                            {"term": {"src_production": True}},
                            {"term": {"dest_production": True}},
                            {"term": {"ipv6": True}},
                            {"term": {"src_netsite": src_site}},
                            {"term": {"dest_netsite": dest_site}}
                        ]
                    }
                },
                "aggs": {
                    "time_series": {
                        "date_histogram": {
                            "field": "timestamp",
                            "fixed_interval": f"{aggregate_minutes}m",
                            "time_zone": "UTC"
                        },
                        "aggs": {
                            "ipv_breakdown": {
                                "terms": {
                                    "field": "ipv6"
                                },
                                "aggs": {
                                    "delay_stats": {
                                        "stats": {"field": "delay_median"}
                                    },
                                    "delay_percentiles": {
                                        "percentiles": {
                                            "field": "delay_median",
                                            "percents": [50, 75, 90, 95, 99]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            try:
                result = hp.es.search(index='ps_owd', body=query)
                
                if 'aggregations' in result and 'time_series' in result['aggregations']:
                    time_buckets = result['aggregations']['time_series']['buckets']
                    
                    for time_bucket in time_buckets:
                        timestamp = time_bucket['key_as_string']
                        ipv_buckets = time_bucket['ipv_breakdown']['buckets']
                        
                        for ipv_bucket in ipv_buckets:
                            ipv6 = ipv_bucket['key']
                            stats = ipv_bucket['delay_stats']
                            percentiles = ipv_bucket['delay_percentiles']['values']
                            
                            if stats['count'] > 0:
                                all_owd_records.append({
                                    'timestamp': timestamp,
                                    'src_site': src_site.upper(),
                                    'dest_site': dest_site.upper(),
                                    'ipv6': ipv6,
                                    'doc_count': stats['count'],
                                    'delay_mean': stats['avg'],
                                    'delay_median': percentiles.get('50.0'),
                                    'delay_p75': percentiles.get('75.0'),
                                    'delay_p90': percentiles.get('90.0'),
                                    'delay_p95': percentiles.get('95.0'),
                                    'delay_p99': percentiles.get('99.0'),
                                    'delay_min': stats['min'],
                                    'delay_max': stats['max']
                                })
                
            except Exception as e:
                print(f"      ❌ Error querying {src_site} → {dest_site}: {e}")
                continue
        
        if not all_owd_records:
            print("   ⚠️ No OWD data found for focus pairs")
            return pd.DataFrame()
        
        owd_df = pd.DataFrame(all_owd_records)
        owd_df['dt'] = pd.to_datetime(owd_df['timestamp'], utc=True)
        
        print(f"   ✅ Collected {len(owd_df)} OWD measurements for focus pairs")
        
        # Get baselines for focus pairs
        print("   📊 Calculating OWD baselines...")
        
        owd_df['baseline_min_owd'] = None
        owd_df['baseline_p95_owd'] = None
        owd_df['baseline_median_owd'] = None
        
        # Use uppercase site names for baseline queries (as expected by baseline_manager)
        for src_site, dest_site in self.focus_pairs:
            baseline = self.baseline_manager.get_expected_owd(
                src_site.upper(), 
                dest_site.upper(),
                time_window=7,  # Use 7 days for OWD baselines
                field_type='netsite',
                reference_date=self.date_from_dt
            )
            
            if baseline and baseline.get('owd_stats'):
                stats = baseline['owd_stats']
                mask = (owd_df['src_site'] == src_site.upper()) & (owd_df['dest_site'] == dest_site.upper())
                
                owd_df.loc[mask, 'baseline_min_owd'] = stats.get('min')
                owd_df.loc[mask, 'baseline_p95_owd'] = stats.get('p95')
                owd_df.loc[mask, 'baseline_median_owd'] = stats.get('median')
        
        # Calculate performance metrics and flags
        owd_df['owd_ratio'] = owd_df['delay_p95'] / owd_df['baseline_min_owd']
        owd_df['owd_ratio'] = owd_df['owd_ratio'].clip(upper=500.0)  # Cap at 500x
        
        # Calculate z_mad (robust z-score using MAD)
        def calculate_z_mad(group):
            values = group['delay_p95'].dropna()
            if len(values) < 3:
                return pd.Series([np.nan] * len(group), index=group.index)
            
            median_val = values.median()
            mad = np.median(np.abs(values - median_val))
            if mad == 0:
                return pd.Series([0] * len(group), index=group.index)
            
            z_mad_scores = (group['delay_p95'] - median_val) / mad
            return z_mad_scores
        
        owd_df['z_mad_owd'] = owd_df.groupby(['src_site', 'dest_site']).apply(calculate_z_mad).reset_index(level=[0,1], drop=True)
        
        # Performance flags
        owd_df['owd_ratio_flag'] = (owd_df['owd_ratio'] >= 1.6) & owd_df['owd_ratio'].notna()
        owd_df['owd_z_mad_flag'] = (owd_df['z_mad_owd'] >= 3.0) & owd_df['z_mad_owd'].notna()
        owd_df['owd_perf_flag'] = owd_df['owd_ratio_flag'] | owd_df['owd_z_mad_flag']
        
        print(f"   🚨 Performance flags: {owd_df['owd_perf_flag'].sum()} high OWD events")
        
        return owd_df

    def collect_packetloss_data_focused(self, aggregate_minutes=10):
        print(f"\n📦 Collecting packet loss data for focus pairs (aggregated to {aggregate_minutes}min intervals)...")
        
        date_from_ms = int(self.date_from_dt.timestamp() * 1000)
        date_to_ms = int(self.date_to_dt.timestamp() * 1000)
        
        all_loss_records = []
        
        # Query each focus pair individually (using database site names)
        for src_site, dest_site in self.db_focus_pairs:
            query = {
                "size": 0,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "timestamp": {
                                        "gt": date_from_ms,
                                        "lte": date_to_ms,
                                        "format": "epoch_millis"
                                    }
                                }
                            },
                            {"term": {"src_production": True}},
                            {"term": {"dest_production": True}},
                            {"term": {"ipv6": True}},
                            {"term": {"src_netsite": src_site}},
                            {"term": {"dest_netsite": dest_site}}
                        ]
                    }
                },
                "aggs": {
                    "time_series": {
                        "date_histogram": {
                            "field": "timestamp",
                            "fixed_interval": f"{aggregate_minutes}m",
                            "time_zone": "UTC"
                        },
                        "aggs": {
                            "ipv_breakdown": {
                                "terms": {
                                    "field": "ipv6"
                                },
                                "aggs": {
                                    "packet_loss_stats": {
                                        "stats": {"field": "packet_loss"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            try:
                result = hp.es.search(index='ps_packetloss', body=query)
                
                if 'aggregations' in result and 'time_series' in result['aggregations']:
                    time_buckets = result['aggregations']['time_series']['buckets']
                    
                    for time_bucket in time_buckets:
                        timestamp = time_bucket['key_as_string']
                        ipv_buckets = time_bucket['ipv_breakdown']['buckets']
                        
                        for ipv_bucket in ipv_buckets:
                            ipv6 = ipv_bucket['key']
                            stats = ipv_bucket['packet_loss_stats']
                            
                            if stats['count'] > 0:
                                all_loss_records.append({
                                    'timestamp': timestamp,
                                    'src_site': src_site.upper(),
                                    'dest_site': dest_site.upper(),
                                    'ipv6': ipv6,
                                    'doc_count': stats['count'],
                                    'packet_loss_avg': stats['avg'],
                                    'packet_loss_min': stats['min'],
                                    'packet_loss_max': stats['max']
                                })
                
            except Exception as e:
                print(f"      ❌ Error querying packet loss for {src_site} → {dest_site}: {e}")
                continue
        
        if not all_loss_records:
            print("   ⚠️ No packet loss data found for focus pairs")
            return pd.DataFrame()
        
        loss_df = pd.DataFrame(all_loss_records)
        loss_df['dt'] = pd.to_datetime(loss_df['timestamp'], utc=True)
        
        # Convert to percentage and add performance flags
        loss_df['packet_loss_pct'] = loss_df['packet_loss_avg'] * 100
        loss_df['loss_perf_flag'] = loss_df['packet_loss_avg'] >= 0.002  # 0.2%
        
        print(f"   ✅ Collected {len(loss_df)} packet loss measurements for focus pairs")
        print(f"   🚨 Performance flags: {loss_df['loss_perf_flag'].sum()} high loss events")
        
        return loss_df

    def collect_traceroute_data_focused(self):
        print(f"\n🛤️ Collecting traceroute data for focus pairs...")
        
        # Convert dates to ISO format for ES query
        date_from_iso = self.date_from_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        date_to_iso = self.date_to_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        all_traces = []
        
        # Query each focus pair individually to limit data (using database site names)
        for src_site, dest_site in self.db_focus_pairs:
            query = {
                "size": 10000,  # Reduced limit per pair
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "timestamp": {
                                        "gt": date_from_iso,
                                        "lte": date_to_iso,
                                        "format": "strict_date_optional_time"
                                    }
                                }
                            },
                            {"term": {"src_production": True}},
                            {"term": {"dest_production": True}},
                            {"term": {"ipv6": True}},
                            {"term": {"src_netsite": src_site}},
                            {"term": {"dest_netsite": dest_site}}
                        ]
                    }
                },
                "_source": [
                    "timestamp", "created_at", "src_netsite", "dest_netsite", "ipv6",
                    "src_host", "dest_host", "src", "dest", 
                    "hops", "asns", "ttls", "rtts",
                    "destination_reached", "path_complete", "route-sha1"
                ],
                "sort": [{"timestamp": {"order": "desc"}}]
            }
            
            try:
                result = hp.es.search(index='ps_trace', body=query)
                traces_for_pair = result['hits']['hits']
                all_traces.extend(traces_for_pair)
                
                # print(f"      • {src_site} → {dest_site}: {len(traces_for_pair)} traces")
                
            except Exception as e:
                print(f"      ❌ Error querying traces for {src_site} → {dest_site}: {e}")
                continue
        
        if not all_traces:
            print("   ⚠️ No traceroute data found for focus pairs")
            return pd.DataFrame()
        
        # Parse traceroute records
        trace_records = []
        for hit in all_traces:
            source = hit['_source']
            
            hops = source.get('hops', [])
            asns = source.get('asns', [])
            ttls = source.get('ttls', [])
            rtts = source.get('rtts', [])
            
            # Create hop-by-hop details
            hop_details = []
            max_len = max(len(hops), len(asns), len(ttls), len(rtts))
            
            for i in range(max_len):
                hop_details.append({
                    'hop_num': i + 1,
                    'ip': hops[i] if i < len(hops) else None,
                    'asn': asns[i] if i < len(asns) else None,
                    'ttl': ttls[i] if i < len(ttls) else None,
                    'rtt': rtts[i] if i < len(rtts) else None
                })
            
            trace_records.append({
                'timestamp': source.get('timestamp'),
                'created_at': source.get('created_at'),
                'src_site': source.get('src_netsite', '').upper(),
                'dest_site': source.get('dest_netsite', '').upper(),
                'ipv6': source.get('ipv6', False),
                'src_host': source.get('src_host'),
                'dest_host': source.get('dest_host'),
                'src_ip': source.get('src'),
                'dest_ip': source.get('dest'),
                'destination_reached': source.get('destination_reached', False),
                'path_complete': source.get('path_complete', False),
                'route_sha1': source.get('route-sha1'),
                'hops': hops,
                'asns': asns,
                'hop_count': len(hops),
                'asn_path': '-'.join([str(asn) for asn in asns if asn is not None]),
                'ip_path': '->'.join([str(hop) for hop in hops if hop is not None]),
                'hop_details': hop_details
            })
        
        trace_df = pd.DataFrame(trace_records)
        trace_df['dt'] = pd.to_datetime(trace_df['timestamp'], utc=True)
        
        print(f"   ✅ Collected {len(trace_df)} traceroute measurements for focus pairs")
        print(f"      • Unique paths (route SHA1): {trace_df['route_sha1'].nunique()}")
        print(f"      • Destination reached: {trace_df['destination_reached'].sum()}/{len(trace_df)}")
        
        return trace_df

    def collect_focused_data(self):
        """Collect all datasets for the focus pairs"""
        print("=" * 80)
        print("🎯 FOCUSED NETWORK DATA COLLECTION")
        print("=" * 80)
        
        results = {}
        
        # Collect individual datasets (all filtered at query level)
        print(f"\n📊 COLLECTING DATA FOR {len(self.db_focus_pairs)} FOCUS PAIRS")
        results['throughput_df'] = self.collect_throughput_data_focused()
        results['owd_df'] = self.collect_owd_data_focused()
        results['loss_df'] = self.collect_packetloss_data_focused()
        results['trace_df'] = self.collect_traceroute_data_focused()
        
        # Handle bidirectional throughput filling after trace data is available
        if self.include_reverse_pairs and not results['throughput_df'].empty and not results['trace_df'].empty:
            print(f"\n🔄 APPLYING SYMMETRIC THROUGHPUT FILLING")
            results['throughput_df'] = self._fill_symmetric_throughput(
                results['throughput_df'], 
                results['trace_df'],
                validation_mode='relaxed'  # Try relaxed validation first
            )
        
        # Create unified performance dataset
        print("\n🔗 CREATING UNIFIED PERFORMANCE DATASET")
        results['performance_df'] = self.create_performance_dataset(
            results['throughput_df'], 
            results['owd_df'], 
            results['loss_df']
        )
        
        # Add focus pairs info (both uppercase and database names)
        results['focus_pairs'] = self.focus_pairs
        results['db_focus_pairs'] = self.db_focus_pairs
        
        # Summary
        print("\n" + "=" * 80)
        print("📋 FOCUSED COLLECTION SUMMARY")
        print("=" * 80)
        for name, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"   {name}: {len(df)} records")
                if 'perf_flag' in df.columns or 'any_perf_flag' in df.columns:
                    flag_col = 'any_perf_flag' if 'any_perf_flag' in df.columns else [col for col in df.columns if 'perf_flag' in col][0]
                    print(f"      → Performance events: {df[flag_col].sum()}")
            else:
                print(f"   {name}: No data")
        
        print(f"\n✅ Focused data collection complete for {len(self.db_focus_pairs)} pairs")
        print(f"   Period: {self.date_from} to {self.date_to}")
        
        return results

    def create_performance_dataset(self, throughput_df, owd_df, loss_df):
        """Create unified performance dataset from focused data"""
        print(f"   🔗 Creating unified dataset from focused performance data...")
        
        if throughput_df.empty and owd_df.empty and loss_df.empty:
            print("      ⚠️ No performance data to unify")
            return pd.DataFrame()
        
        # Get all timestamps from OWD and loss (higher frequency)
        all_times = []
        if not owd_df.empty:
            all_times.extend(owd_df['dt'].tolist())
        if not loss_df.empty:
            all_times.extend(loss_df['dt'].tolist())
        
        if not all_times:
            print("      ⚠️ No timestamp data available")
            return pd.DataFrame()
        
        # Combine OWD and loss data by timestamp and pair
        combined_df = pd.DataFrame()
        
        if not owd_df.empty:
            combined_df = owd_df.copy()
            
        if not loss_df.empty:
            if combined_df.empty:
                combined_df = loss_df.copy()
            else:
                # Merge on timestamp and pair
                combined_df = pd.merge(
                    combined_df, 
                    loss_df, 
                    on=['timestamp', 'src_site', 'dest_site', 'ipv6', 'dt'], 
                    how='outer',
                    suffixes=('', '_loss')
                )
        
        # Add throughput data (lower frequency, wider time window)
        if not throughput_df.empty and not combined_df.empty:
            # For each combined record, find closest throughput measurement
            for idx, row in combined_df.iterrows():
                # Find throughput data for this pair within 1 hour
                # Ensure both timestamps are timezone-aware
                row_dt = row['dt']
                if row_dt.tz is None:
                    row_dt = pd.to_datetime(row_dt, utc=True)
                
                thr_match = throughput_df[
                    (throughput_df['src_site'] == row['src_site']) &
                    (throughput_df['dest_site'] == row['dest_site']) &
                    (throughput_df['ipv6'] == row['ipv6'])
                ]
                
                if not thr_match.empty:
                    # Calculate time differences manually to avoid timezone issues
                    time_diffs = []
                    for _, thr_row in thr_match.iterrows():
                        thr_dt = thr_row['dt']
                        if thr_dt.tz is None:
                            thr_dt = pd.to_datetime(thr_dt, utc=True)
                        time_diff = abs((thr_dt - row_dt).total_seconds())
                        time_diffs.append(time_diff)
                    
                    # Filter to within 1 hour
                    valid_indices = [i for i, diff in enumerate(time_diffs) if diff <= 3600]
                    if valid_indices:
                        thr_match = thr_match.iloc[valid_indices]
                    else:
                        thr_match = pd.DataFrame()  # Empty if no matches within time window
                
                if not thr_match.empty:
                    closest_thr = thr_match.iloc[0]
                    combined_df.at[idx, 'throughput_mbps'] = closest_thr['value_mbps']
                    combined_df.at[idx, 'baseline_median_throughput'] = closest_thr['baseline_median_mbps']
                    combined_df.at[idx, 'thr_ratio'] = closest_thr['thr_ratio']
                    combined_df.at[idx, 'thr_perf_flag'] = closest_thr['thr_perf_flag']
        
        if combined_df.empty:
            print("      ⚠️ No data could be combined")
            return pd.DataFrame()
        
        # Overall performance flag
        combined_df['any_perf_flag'] = (
            combined_df.get('owd_perf_flag', False) |
            combined_df.get('loss_perf_flag', False) |
            combined_df.get('thr_perf_flag', False)
        )
        
        print(f"      ✅ Created unified dataset: {len(combined_df)} records")
        print(f"         • Performance events: {combined_df['any_perf_flag'].sum()}")
        
        return combined_df


def main_from_cooccurrence_windows(cooccurrence_windows_df, date_from, date_to, include_reverse_pairs=False):
    focus_pairs = cooccurrence_windows_df[['src_site', 'dest_site']].drop_duplicates().values.tolist()
    
    print(f"🎯 Extracted {len(focus_pairs)} focus pairs from cooccurrence windows")
    collector = FocusedDataCollector(date_from, date_to, focus_pairs, baseline_days=21, include_reverse_pairs=include_reverse_pairs)
    datasets = collector.collect_focused_data()
    
    return datasets




def main_with_pairs(focus_pairs_list, date_from, date_to, include_reverse_pairs=False):
    collector = FocusedDataCollector(date_from, date_to, focus_pairs_list, baseline_days=21, include_reverse_pairs=include_reverse_pairs)
    datasets = collector.collect_focused_data()
    
    return datasets