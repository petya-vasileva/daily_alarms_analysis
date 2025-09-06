"""
Network Data Queries Module

Contains all Elasticsearch queries for network performance data collection.
Separated from baseline_manager.py to reduce file size and improve organization.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import utils.helpers as hp


def query_throughput_raw(src_site: str, dest_site: str, date_from_iso: str, date_to_iso: str) -> List[Dict[str, Any]]:

    # Convert ISO dates to milliseconds for ES timestamp query
    date_from_dt = datetime.strptime(date_from_iso, '%Y-%m-%dT%H:%M:%S.000Z')
    date_to_dt = datetime.strptime(date_to_iso, '%Y-%m-%dT%H:%M:%S.000Z')
    
    date_from_ms = int(date_from_dt.timestamp() * 1000)
    date_to_ms = int(date_to_dt.timestamp() * 1000)

    query = {
        "size": 10000,  # Get all raw measurements
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
        "_source": [
            "timestamp", "src", "dest", "src_host", "dest_host", 
            "src_netsite", "dest_netsite", "throughput", "ipv6"
        ],
        "sort": [{"timestamp": {"order": "desc"}}]
    }
    
    try:
        result = hp.es.search(index='ps_throughput', body=query)
        
        throughput_records = []
        for hit in result['hits']['hits']:
            source = hit['_source']
            # Convert timestamp to milliseconds if it's a string
            timestamp_value = source['timestamp']
            if isinstance(timestamp_value, str):
                # Parse ISO string and convert to milliseconds
                timestamp_dt = pd.to_datetime(timestamp_value, utc=True)
                timestamp_ms = int(timestamp_dt.timestamp() * 1000)
            else:
                # Already in milliseconds
                timestamp_ms = int(timestamp_value)
            
            throughput_records.append({
                'hash': str(source['src'] + '-' + source['dest']),
                'from': date_from_ms,
                'to': date_to_ms,
                'ipv6': source['ipv6'],
                'src': source['src'],
                'dest': source['dest'],
                'src_host': source['src_host'],
                'dest_host': source['dest_host'],
                'src_site': source['src_netsite'],
                'dest_site': source['dest_netsite'],
                'value': source['throughput'],
                'timestamp_ms': timestamp_ms,
                'doc_count': 1  # Each record represents one measurement
            })
        
        return throughput_records
        
    except Exception as e:
        print(f"Error querying throughput for {src_site} → {dest_site}: {e}")
        return []


def query_owd_aggregated(src_site: str, dest_site: str, date_from_iso: str, date_to_iso: str, 
                        aggregate_minutes: int = 10) -> List[Dict[str, Any]]:

    # Convert ISO dates to milliseconds
    date_from_dt = datetime.strptime(date_from_iso, '%Y-%m-%dT%H:%M:%S.000Z')
    date_to_dt = datetime.strptime(date_to_iso, '%Y-%m-%dT%H:%M:%S.000Z')
    
    date_from_ms = int(date_from_dt.timestamp() * 1000)
    date_to_ms = int(date_to_dt.timestamp() * 1000)
    
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
        
        owd_records = []
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
                        owd_records.append({
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
        
        return owd_records
        
    except Exception as e:
        print(f"Error querying OWD for {src_site} → {dest_site}: {e}")
        return []


def query_packetloss_aggregated(src_site: str, dest_site: str, date_from_iso: str, date_to_iso: str,
                               aggregate_minutes: int = 10) -> List[Dict[str, Any]]:

    # Convert ISO dates to milliseconds
    date_from_dt = datetime.strptime(date_from_iso, '%Y-%m-%dT%H:%M:%S.000Z')
    date_to_dt = datetime.strptime(date_to_iso, '%Y-%m-%dT%H:%M:%S.000Z')
    
    date_from_ms = int(date_from_dt.timestamp() * 1000)
    date_to_ms = int(date_to_dt.timestamp() * 1000)
    
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
        
        loss_records = []
        if 'aggregations' in result and 'time_series' in result['aggregations']:
            time_buckets = result['aggregations']['time_series']['buckets']
            
            for time_bucket in time_buckets:
                timestamp = time_bucket['key_as_string']
                ipv_buckets = time_bucket['ipv_breakdown']['buckets']
                
                for ipv_bucket in ipv_buckets:
                    ipv6 = ipv_bucket['key']
                    stats = ipv_bucket['packet_loss_stats']
                    
                    if stats['count'] > 0:
                        loss_records.append({
                            'timestamp': timestamp,
                            'src_site': src_site.upper(),
                            'dest_site': dest_site.upper(),
                            'ipv6': ipv6,
                            'doc_count': stats['count'],
                            'packet_loss_avg': stats['avg'],
                            'packet_loss_min': stats['min'],
                            'packet_loss_max': stats['max']
                        })
        
        return loss_records
        
    except Exception as e:
        print(f"Error querying packet loss for {src_site} → {dest_site}: {e}")
        return []


def query_traceroute_raw(src_site: str, dest_site: str, date_from_iso: str, date_to_iso: str) -> List[Dict[str, Any]]:

    query = {
        "size": 10000,
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
        return result['hits']['hits']
        
    except Exception as e:
        print(f"Error querying traces for {src_site} → {dest_site}: {e}")
        return []


def query_owd_baseline(src: str, dest: str, start_time_iso: str, end_time_iso: str, 
                      field_type: str = "netsite") -> Dict[str, Any]:

    src_field = f"src_{field_type}"
    dest_field = f"dest_{field_type}"
    
    query = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_time_iso,
                                "lte": end_time_iso
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
                                "lt": 10000
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
    
    try:
        return hp.es.search(index='ps_owd*', body=query)
    except Exception as e:
        print(f"Error querying OWD baseline for {src} → {dest}: {e}")
        return {}


def query_throughput_baseline(date_from_iso: str, date_to_iso: str) -> List[Dict[str, Any]]:

    # Convert ISO to milliseconds for ES timestamp query
    date_from_dt = datetime.strptime(date_from_iso, '%Y-%m-%dT%H:%M:%S.000Z')
    date_to_dt = datetime.strptime(date_to_iso, '%Y-%m-%dT%H:%M:%S.000Z')
    
    date_from_ms = int(date_from_dt.timestamp() * 1000)
    date_to_ms = int(date_to_dt.timestamp() * 1000)
    
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
                {"term": {"ipv6": True}}
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
        
        results = []
        for item in aggdata['aggregations']['groupby']['buckets']:
            results.append({
                'hash': str(item['key']['src'] + '-' + item['key']['dest']),
                'from': date_from_ms,
                'to': date_to_ms,
                'from_str': date_from_dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'to_str': date_to_dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
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
        
        return results
        
    except Exception as e:
        print(f"Error querying throughput baseline: {e}")
        return []


def query_trace_baseline(src: str, dest: str, start_time_iso: str, end_time_iso: str,
                        field_type: str = "netsite", query_type: str = "path_length") -> Dict[str, Any]:

    src_field = f"src_{field_type}"
    dest_field = f"dest_{field_type}"
    
    if query_type == "path_length":
        query = {
            "size": 5000,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time_iso,
                                    "lte": end_time_iso
                                }
                            }
                        },
                        {"term": {src_field: src}},
                        {"term": {dest_field: dest}},
                        {"term": {"ipv6": True}},
                        {"exists": {"field": "hops"}}
                    ]
                }
            },
            "_source": [
                "timestamp", "hops", "destination_reached", "path_complete", 
                "n_hops", "ttls", "max_rtt"
            ],
            "sort": [{"timestamp": {"order": "desc"}}]
        }
    else:  # reachability
        query = {
            "size": 1000,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time_iso,
                                    "lte": end_time_iso
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
    
    try:
        return hp.es.search(index='ps_trace*', body=query)
    except Exception as e:
        print(f"Error querying trace baseline for {src} → {dest}: {e}")
        return {}