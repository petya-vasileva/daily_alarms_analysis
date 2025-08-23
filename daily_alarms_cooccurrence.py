"""
Daily Alarm Cooccurrence Analysis

This module analyzes network alarm data to identify cooccurrence patterns between
ASN path anomalies (routing changes) and performance degradation alarms.

Complexity Score Description:
========================
The complexity score is a normalized metric (0-1) that quantifies daily network instability
by combining multiple factors:

1. Volume (V): Logarithmically scaled alarm count (weight: 30%)
2. Performance Impact (P): Weighted severity based on alarm types (weight: 20%)  
3. Event Evenness (E): Entropy-based measure of alarm type diversity (weight: 15%)
4. Site Spread (S): Number of unique sites affected (weight: 10%)
5. Country Spread (C): Number of unique countries involved (weight: 5%)
6. Correlation (R): ASN-Performance co-occurrence on same site pairs (weight: 20%)

Current Formula: Score = 0.30V + 0.20P + 0.15E + 0.10S + 0.05C + 0.20R_eff

Where R_eff applies a dampening factor based on the number of co-occurring pairs
to avoid overweighting days with few correlations.

Higher scores indicate more complex, widespread network instability with diverse
alarm types affecting multiple sites and countries.
"""

# Standard library imports
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")
import warnings
warnings.filterwarnings('ignore')

# Local imports
import helpers as hp
from site_geography import add_geography_to_dataframe





# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

REROUTING_EVENT = 'ASN path anomalies'

def _minmax(x, lo, hi):
    """Normalize value to [0,1] range using min-max scaling."""
    if pd.isna(x) or hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def _log_minmax(x, lo, hi):
    """Normalize value using log transformation then min-max scaling."""
    return _minmax(np.log1p(x), np.log1p(lo), np.log1p(hi))


def _entropy_evenness(series, K_total):
    """Calculate normalized entropy in [0,1]; rewards balanced variety, not just count."""
    if len(series) == 0 or series.sum() == 0 or K_total <= 1:
        return 0.0
    p = (series / series.sum()).astype(float)
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(H / np.log(K_total))


def _event_evenness_pairs(day_df, K_total):
    """
    Calculate entropy-based evenness over unique (event, src_site, dest_site) tuples
    so each pair contributes at most once per event type per day.
    """
    base = (day_df.dropna(subset=['src_site','dest_site'])
                  .drop_duplicates(['event','src_site','dest_site']))
    counts = base['event'].value_counts()
    return _entropy_evenness(counts, max(K_total, 1))


def _severity_proxy(day_df):
    """
    Calculate severity proxy based on numeric fields if available,
    otherwise use per-type weights.
    """
    if set(['owd_ratio','loss_pct','throughput_ratio']).issubset(day_df.columns):
        owd = ((day_df['owd_ratio'] - 1.0).clip(lower=0)).sum()
        loss = (day_df['loss_pct'].clip(lower=0)).sum()
        thr  = ((1.0 - day_df['throughput_ratio']).clip(lower=0)).sum()
        return 1.0*owd + 1.5*loss + 1.0*thr

    weights = {
        'high one-way delay': 1.0,
        'high delay from/to multiple sites': 1.2,
        'high packet loss': 1.2,
        'high packet loss on multiple links': 1.6,
        'bandwidth decreased': 2.0,
        'bandwidth decreased from/to multiple sites': 2.0,
    }
    return float(day_df['event'].map(
        lambda e: next((w for k,w in weights.items() if isinstance(e,str) and k in e), 0.5)
    ).sum())


# ============================================================================
# ALARM PROCESSING FUNCTIONS
# ============================================================================

def expand_multisite_alarm(alarm_data):
    """
    Expand multi-site alarms into individual site pairs
    Returns list of expanded alarm records
    """
    expanded = []

    # Check if this is a multi-site alarm with actual data
    has_dest_sites = alarm_data.get('dest_sites') and isinstance(alarm_data['dest_sites'], list) and len(alarm_data['dest_sites']) > 0
    has_src_sites = alarm_data.get('src_sites') and isinstance(alarm_data['src_sites'], list) and len(alarm_data['src_sites']) > 0

    if has_dest_sites or has_src_sites:
        central_site = alarm_data.get('site')

        if central_site:  # Only expand if we have a central site
            # Expand destination sites (central_site -> dest_sites)
            if has_dest_sites:
                for dest_site in alarm_data['dest_sites']:
                    if dest_site:  # Skip None/empty values
                        expanded_alarm = alarm_data.copy()
                        expanded_alarm['src_site'] = central_site
                        expanded_alarm['dest_site'] = dest_site
                        expanded_alarm['is_multisite_expanded'] = True
                        expanded_alarm['expansion_type'] = 'central_to_dest'
                        expanded.append(expanded_alarm)

            # Expand source sites (src_sites -> central_site)
            if has_src_sites:
                for src_site in alarm_data['src_sites']:
                    if src_site:  # Skip None/empty values
                        expanded_alarm = alarm_data.copy()
                        expanded_alarm['src_site'] = src_site
                        expanded_alarm['dest_site'] = central_site
                        expanded_alarm['is_multisite_expanded'] = True
                        expanded_alarm['expansion_type'] = 'src_to_central'
                        expanded.append(expanded_alarm)
        else:
            # No central site, keep as-is
            alarm_data['is_multisite_expanded'] = False
            alarm_data['expansion_type'] = 'individual'
            expanded.append(alarm_data)
    else:
        # Not a multi-site alarm or no data in arrays, return as-is
        alarm_data['is_multisite_expanded'] = False
        alarm_data['expansion_type'] = 'individual'
        expanded.append(alarm_data)

    return expanded


# ============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# ============================================================================

def same_day_pair_overlap_exact(day_df, unify_direction=False):
    """
    Same-day, same-(src,dst) co-occurrence:
    For each (src_site, dest_site) on this day, mark if there was ANY ASN event
    and ANY performance event. No timing finesse—exactly the user's rule.

    Returns:
      pairs_df: rows per pair with has_asn/has_perf/cooccur flags
      summary: single-row DataFrame with counts and R metrics for the day
    """
    df = day_df.dropna(subset=['src_site','dest_site']).copy()

    if unify_direction:
        # treat A↔B as the same pair
        pair = df[['src_site','dest_site']].apply(lambda r: tuple(sorted([r['src_site'], r['dest_site']])), axis=1)
        df['src_site'], df['dest_site'] = zip(*pair)

    df['is_asn']  = (df['event'] == REROUTING_EVENT)
    df['is_perf'] = ~df['is_asn']  # everything else = performance

    pairs_df = (df.groupby(['src_site','dest_site'], as_index=False)
                  .agg(has_asn = ('is_asn',  'any'),
                       has_perf= ('is_perf', 'any')))
    pairs_df['cooccur'] = pairs_df['has_asn'] & pairs_df['has_perf']

    # day-level summary
    n_asn_pairs   = int(pairs_df['has_asn'].sum())
    n_perf_pairs  = int(pairs_df['has_perf'].sum())
    n_both_pairs  = int(pairs_df['cooccur'].sum())
    n_union_pairs = int(n_asn_pairs + n_perf_pairs - n_both_pairs)

    summary = pd.DataFrame([{
        'n_pairs_asn': n_asn_pairs,
        'n_pairs_perf': n_perf_pairs,
        'n_pairs_both': n_both_pairs,
        'n_pairs_union': n_union_pairs,
        # R_asn: fraction of ASN pairs that also had perf
        'R_asn':   (n_both_pairs / max(1, n_asn_pairs)),
        # R_union: fraction of all (asn ∪ perf) pairs that had both
        'R_union': (n_both_pairs / max(1, n_union_pairs)),
        'has_cooccurrence': (n_both_pairs > 0)
    }])

    return pairs_df, summary


def cooccurrence_windows_for_day_exact(day_df, pad_minutes=60, unify_direction=False):
    """
    Build coarse incident windows ONLY for pairs that had same-day co-occurrence.
    With one timestamp per event type, window = [min(ts_asn, ts_perf)-pad, max(...)+pad].
    """
    df = day_df.dropna(subset=['src_site','dest_site']).copy()
    if unify_direction:
        pair = df[['src_site','dest_site']].apply(lambda r: tuple(sorted([r['src_site'], r['dest_site']])), axis=1)
        df['src_site'], df['dest_site'] = zip(*pair)

    df['is_asn']  = (df['event'] == REROUTING_EVENT)
    df['is_perf'] = ~df['is_asn']

    flags = (df.groupby(['src_site','dest_site'])
               .agg(has_asn=('is_asn','any'), has_perf=('is_perf','any')))
    co_pairs = flags[flags['has_asn'] & flags['has_perf']].index

    if len(co_pairs) == 0:
        return pd.DataFrame(columns=['src_site','dest_site','start','end','n_asn','n_perf'])

    pad = pd.Timedelta(minutes=pad_minutes)
    rows = []
    for s,d in co_pairs:
        sub = df[(df['src_site']==s) & (df['dest_site']==d)]
        t_asn  = pd.to_datetime(sub.loc[sub['is_asn'],  'timestamp'], utc=True, errors='coerce').dropna()
        t_perf = pd.to_datetime(sub.loc[sub['is_perf'], 'timestamp'], utc=True, errors='coerce').dropna()
        if t_asn.empty or t_perf.empty:
            continue
        t0 = min(t_asn.min(), t_perf.min()) - pad
        t1 = max(t_asn.max(), t_perf.max()) + pad
        rows.append(dict(src_site=s, dest_site=d, start=t0, end=t1,
                         n_asn=len(t_asn), n_perf=len(t_perf)))
    return pd.DataFrame(rows)


def analyze_alarm_correlations(days_back=14):
    """
     analysis with proper multi-site expansion and ASN correlation
    """

    # Query recent alarms
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    print(f"📅 Analyzing {days_back} days: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    alarm_query = {
        "size": 10000,
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {"range": {"source.to": {"gte": start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                                                        "lte": end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')}}},
                                {"range": {"created_at": {"gte": start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                                                          "lte": end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')}}}
                            ]
                        }
                    },
                    {
                        "terms": {
                            "event": [
                            "bandwidth decreased",
                            "bandwidth decreased from/to multiple sites",
                            "high one-way delay",
                            "high delay from/to multiple sites",
                            "high packet loss",
                            "high packet loss on multiple links",
                            "ASN path anomalies"
                            ]
                        }
                    }
                ]
            }
        },
        "sort": [{"source.to": {"order": "desc", "missing": "_last"}},
                {"created_at": {"order": "desc", "missing": "_last"}}]
    }

    try:
        response = hp.es.search(index='aaas_alarms', body=alarm_query)
        print(f"📊 Found {len(response['hits']['hits'])} raw alarms")
        # print(str(alarm_query).replace('\'', '"'))

        # Process and expand alarms
        all_alarms = []
        all_alarms_raw = []

        for hit in response['hits']['hits']:
            source = hit['_source']
            source_data = source.get('source', {})
            event = source.get('event', 'unknown')
            # Handle timestamp
            timestamp_raw = source_data.get('to') or source.get('to_date') or source.get('created_at')

            try:
                if timestamp_raw:
                    # Handle numeric timestamp (e.g., milliseconds since epoch)
                    if isinstance(timestamp_raw, (int, float)) or (isinstance(timestamp_raw, str) and timestamp_raw.isdigit()):
                        # Convert to int if string
                        ts_val = int(timestamp_raw)
                        # If it's too large, assume milliseconds
                        if ts_val > 1e12:
                            timestamp = pd.to_datetime(ts_val, unit='ms', utc=True)
                        else:
                            timestamp = pd.to_datetime(ts_val, unit='s', utc=True)
                    else:
                        timestamp = pd.to_datetime(timestamp_raw)
                        if timestamp.tz is None:
                            timestamp = timestamp.tz_localize('UTC')
                        else:
                            timestamp = timestamp.tz_convert('UTC')
                else:
                    timestamp = pd.to_datetime('now', utc=True)
            except Exception as ex:
                print(f"❗️ Invalid timestamp format: {timestamp_raw} ({ex})")
                timestamp = pd.to_datetime('now', utc=True)

            # Build alarm data with all fields
            alarm_data = {
                'timestamp': timestamp,
                'date': timestamp.date(),
                'event': source.get('event', 'unknown'),
                'category': source.get('category', 'unknown'),
                'subcategory': source.get('subcategory', 'unknown'),
                # Handle both regular sites and netsites
                'src_site': source_data.get('src_site') or source_data.get('src_netsite'),
                'dest_site': source_data.get('dest_site') or source_data.get('dest_netsite'),
                'site': source_data.get('site'),
                # Multi-site fields
                'src_sites': source_data.get('src_sites'),
                'dest_sites': source_data.get('dest_sites'),
                # Other fields
                'body': source.get('body', ''),
                'alarm_id': source_data.get('alarm_id'),
                'source_data': source_data
            }

            # Keep raw alarm
            all_alarms_raw.append(alarm_data.copy())

            # Expand multi-site alarms
            expanded_alarms = expand_multisite_alarm(alarm_data)
            all_alarms.extend(expanded_alarms)

        alarms_df = pd.DataFrame(all_alarms)
        alarms_raw_df = pd.DataFrame(all_alarms_raw)

        print(f"\n📈 Alarm Processing Results:")
        print(f"  • Raw alarms: {len(alarms_raw_df)}")
        print(f"  • Expanded alarms: {len(alarms_df)}")
        print(f"  • Multi-site expansions: {len(alarms_df[alarms_df['is_multisite_expanded'] == True])}")

        # Event type breakdown after expansion
        print(f"\n📊 Event Types After Expansion:")
        event_counts = alarms_df['event'].value_counts()
        for event, count in event_counts.head(15).items():
            print(f"  • {event}: {count}")

        # Identify ASN alarms
        asn_df = alarms_df[
            alarms_df['event'].str.contains('ASN', case=False, na=False) |
            alarms_df['event'].str.contains('path', case=False, na=False)
        ]

        print(f"\n🛣️ ASN/Path alarms found: {len(asn_df)}")
        if not asn_df.empty:
            print(f"  ASN event types: {asn_df['event'].unique()[:5]}")

        # *** FIXED: DAILY CO-OCCURRENCE ANALYSIS ***
        print(f"\n🔗 DAILY CO-OCCURRENCE ANALYSIS")
        print("=" * 40)

        # Group alarms by date and analyze co-occurrences per day
        daily_correlations = []
        co_occurring_days = []

        # Get unique dates
        unique_dates = sorted(alarms_df['date'].unique())
        print(f"📅 Analyzing {len(unique_dates)} unique dates for co-occurrences...")

        for date in unique_dates:
            daily_alarms = alarms_df[alarms_df['date'] == date].copy()
            
            # Find ASN and performance alarms on this date
            daily_asn = daily_alarms[
                daily_alarms['event']=='ASN path anomalies'
            ]

            daily_performance = daily_alarms[
                daily_alarms['event']!='ASN path anomalies'
            ]

            if len(daily_asn) > 0 and len(daily_performance) > 0:
                # This day has both ASN and performance issues
                print(f"  🎯 {date}: {len(daily_asn)} ASN + {len(daily_performance)} performance alarms")
                
                # Find correlations on this day
                day_correlations = []
                
                # Look for site pair correlations within this day
                for _, asn_alarm in daily_asn.iterrows():
                    asn_src = asn_alarm['src_site']
                    asn_dest = asn_alarm['dest_site']

                    if pd.notna(asn_src) and pd.notna(asn_dest):
                        # Find performance alarms for same site pair (bidirectional)
                        matching_perf = daily_performance[
                            ((daily_performance['src_site'] == asn_src) & 
                             (daily_performance['dest_site'] == asn_dest)) |
                            ((daily_performance['src_site'] == asn_dest) & 
                             (daily_performance['dest_site'] == asn_src))
                        ]

                        for _, perf_alarm in matching_perf.iterrows():
                            time_diff = abs((perf_alarm['timestamp'] - asn_alarm['timestamp']).total_seconds() / 60)

                            correlation = {
                                'date': date,
                                'asn_alarm_time': asn_alarm['timestamp'],
                                'perf_alarm_time': perf_alarm['timestamp'],
                                'time_diff_minutes': time_diff,
                                'asn_event': asn_alarm['event'],
                                'perf_event': perf_alarm['event'],
                                'src_site': asn_src,
                                'dest_site': asn_dest,
                                'correlation_strength': max(0, (1440 - time_diff) / 1440),  # 24hr window
                                'is_multisite': asn_alarm.get('is_multisite_expanded', False) or 
                                               perf_alarm.get('is_multisite_expanded', False)
                            }
                            day_correlations.append(correlation)

                daily_correlations.extend(day_correlations)
                
                # Store information about this co-occurring day
                co_occurring_days.append({
                    'date': date,
                    'asn_count': len(daily_asn),
                    'perf_count': len(daily_performance),
                    'total_alarms': len(daily_alarms),
                    'correlations_found': len(day_correlations),
                    'unique_sites_affected': len(set(
                        [site for alarm in daily_alarms.itertuples() 
                         for site in [alarm.src_site, alarm.dest_site] 
                         if pd.notna(site)]
                    )),
                    'event_types': list(daily_alarms['event'].unique())
                })

        print(f"\n📊 Daily Co-occurrence Results:")
        print(f"  • Days with both ASN and performance issues: {len(co_occurring_days)}")
        print(f"  • Total correlations found: {len(daily_correlations)}")

        # Find the most problematic days
        if co_occurring_days:
            # Sort by total correlations found
            co_occurring_days_df = pd.DataFrame(co_occurring_days)
            top_days = co_occurring_days_df.nlargest(5, 'correlations_found')
            
            print(f"\n🎯 TOP PROBLEMATIC DAYS (with both ASN + performance issues):")
            for _, day in top_days.iterrows():
                print(f"  • {day['date']}: {day['correlations_found']} correlations")
                print(f"    - {day['asn_count']} ASN alarms, {day['perf_count']} performance alarms")
                print(f"    - {day['unique_sites_affected']} unique sites affected")
                print(f"    - Total alarms: {day['total_alarms']}")

        # 2. Site Rerouting Impact Analysis (using expanded data)
        print(f"\n🔄 SITE REROUTING IMPACT ANALYSIS")
        print("=" * 40)

        # Group alarms by site pairs
        site_pair_events = defaultdict(list)

        for _, alarm in alarms_df.iterrows():
            if pd.notna(alarm['src_site']) and pd.notna(alarm['dest_site']):
                # Create canonical site pair (sorted)
                site_pair = tuple(sorted([alarm['src_site'], alarm['dest_site']]))
                site_pair_events[site_pair].append({
                    'timestamp': alarm['timestamp'],
                    'date': alarm['date'],
                    'event': alarm['event'],
                    'is_asn': 'ASN' in alarm['event'] or 'path' in alarm['event'].lower()
                })

        # Find site pairs with ASN changes and multiple events
        rerouted_pairs_impact = []

        for site_pair, events in site_pair_events.items():
            # Check if this pair has ASN path changes
            has_asn_change = any(e['is_asn'] for e in events)

            if has_asn_change and len(events) > 1:
                # Sort events by time
                events_sorted = sorted(events, key=lambda x: x['timestamp'])

                # Find ASN event(s)
                asn_events = [e for e in events_sorted if e['is_asn']]
                other_events = [e for e in events_sorted if not e['is_asn']]

                if asn_events and other_events:
                    # Get unique dates where events occurred
                    event_dates = set(e['date'] for e in events)
                    
                    rerouted_pairs_impact.append({
                        'site_pair': site_pair,
                        'total_events': len(events),
                        'asn_events': len(asn_events),
                        'other_events': len(other_events),
                        'event_types': list(set(e['event'] for e in events)),
                        'time_span_hours': (events_sorted[-1]['timestamp'] - events_sorted[0]['timestamp']).total_seconds() / 3600,
                        'days_affected': len(event_dates),
                        'date_range': f"{min(event_dates)} to {max(event_dates)}"
                    })

        print(f"📊 Rerouting Impact Results:")
        print(f"  • Site pairs analyzed: {len(site_pair_events)}")
        print(f"  • Site pairs with rerouting + other events: {len(rerouted_pairs_impact)}")

        if rerouted_pairs_impact:
            # Sort by number of events
            top_impacted = sorted(rerouted_pairs_impact, key=lambda x: x['total_events'], reverse=True)[:10]
            print(f"\n  Top Impacted Site Pairs:")
            for impact in top_impacted:
                print(f"    • {impact['site_pair'][0]} ↔ {impact['site_pair'][1]}")
                print(f"      Events: {impact['total_events']} ({impact['asn_events']} ASN, {impact['other_events']} other)")
                print(f"      Days affected: {impact['days_affected']}, Time span: {impact['time_span_hours']:.1f} hours")
                print(f"      Date range: {impact['date_range']}")

        return {
            'alarms_df': alarms_df,
            'alarms_raw_df': alarms_raw_df,
            'asn_df': asn_df,
            'daily_correlations': daily_correlations,
            'co_occurring_days': co_occurring_days,
            'rerouted_pairs_impact': rerouted_pairs_impact,
            'site_pair_events': dict(site_pair_events)
        }

    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the  analysis
print("🚀 Starting  correlation analysis...")
analysis_results = analyze_alarm_correlations(days_back=30)

if analysis_results:
    print(f"\n✅  analysis complete!")
    print(f"Data available in 'analysis_results' with:")
    print(f"  • alarms_df ({len(analysis_results['alarms_df'])} expanded alarms)")
    print(f"  • daily_correlations ({len(analysis_results['daily_correlations'])} daily correlations)")
    print(f"  • co_occurring_days ({len(analysis_results['co_occurring_days'])} problematic days)")
    print(f"  • rerouted_pairs_impact ({len(analysis_results['rerouted_pairs_impact'])} impacted pairs)")
else:
    print(f"\n❌ Analysis failed!")
    analysis_results = None


# ============================================================================
#  ANALYSIS & VISUALIZATION
# ============================================================================

def corr_analysis(analysis_results, unify_direction=False, pad_minutes=60):
    """
     analysis with normalized daily 'complexity score' and your visuals.
    - Uses simple same-day, same-pair ASN↔Performance co-occurrence (R in [0,1])
    - No traceroute coverage term
    """
    if not analysis_results:
        print("⚠️ No analysis results available")
        return

    alarms_df = analysis_results['alarms_df'].copy()

    print("📊  ANALYSIS - NORMALIZED SCORE")
    print("=" * 60)

    # Basic hygiene
    alarms_df = alarms_df.dropna(subset=['timestamp'])
    alarms_df['timestamp'] = pd.to_datetime(alarms_df['timestamp'], utc=True, errors='coerce')
    alarms_df = alarms_df.dropna(subset=['timestamp'])
    alarms_df['date'] = alarms_df['timestamp'].dt.date

    print(f"  • Alarms in range: {len(alarms_df)}")
    print(f"  • Date range: {alarms_df['timestamp'].min()} → {alarms_df['timestamp'].max()}")

    # Country mapping
    alarms_df = add_geography_to_dataframe(alarms_df)

    # Country involvement counts (vectorized)
    countries = pd.concat([alarms_df['src_country'], alarms_df['dest_country']], ignore_index=True).dropna()
    countries = countries[countries != 'Unknown']
    country_stats = countries.value_counts().to_dict()

    #  per-day metrics 
    days = []
    all_dates = sorted(alarms_df['date'].unique())
    K_types = int(alarms_df['event'].nunique())

    for d in all_dates:
        day_df = alarms_df[alarms_df['date'] == d].copy()

        total_alarms = len(day_df)
        unique_events = int(day_df['event'].nunique())
        unique_sites = len(set(day_df['src_site'].dropna()) | set(day_df['dest_site'].dropna()))
        unique_countries = len(set(day_df['src_country'].dropna()) | set(day_df['dest_country'].dropna()) - {'Unknown'})

        # unique-pair base for the day
        pair_base = (day_df.dropna(subset=['src_site','dest_site'])
                            .drop_duplicates(['event','src_site','dest_site']))

        # entropy-based event variety (0..1) over unique pairs
        event_evenness = _event_evenness_pairs(day_df, K_types)

        # routing vs performance UNIQUE-PAIR counts (your exact rule)
        asn_pairs  = pair_base[pair_base['event'] == REROUTING_EVENT]
        perf_pairs = pair_base[pair_base['event'] != REROUTING_EVENT]
        asn_alarms  = int(len(asn_pairs))          # #pairs with an ASN alarm
        perf_alarms = int(len(perf_pairs))         # #pairs with any perf alarm

        # same-day same-pair co-occurrence summary
        _, day_summary = same_day_pair_overlap_exact(day_df, unify_direction=unify_direction)
        if not day_summary.empty:
            R = float(day_summary.iloc[0]['R_asn'])                # 0..1
            has_corr = bool(day_summary.iloc[0]['has_cooccurrence'])
            n_pairs_both = int(day_summary.iloc[0]['n_pairs_both'])
        else:
            R, has_corr, n_pairs_both = 0.0, False, 0

        # severity proxy
        severity_value = _severity_proxy(day_df)

        days.append(dict(
            date=d,
            total_alarms=total_alarms,
            unique_events=unique_events,
            unique_sites=unique_sites,
            unique_countries=unique_countries,
            event_evenness=event_evenness,
            asn_alarms=asn_alarms,
            perf_alarms=perf_alarms,
            R=R,
            has_correlation=has_corr,
            pairs_with_both=n_pairs_both,
            severity_value=severity_value
        ))

    daily_df = pd.DataFrame(days)

    #  baselines for normalization (p05..p95) 
    def p05(x): return np.nanpercentile(x, 5)
    def p95(x): return np.nanpercentile(x, 95)

    v_lo, v_hi   = p05(daily_df['total_alarms']),    p95(daily_df['total_alarms'])
    s_lo, s_hi   = p05(daily_df['unique_sites']),    p95(daily_df['unique_sites'])
    c_lo, c_hi   = p05(daily_df['unique_countries']),p95(daily_df['unique_countries'])
    sev_lo, sev_hi = p05(daily_df['severity_value']),p95(daily_df['severity_value'])

    #  normalized components (0..1) 
    daily_df['V'] = daily_df['total_alarms'].apply(lambda x: _log_minmax(x, v_lo, v_hi))
    daily_df['E'] = daily_df['event_evenness']
    daily_df['S'] = daily_df['unique_sites'].apply(lambda x: _minmax(x, s_lo, s_hi))
    daily_df['C'] = daily_df['unique_countries'].apply(lambda x: _minmax(x, c_lo, c_hi))
    daily_df['P'] = daily_df['severity_value'].apply(lambda x: _minmax(x, sev_lo, sev_hi))
    daily_df['R'] = daily_df['R'].clip(0, 1)                                # ensure bounds

    k = 5
    daily_df['R_eff'] = daily_df.apply(
        lambda r: (r['R'] * (r['pairs_with_both'] / (r['pairs_with_both'] + k)))
                if r['pairs_with_both'] > 0 else 0.0,
        axis=1
    )
    #  composite score (weights sum to 1) 
    w = dict(V=0.30, P=0.20, E=0.15, S=0.10, C=0.05, R=0.20)
    daily_df['complexity_score'] = (
        w['V']*daily_df['V'] + w['P']*daily_df['P'] + w['E']*daily_df['E'] +
        w['S']*daily_df['S'] + w['C']*daily_df['C'] + w['R']*daily_df['R_eff']
    )

    #  console summary 
    top_complex_days = daily_df.nlargest(5, 'complexity_score')
    print(f"\n📊 Most Complex Days")
    for _, day in top_complex_days.iterrows():
        print(f"  • {day['date']}: Score={day['complexity_score']:.3f}  "
              f"[V={day['V']:.2f}, P={day['P']:.2f}, E={day['E']:.2f}, S={day['S']:.2f}, C={day['C']:.2f}, R={day['R_eff']:.2f}]  "
              f"alarms={day['total_alarms']} sites={day['unique_sites']} countries={day['unique_countries']} "
              f"co-occurring pairs={day['pairs_with_both']}")
              
    # Correlation insights
    corr_days = daily_df[daily_df['has_correlation']]
    high_corr_days = daily_df[daily_df['R'] > 0.3]
    avg_R = daily_df['R'].mean()
    max_pairs = daily_df['pairs_with_both'].max()
    
    print(f"\n🔗 Correlation Analysis Summary:")
    print(f"  • Days with correlations: {len(corr_days)}/{len(daily_df)} ({len(corr_days)/len(daily_df)*100:.1f}%)")
    print(f"  • Days with strong correlations (R>0.3): {len(high_corr_days)}")
    print(f"  • Average correlation strength: {avg_R:.3f}")
    print(f"  • Maximum co-occurring pairs in one day: {max_pairs}")
    print(f"  • Most correlated day: {daily_df.loc[daily_df['R'].idxmax(), 'date']} (R={daily_df['R'].max():.3f})")
    
    if len(high_corr_days) > 0:
        print(f"\n🎯 High Correlation Days (R > 0.3):")
        for _, day in high_corr_days.nlargest(3, 'R').iterrows():
            print(f"  • {day['date']}: R={day['R']:.3f}, {day['pairs_with_both']} pairs, {day['asn_alarms']} ASN + {day['perf_alarms']} perf alarms")

    #  PLOTS 
    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    fig.suptitle(' Alarm Analysis - Daily Variety Tracking', fontsize=16, y=0.98)

    # 1) Complexity score timeline
    ax = axes[0,0]
    dates_str = [str(d) for d in daily_df['date']]
    ax.plot(dates_str, daily_df['complexity_score'], 'o-', linewidth=2, markersize=4, color='red', alpha=0.7)
    corr_days = daily_df[daily_df['has_correlation']]
    if not corr_days.empty:
        ax.scatter([str(d) for d in corr_days['date']], corr_days['complexity_score'], color='orange',
                   s=60, zorder=10, alpha=0.8, label=f'ASN↔Perf co-occurrence ({len(corr_days)})')
        ax.legend()
    ax.set_title('Daily Network Complexity Score')
    ax.set_ylabel('Score (0–1)')
    ax.set_xlabel('Date'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    # 2) Variety metrics (stacked-ish view)
    ax = axes[0,1]
    ax.fill_between(dates_str, daily_df['unique_events'], alpha=0.5, label='Event Types', color='blue')
    ax.fill_between(dates_str, daily_df['unique_sites']*0.2, alpha=0.5, label='Sites (scaled)', color='green')
    ax.fill_between(dates_str, daily_df['unique_countries']*2.0, alpha=0.3, label='Countries (scaled)', color='red')
    ax.set_title('Daily Variety Metrics'); ax.set_ylabel('Count')
    ax.set_xlabel('Date'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7); ax.legend()

    # 3) ASN vs Performance volumes
    ax = axes[0,2]
    ax.bar(dates_str, daily_df['asn_alarms'], alpha=0.6, label='ASN Alarms', color='red', width=0.8)
    ax.bar(dates_str, daily_df['perf_alarms'], alpha=0.6, label='Performance Alarms', color='blue',
           width=0.8, bottom=daily_df['asn_alarms'])
    ax.set_title('Daily ASN vs Performance Alarms'); ax.set_ylabel('Alarm Count')
    ax.set_xlabel('Date'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7); ax.legend()

    # 4) Correlation Timeline - NEW PLOT
    ax = axes[1,0]
    # Primary axis: R values (correlation strength)
    ax.plot(dates_str, daily_df['R'], 'o-', linewidth=2, markersize=5, color='purple', alpha=0.8, label='Correlation Strength (R)')
    ax.fill_between(dates_str, daily_df['R'], alpha=0.3, color='purple')
    ax.set_ylabel('Correlation Strength (R)', color='purple')
    ax.set_title('Daily ASN↔Performance Correlations')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelcolor='purple')
    
    # Secondary axis: number of co-occurring pairs
    ax2 = ax.twinx()
    ax2.bar(dates_str, daily_df['pairs_with_both'], alpha=0.4, color='orange', width=0.6, label='Co-occurring Pairs')
    ax2.set_ylabel('# Co-occurring Pairs', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Highlight significant correlation days
    high_corr_days = daily_df[daily_df['R'] > 0.3]  # Threshold for significant correlation
    if not high_corr_days.empty:
        ax.scatter([str(d) for d in high_corr_days['date']], high_corr_days['R'], 
                  color='red', s=80, zorder=10, alpha=0.9, label=f'High Correlation Days ({len(high_corr_days)})')
    
    ax.set_xlabel('Date'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')

    # 5) Country involvement (top 8)
    ax = axes[1,1]
    countries_data = pd.Series(country_stats).sort_values(ascending=False).head(8)
    bars = ax.bar(range(len(countries_data)), countries_data.values, alpha=0.7, color='skyblue')
    ax.set_xticks(range(len(countries_data))); ax.set_xticklabels(countries_data.index, rotation=45, ha='right', fontsize=10)
    ax.set_title('Country Involvement in Alarms'); ax.set_ylabel('Alarm Count')
    for bar, value in zip(bars, countries_data.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{int(value)}',
                ha='center', va='bottom', fontsize=8)

    # 6) Top complex days
    ax = axes[1,2]
    top_8 = daily_df.nlargest(8, 'complexity_score')[['date','complexity_score']]
    bars = ax.barh(range(len(top_8)), top_8['complexity_score'], alpha=0.7, color='orange')
    ax.set_yticks(range(len(top_8))); ax.set_yticklabels([str(d)[-5:] for d in top_8['date']], fontsize=9)
    ax.set_title('Most Complex Days (normalized)'); ax.set_xlabel('Complexity Score (0–1)')
    for bar,val in zip(bars, top_8['complexity_score']):
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f'{val:.2f}',
                ha='left', va='center', fontsize=8)

    # 7) Site diversity vs alarm volume
    ax = axes[2,0]
    sc = ax.scatter(daily_df['total_alarms'], daily_df['unique_sites'],
                    c=daily_df['complexity_score'], s=150, alpha=0.6, cmap='viridis')
    ax.set_title('Site Diversity vs Alarm Volume'); ax.set_xlabel('Total Alarms'); ax.set_ylabel('Unique Sites')
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('Complexity Score (0–1)', fontsize=9)
    
    # 8) Correlation Patterns Analysis
    ax = axes[2,1]
    # Show correlation effectiveness: R vs number of ASN alarms
    scatter = ax.scatter(daily_df['asn_alarms'], daily_df['R'], 
                       c=daily_df['pairs_with_both'], s=100, alpha=0.7, cmap='viridis')
    ax.set_xlabel('ASN Alarms per Day')
    ax.set_ylabel('Correlation Strength (R)')
    ax.set_title('Correlation Effectiveness vs ASN Activity')
    cbar2 = plt.colorbar(scatter, ax=ax); cbar2.set_label('# Co-occurring Pairs', fontsize=9)
    
    # 9) Correlation Impact Timeline
    ax = axes[2,2]
    # Show when correlations have highest impact (R_eff)
    ax.plot(dates_str, daily_df['R_eff'], 'o-', linewidth=2, markersize=4, color='darkgreen', alpha=0.8)
    ax.fill_between(dates_str, daily_df['R_eff'], alpha=0.3, color='darkgreen')
    ax.set_ylabel('Effective Correlation (R_eff)')
    ax.set_title('Correlation Impact Over Time')
    ax.set_xlabel('Date'); plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    
    # Mark days with significant correlation impact
    impact_days = daily_df[daily_df['R_eff'] > daily_df['R_eff'].quantile(0.75)]  # Top 25%
    if not impact_days.empty:
        ax.scatter([str(d) for d in impact_days['date']], impact_days['R_eff'],
                  color='red', s=60, zorder=10, alpha=0.9, label=f'High Impact ({len(impact_days)})')
        ax.legend()

    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()
    
    # Print detailed descriptions after plots
    print("\n" + "="*80)
    print("📊 COMPLEXITY SCORE DESCRIPTION")
    print("="*80)
    print("""

INTERPRETATION:
• 0.0-0.3: Low complexity (routine network operations?)
• 0.3-0.6: Moderate complexity (localized issues)
• 0.6-0.8: High complexity (widespread instability)
• 0.8-1.0: Critical complexity (possibly major network incidents)

Higher scores indicate more complex, widespread network instability with diverse
alarm types affecting multiple sites and countries.
    """)


    # Convenience: compute coarse co-occurrence windows for the peak day
    peak_day = daily_df.loc[daily_df['complexity_score'].idxmax(), 'date']
    peak_day_df = alarms_df[alarms_df['date'] == peak_day].copy()
    co_windows = cooccurrence_windows_for_day_exact(peak_day_df, pad_minutes=pad_minutes,
                                                    unify_direction=unify_direction)

    return dict(
        valid_alarms=alarms_df,
        daily_metrics=daily_df,
        top_complex_days=top_complex_days,
        peak_day=peak_day,
        country_stats=country_stats,
        cooccurrence_windows_peak_day=co_windows
    )


def same_day_pair_overlap_exact(day_df, unify_direction=False):
    """Same-day, same-(src,dst) co-occurrence analysis."""
    df = day_df.dropna(subset=['src_site','dest_site']).copy()

    if unify_direction:
        pair = df[['src_site','dest_site']].apply(lambda r: tuple(sorted([r['src_site'], r['dest_site']])), axis=1)
        df['src_site'], df['dest_site'] = zip(*pair)

    df['is_asn']  = (df['event'] == REROUTING_EVENT)
    df['is_perf'] = ~df['is_asn']

    pairs_df = (df.groupby(['src_site','dest_site'], as_index=False)
                  .agg(has_asn = ('is_asn',  'any'),
                       has_perf= ('is_perf', 'any')))
    pairs_df['cooccur'] = pairs_df['has_asn'] & pairs_df['has_perf']

    n_asn_pairs   = int(pairs_df['has_asn'].sum())
    n_perf_pairs  = int(pairs_df['has_perf'].sum())
    n_both_pairs  = int(pairs_df['cooccur'].sum())
    n_union_pairs = int(n_asn_pairs + n_perf_pairs - n_both_pairs)

    summary = pd.DataFrame([{
        'n_pairs_asn': n_asn_pairs,
        'n_pairs_perf': n_perf_pairs,
        'n_pairs_both': n_both_pairs,
        'n_pairs_union': n_union_pairs,
        'R_asn':   (n_both_pairs / max(1, n_asn_pairs)),
        'R_union': (n_both_pairs / max(1, n_union_pairs)),
        'has_cooccurrence': (n_both_pairs > 0)
    }])

    return pairs_df, summary


def plot_correlation_heatmaps(alarms_df, target_date, figsize=(20, 8)):
    """
    Create site and country correlation heatmaps for a specific day.
    
    Site heatmap: Binary (1=red, 0=white) indicating co-occurrence
    Country heatmap: Count of co-occurring site pairs between countries
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    
    day_alarms = alarms_df[alarms_df['date'] == target_date].copy()
    
    if day_alarms.empty:
        print(f"⚠️ No alarm data found for {target_date}")
        return None, None
    
    # Add geography if not present
    if 'src_country' not in day_alarms.columns:
        day_alarms = add_geography_to_dataframe(day_alarms)
    
    # Get correlation data
    pairs_df, summary = same_day_pair_overlap_exact(day_alarms)
    
    if pairs_df.empty:
        print(f"⚠️ No site pairs found for {target_date}")
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Correlation Heatmaps for {target_date}', fontsize=16, y=0.98)
    
    # === SITE PAIR HEATMAP (BINARY) ===
    cooccur_pairs = pairs_df[pairs_df['cooccur']].copy()
    
    if not cooccur_pairs.empty:
        # Get unique sites
        all_sites = set(pairs_df['src_site'].unique()) | set(pairs_df['dest_site'].unique())
        all_sites = sorted([s for s in all_sites if pd.notna(s)])
        
        # Create BINARY correlation matrix (1=co-occurrence, 0=no co-occurrence)
        site_matrix = pd.DataFrame(0, index=all_sites, columns=all_sites)
        
        # Fill matrix with BINARY values
        for _, row in cooccur_pairs.iterrows():
            src, dest = row['src_site'], row['dest_site']
            if pd.notna(src) and pd.notna(dest):
                site_matrix.loc[src, dest] = 1  # RED for co-occurrence
                site_matrix.loc[dest, src] = 1  # Make symmetric
        
        # Plot with RED/WHITE colormap
        sns.heatmap(site_matrix, annot=False, cmap='Reds', vmin=0, vmax=1,
                   cbar_kws={'label': 'Co-occurrence', 'ticks': [0, 1]}, 
                   ax=ax1, square=True, linewidths=0.5)
        ax1.set_title(f'Site Pair Correlations\n({len(cooccur_pairs)} co-occurring pairs)', fontsize=12)
        ax1.set_xlabel('Destination Site')
        ax1.set_ylabel('Source Site')
        
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)
        
        site_heatmap_data = site_matrix
    else:
        ax1.text(0.5, 0.5, f'No co-occurring pairs\nfound for {target_date}', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Site Pair Correlations - No Data')
        site_heatmap_data = None
    
    # === COUNTRY PAIR HEATMAP (COUNT) ===
    day_alarms_clean = day_alarms.dropna(subset=['src_site', 'dest_site', 'src_country', 'dest_country'])
    
    if not day_alarms_clean.empty and not cooccur_pairs.empty:
        # Get co-occurring pairs with country info
        cooccur_with_geo = cooccur_pairs.merge(
            day_alarms_clean[['src_site', 'dest_site', 'src_country', 'dest_country']].drop_duplicates(),
            on=['src_site', 'dest_site'], how='left'
        )
        
        # Filter out unknown countries
        cooccur_with_geo = cooccur_with_geo[
            (cooccur_with_geo['src_country'] != 'Unknown') & 
            (cooccur_with_geo['dest_country'] != 'Unknown')
        ]
        
        if not cooccur_with_geo.empty:
            # COUNT pairs between countries
            country_counts = cooccur_with_geo.groupby(['src_country', 'dest_country']).size().reset_index(name='pair_count')
            
            # Get unique countries
            all_countries = set(country_counts['src_country'].unique()) | set(country_counts['dest_country'].unique())
            all_countries = sorted([c for c in all_countries if c != 'Unknown'])
            
            # Create country COUNT matrix
            country_matrix = pd.DataFrame(0, index=all_countries, columns=all_countries)
            
            # Fill matrix with COUNTS
            for _, row in country_counts.iterrows():
                src_country, dest_country, count = row['src_country'], row['dest_country'], row['pair_count']
                country_matrix.loc[src_country, dest_country] = count
                country_matrix.loc[dest_country, src_country] = count  # Make symmetric
            
            # Plot country heatmap with counts
            max_count = country_matrix.values.max()
            sns.heatmap(country_matrix, annot=True, cmap='Blues', fmt='d',
                       cbar_kws={'label': 'Co-occurring Pairs Count'}, 
                       ax=ax2, square=True, linewidths=0.5,
                       vmin=0, vmax=max_count)
            ax2.set_title(f'Country Pair Correlations\n(Total: {country_counts["pair_count"].sum()} pairs)', fontsize=12)
            ax2.set_xlabel('Destination Country')
            ax2.set_ylabel('Source Country')
            
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
            
            country_heatmap_data = country_matrix
        else:
            ax2.text(0.5, 0.5, f'No valid country pairs\nwith correlations', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Country Pair Correlations - No Data')
            country_heatmap_data = None
    else:
        ax2.text(0.5, 0.5, f'No geographic data\nor correlations available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Country Pair Correlations - No Data')
        country_heatmap_data = None
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    if not pairs_df.empty:
        print(f"\n📊 {target_date} Correlation Summary:")
        print(f"  • Total site pairs analyzed: {len(pairs_df)}")
        print(f"  • Co-occurring site pairs: {len(cooccur_pairs) if not cooccur_pairs.empty else 0}")
        if not summary.empty:
            print(f"  • Correlation strength (R): {summary.iloc[0]['R_asn']:.3f}")
        
        if site_heatmap_data is not None:
            sites_with_corr = (site_heatmap_data.sum(axis=1) > 0).sum()
            print(f"  • Sites involved in correlations: {sites_with_corr}")
        
        if country_heatmap_data is not None:
            total_country_pairs = int(country_heatmap_data.sum().sum() / 2)  # Divide by 2 since symmetric
            print(f"  • Total country-level co-occurring pairs: {total_country_pairs}")
    
    return site_heatmap_data, country_heatmap_data



# if __name__ == "__main__":
#     analysis_results = analyze_alarm_correlations(days_back=30)
    
#     if analysis_results:
#         print(f"Data available in 'analysis_results' with:")
#         print(f"  • alarms_df ({len(analysis_results['alarms_df'])} expanded alarms)")
#         print(f"  • daily_correlations ({len(analysis_results['daily_correlations'])} daily correlations)")
#         print(f"  • co_occurring_days ({len(analysis_results['co_occurring_days'])} problematic days)")
#         print(f"  • rerouted_pairs_impact ({len(analysis_results['rerouted_pairs_impact'])} impacted pairs)")
        
#         # Run  analysis if basic analysis succeeded
#         print("\n🚀 Running  analysis with daily variety tracking...")
#         daily_analysis_results = corr_analysis(analysis_results)
        
#         if daily_analysis_results:
#             print(f"\n✅  analysis complete!")
#             print(f"Daily metrics calculated for {len(daily_analysis_results['daily_metrics'])} days")
#             print(f"Peak complexity day: {daily_analysis_results['top_complex_days'].iloc[0]['date']}")
#         else:
#             print(f"\n❌  analysis failed!")
#     else:
#         print(f"\n❌ Analysis failed!")
#         analysis_results = None