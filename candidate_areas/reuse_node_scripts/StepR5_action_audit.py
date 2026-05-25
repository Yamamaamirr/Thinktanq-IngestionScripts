"""
StepR5_action_audit -- Breakdown of recommended_action distribution
and sanity-checks the Parcel Pull / Shortlist / Ignore lists row-by-row
to verify the action labels match the underlying attributes.

Output is a printed report. No file written.

Run: python candidate_areas/reuse_node_scripts/StepR5_action_audit.py
"""
from pathlib import Path
import geopandas as gpd
import pandas as pd


PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')


def main():
    g = gpd.read_parquet(PATH)

    print('=' * 72)
    print('RECOMMENDED ACTION BREAKDOWN')
    print('=' * 72)
    print(f'Total reuse nodes: {len(g):,}\n')

    action_order = ['Shortlist','Reuse Diligence','Parcel Pull','Monitor','Manual Review','Ignore']
    ac = g.recommended_action.value_counts()
    meanings = {
        'Shortlist':       'Top-tier -- ready to pitch / immediate diligence priority',
        'Reuse Diligence': 'Score >=60 AND contam/legacy flag -- ESA required first',
        'Parcel Pull':     'Score >=75 -- pull APN + owner before action',
        'Monitor':         'Score 65-74 -- track but not actionable now',
        'Manual Review':   'Failed kill-gate -- analyst inspection required',
        'Ignore':          'Score <65 -- not a viable candidate',
    }
    print(f'{"Action":<20} {"Count":>7} {"% total":>9}  {"Meaning":<55}')
    for a in action_order:
        n = int(ac.get(a, 0))
        print(f'  {a:<18} {n:>7,} {100*n/len(g):>8.1f}%  {meanings[a]}')

    actionable = int(ac.get('Shortlist',0) + ac.get('Reuse Diligence',0) + ac.get('Parcel Pull',0))
    print(f'\n>>> ACTIONABLE PIPELINE (Shortlist + Diligence + Parcel Pull): '
          f'{actionable:,} sites ({100*actionable/len(g):.1f}%)')
    print(f'>>> NOT ACTIONABLE (Monitor + Ignore + Manual Review): '
          f'{len(g)-actionable:,} sites ({100*(len(g)-actionable)/len(g):.1f}%)')

    # ---- SHORTLIST ----
    print('\n' + '=' * 72)
    print('SHORTLIST -- the 33 ready-to-pitch sites')
    print('=' * 72)
    sl = g[g.recommended_action == 'Shortlist'].sort_values('composite_score', ascending=False)
    print(f'Total: {len(sl)}\n')
    print('By source:')
    print(sl.source.value_counts().to_string())
    print('\nBy state:')
    print(sl.state.value_counts().to_string())
    print('\nFull list:')
    for _, r in sl.iterrows():
        nm = (str(r.site_name)[:28] if pd.notna(r.site_name) else '(unnamed)')
        pa = (str(r.primary_anchor_name)[:24] if pd.notna(r.primary_anchor_name) else '-')
        print(f'  {r.composite_score:>5.1f}  {r.state}  {r.candidate_id[:22]:<22}  {nm:<28}  '
              f'{r.area_acres:>5.0f}ac  {r.reuse_asset_type[:15]:<15}  '
              f'util={r.utility_score:>3.0f}  env={r.reuse_environmental_score:>3.0f}  '
              f'{r.num_anchors_in_range:>2}A -> {pa}')

    # ---- PARCEL PULL ----
    print('\n' + '=' * 72)
    print('PARCEL PULL -- 2,901 sites scoring 75-89')
    print('=' * 72)
    pp = g[g.recommended_action == 'Parcel Pull']
    print(f'Total: {len(pp):,}\n')
    print('By source:')
    print(pp.source.value_counts().to_string())
    print('\nBy state:')
    print(pp.state.value_counts().to_string())
    print('\nBy reuse_asset_type:')
    print(pp.reuse_asset_type.value_counts().to_string())
    print('\nAcreage band:')
    print(pd.cut(pp.area_acres, [0, 100, 250, 500, 1000, 5000],
                 labels=['50-100','100-250','250-500','500-1000','>1000']).value_counts().reindex(
        ['50-100','100-250','250-500','500-1000','>1000']).to_string())
    print('\nUtility distribution:')
    print(f'  util=100 (pinned):     {int((pp.utility_score == 100).sum()):>5,}')
    print(f'  util 50-99:            {int(((pp.utility_score >= 50) & (pp.utility_score < 100)).sum()):>5,}')
    print(f'  util <50:              {int((pp.utility_score < 50).sum()):>5,}')
    print('\nEnvironmental risk in Parcel Pull set:')
    print(f'  reuse_env=100 (clean):    {int((pp.reuse_environmental_score == 100).sum()):>5,}')
    print(f'  reuse_env 80-99:          {int(((pp.reuse_environmental_score >= 80) & (pp.reuse_environmental_score < 100)).sum()):>5,}')
    print(f'  reuse_env 60-79:          {int(((pp.reuse_environmental_score >= 60) & (pp.reuse_environmental_score < 80)).sum()):>5,}')
    print(f'  reuse_env <60:            {int((pp.reuse_environmental_score < 60).sum()):>5,}')

    print('\n--- 10 random Parcel Pull rows (sanity check) ---')
    sample = pp.sample(10, random_state=42).sort_values('composite_score', ascending=False)
    for _, r in sample.iterrows():
        nm = (str(r.site_name)[:24] if pd.notna(r.site_name) else '(unnamed)')
        flags = []
        if r.known_contamination_flag: flags.append('contam')
        if r.legacy_asset_risk_flag:   flags.append('legacy')
        if r.environmental_review_required: flags.append('env_review')
        print(f'  {r.composite_score:>5.1f}  {r.state}  {r.candidate_id[:22]:<22}  {nm:<24}  '
              f'{r.area_acres:>5.0f}ac  {r.reuse_asset_type[:15]:<15}  '
              f'util={r.utility_score:>3.0f} env={r.reuse_environmental_score:>3.0f}  '
              f'src={r.source[:6]:<6}  flags={flags if flags else "none"}')

    # ---- REUSE DILIGENCE ----
    print('\n' + '=' * 72)
    print('REUSE DILIGENCE -- 1,577 sites with contam/legacy + score >=60')
    print('=' * 72)
    rd = g[g.recommended_action == 'Reuse Diligence']
    print(f'Total: {len(rd):,}\n')
    print('By source:')
    print(rd.source.value_counts().to_string())
    print('\nWhat triggered the flag:')
    only_contam  = (rd.known_contamination_flag & ~rd.legacy_asset_risk_flag)
    only_legacy  = (~rd.known_contamination_flag & rd.legacy_asset_risk_flag)
    both         = (rd.known_contamination_flag & rd.legacy_asset_risk_flag)
    print(f'  contamination only:           {int(only_contam.sum()):>5,}')
    print(f'  legacy_asset only:            {int(only_legacy.sum()):>5,}')
    print(f'  both contam AND legacy:       {int(both.sum()):>5,}')
    print(f'\nScore band:')
    for band, mask in [
        ('60-70', (rd.composite_score >= 60) & (rd.composite_score < 70)),
        ('70-80', (rd.composite_score >= 70) & (rd.composite_score < 80)),
        ('80-90', (rd.composite_score >= 80) & (rd.composite_score < 90)),
        ('>=90',  rd.composite_score >= 90),
    ]:
        print(f'  {band:<6} {int(mask.sum()):>5,}')

    print('\n--- 5 random Reuse Diligence rows ---')
    sample = rd.sample(5, random_state=42)
    for _, r in sample.iterrows():
        nm = (str(r.site_name)[:28] if pd.notna(r.site_name) else '(unnamed)')
        flags = []
        if r.known_contamination_flag: flags.append('contam')
        if r.legacy_asset_risk_flag:   flags.append('legacy')
        prog = r.epa_program if pd.notna(r.epa_program) else '-'
        print(f'  {r.composite_score:>5.1f}  {r.state}  {r.candidate_id[:22]:<22}  {nm:<28}  '
              f'{r.area_acres:>5.0f}ac  {r.reuse_asset_type[:15]:<15}  src={r.source[:6]:<6}  '
              f'epa_prog={prog}  flags={flags}')

    # ---- IGNORE ----
    print('\n' + '=' * 72)
    print('IGNORE -- 1,635 sites scoring <65')
    print('=' * 72)
    ig = g[g.recommended_action == 'Ignore']
    print(f'Total: {len(ig):,}\n')
    print('By source:')
    print(ig.source.value_counts().to_string())
    print('\nBy reuse_asset_type:')
    print(ig.reuse_asset_type.value_counts().to_string())
    print('\nWhy ignored (root cause):')
    print(f'  utility_score == 0 (no anchors in 50km):  {int((ig.utility_score == 0).sum()):>5,}  '
          f'({100*(ig.utility_score == 0).mean():.0f}%)')
    print(f'  zone_fallback_used:                       {int(ig.zone_fallback_used.sum()):>5,}')
    print(f'  reuse_env_score <=50 (heavy contam):      {int((ig.reuse_environmental_score <= 50).sum()):>5,}')
    print(f'  acreage_tier=small (<100ac):              {int((ig.acreage_tier == "small").sum()):>5,}')

    print('\n--- 5 random Ignored rows ---')
    sample = ig.sample(5, random_state=42)
    for _, r in sample.iterrows():
        nm = (str(r.site_name)[:28] if pd.notna(r.site_name) else '(unnamed)')
        print(f'  {r.composite_score:>5.1f}  {r.state}  {r.candidate_id[:22]:<22}  {nm:<28}  '
              f'{r.area_acres:>5.0f}ac  {r.reuse_asset_type[:15]:<15}  src={r.source[:6]:<6}  '
              f'util={r.utility_score:>3.0f}  env={r.reuse_environmental_score:>3.0f}  '
              f'A={r.num_anchors_in_range}')

    # ---- INTERNAL CONSISTENCY CHECKS ----
    print('\n' + '=' * 72)
    print('INTERNAL CONSISTENCY -- do the action labels match attributes?')
    print('=' * 72)
    ok = True

    def chk(label, passed, detail=''):
        nonlocal ok
        sym = 'PASS' if passed else 'FAIL'
        print(f'  [{sym}] {label}', '  ' + detail if detail and not passed else '')
        if not passed: ok = False

    # 1. Every Parcel Pull should have composite >= 75
    pp_under = pp[pp.composite_score < 75]
    chk(f'All Parcel Pull have composite >= 75 (n={len(pp)})', len(pp_under) == 0,
        f'{len(pp_under)} under 75')

    # 2. No Parcel Pull should have a contam or legacy flag (those go to Reuse Diligence first)
    pp_flagged = pp[pp.known_contamination_flag | pp.legacy_asset_risk_flag]
    chk(f'No Parcel Pull has contam/legacy flag (should be Reuse Diligence instead)',
        len(pp_flagged) == 0, f'{len(pp_flagged)} flagged in Parcel Pull')

    # 3. Every Ignore should have composite < 65
    ig_over = ig[ig.composite_score >= 65]
    chk(f'All Ignore have composite < 65', len(ig_over) == 0)

    # 4. Every Shortlist should meet all 5 criteria
    sl_bad_size  = sl[(sl.area_acres < 500) | (sl.area_acres > 5000)]
    sl_bad_conf  = sl[~sl.confidence.isin(['medium','high'])]
    sl_bad_anch  = sl[sl.num_anchors_in_range < 3]
    sl_bad_score = sl[sl.composite_score < 90]
    chk('All Shortlist have 500-5000 ac', len(sl_bad_size) == 0)
    chk('All Shortlist have confidence in {medium, high}', len(sl_bad_conf) == 0)
    chk('All Shortlist have >=3 anchors', len(sl_bad_anch) == 0)
    chk('All Shortlist have composite >= 90', len(sl_bad_score) == 0)

    # 5. Every Reuse Diligence has contam OR legacy + composite >=60
    rd_no_flag = rd[~(rd.known_contamination_flag | rd.legacy_asset_risk_flag)]
    rd_under   = rd[rd.composite_score < 60]
    chk(f'All Reuse Diligence have contam OR legacy flag', len(rd_no_flag) == 0,
        f'{len(rd_no_flag)} without flag')
    chk('All Reuse Diligence have composite >= 60', len(rd_under) == 0)

    # 6. Coverage: every row has exactly one recommended_action
    chk('Every row has a recommended_action', g.recommended_action.notna().all())

    # 7. Are there sites that score very high BUT got Reuse Diligence (instead of Shortlist)?
    #    This would happen if e.g. they have contam/legacy + score >=90.
    rd_high = rd[rd.composite_score >= 90]
    print(f'\n  NOTE: {len(rd_high)} sites have composite >=90 but went to Reuse Diligence '
          f'(because contam/legacy flag took precedence over Shortlist)')
    if len(rd_high) > 0:
        print('  Top 3 such sites:')
        for _, r in rd_high.nlargest(3, 'composite_score').iterrows():
            nm = (str(r.site_name)[:30] if pd.notna(r.site_name) else '(unnamed)')
            print(f'    {r.composite_score:>5.1f}  {r.state}  {nm:<30}  '
                  f'contam={r.known_contamination_flag} legacy={r.legacy_asset_risk_flag}')

    # 8. Are there candidates being IGNORED that actually look promising? (sanity)
    #    E.g. score 60-65, large acreage, real OSM polygon -- might these be misclassified?
    near_miss = g[(g.composite_score >= 60) & (g.composite_score < 65) &
                  (g.area_acres >= 250) & (g.geometry_source == 'OSM_POLYGON')]
    print(f'\n  NOTE: {len(near_miss)} sites with composite 60-65, >=250 ac, real OSM polygon')
    print('  These score JUST under the Monitor threshold. Top 5:')
    for _, r in near_miss.nlargest(5, 'composite_score').iterrows():
        nm = (str(r.site_name)[:30] if pd.notna(r.site_name) else '(unnamed)')
        print(f'    {r.composite_score:>5.1f}  {r.state}  {nm:<30}  '
              f'{r.area_acres:>4.0f}ac  util={r.utility_score:>3.0f}  env={r.reuse_environmental_score:>3.0f}  '
              f'A={r.num_anchors_in_range}  -> {r.recommended_action}')

    print('\n' + '=' * 72)
    print('OVERALL:', 'ALL CONSISTENCY CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    print('=' * 72)


if __name__ == '__main__':
    main()
