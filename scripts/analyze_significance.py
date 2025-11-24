
import json
import argparse
import numpy as np
from scipy.stats import chi2_contingency, binomtest

def calculate_mcnemar(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)

    # We focus on the 'gpt-4o' model for both architectures as per the user's request
    sj_results = data.get('spot_and_judge', {}).get('gpt-4o', [])
    base_results = data.get('baseline_e2e', {}).get('gpt-4o', [])

    if not sj_results or not base_results:
        print("Error: Could not find results for gpt-4o in both architectures.")
        return

    # Organize by sample_id to ensure pairing
    sj_by_id = {}
    for r in sj_results:
        if r['sample_id'] not in sj_by_id:
            sj_by_id[r['sample_id']] = {}
        sj_by_id[r['sample_id']][r['position_order']] = r

    base_by_id = {}
    for r in base_results:
        if r['sample_id'] not in base_by_id:
            base_by_id[r['sample_id']] = {}
        base_by_id[r['sample_id']][r['position_order']] = r

    # 1. Consistent Accuracy (CA) Analysis
    # CA = Correct in BOTH win_first AND win_second
    
    # Contingency Table for CA
    #               | Base CA (Yes) | Base CA (No)
    # S&J CA (Yes)  |      a        |      b
    # S&J CA (No)   |      c        |      d
    
    a = 0 # Both Consistent
    b = 0 # S&J Consistent, Base Not
    c = 0 # Base Consistent, S&J Not
    d = 0 # Both Not Consistent

    valid_samples = 0

    for sample_id in sj_by_id:
        if sample_id not in base_by_id:
            continue
            
        sj_pair = sj_by_id[sample_id]
        base_pair = base_by_id[sample_id]
        
        if 'win_first' not in sj_pair or 'win_second' not in sj_pair:
            continue
        if 'win_first' not in base_pair or 'win_second' not in base_pair:
            continue

        valid_samples += 1

        # Check Consistency for S&J
        sj_consistent = sj_pair['win_first']['correct'] and sj_pair['win_second']['correct']
        
        # Check Consistency for Baseline
        base_consistent = base_pair['win_first']['correct'] and base_pair['win_second']['correct']

        if sj_consistent and base_consistent:
            a += 1
        elif sj_consistent and not base_consistent:
            b += 1
        elif not sj_consistent and base_consistent:
            c += 1
        else:
            d += 1

    print(f"\nAnalysis on {valid_samples} paired samples:")
    print("-" * 40)
    print(f"Spot & Judge CA: {(a+b)/valid_samples:.2%} ({a+b}/{valid_samples})")
    print(f"Baseline CA:     {(a+c)/valid_samples:.2%} ({a+c}/{valid_samples})")
    print("-" * 40)
    print("McNemar's Test Contingency Table (Consistent Accuracy):")
    print(f"                 | Base CA (Yes) | Base CA (No)")
    print(f"S&J CA (Yes)     | {a:^13} | {b:^12}")
    print(f"S&J CA (No)      | {c:^13} | {d:^12}")
    print("-" * 40)

    # Calculate McNemar's Statistic
    # (b - c)^2 / (b + c)
    if b + c > 0:
        chi2 = (abs(b - c) - 1)**2 / (b + c) # Continuity correction
        p_value = 1 - chi2_contingency([[a, b], [c, d]])[1] # This is not quite right for McNemar, calculating manually
        
        # Using exact binomial test for small numbers, or chi2 for large
        # For N > 25, chi2 is fine. Here N=300, so b+c likely > 25.
        
        from scipy.stats import chi2 as chi2_dist
        p_value = chi2_dist.sf(chi2, 1)
        
        print(f"McNemar's chi-squared statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4e}")
        
        if p_value < 0.05:
            print(">> Result is STATISTICALLY SIGNIFICANT (p < 0.05)")
        else:
            print(">> Result is NOT statistically significant (p >= 0.05)")
    else:
        print("Cannot calculate McNemar's: b + c = 0")

    # 2. Overall Accuracy Analysis (Treating all 600 evaluations as independent - technically incorrect for paired data, 
    # but we can look at the paired difference in accuracy per sample)
    
    # Let's look at average accuracy per sample (0, 0.5, or 1.0) and do a paired t-test
    sj_scores = []
    base_scores = []
    
    for sample_id in sj_by_id:
        if sample_id not in base_by_id: continue
        
        sj_pair = sj_by_id[sample_id]
        base_pair = base_by_id[sample_id]
        
        if 'win_first' not in sj_pair or 'win_second' not in sj_pair: continue
        if 'win_first' not in base_pair or 'win_second' not in base_pair: continue
        
        sj_score = (int(sj_pair['win_first']['correct']) + int(sj_pair['win_second']['correct'])) / 2.0
        base_score = (int(base_pair['win_first']['correct']) + int(base_pair['win_second']['correct'])) / 2.0
        
        sj_scores.append(sj_score)
        base_scores.append(base_score)
        
    from scipy.stats import ttest_rel
    t_stat, p_val_ttest = ttest_rel(sj_scores, base_scores)
    
    print("\n" + "="*40)
    print("Overall Accuracy Analysis (Paired t-test on sample scores)")
    print("-" * 40)
    print(f"S&J Mean Accuracy:  {np.mean(sj_scores):.2%}")
    print(f"Base Mean Accuracy: {np.mean(base_scores):.2%}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val_ttest:.4e}")
    
    if p_val_ttest < 0.05:
        print(">> Result is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print(">> Result is NOT statistically significant (p >= 0.05)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Path to results JSON file")
    args = parser.parse_args()
    
    calculate_mcnemar(args.results_file)
