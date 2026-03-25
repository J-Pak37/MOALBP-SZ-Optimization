# -*- coding: utf-8 -*-
import os
import math
import pandas as pd
import numpy as np

# นำเข้า MOEA/D ที่เราเขียนไว้
import MOEAD_soft_constraint as moead

# ==========================================
# ⚙️ 1. SETUP FACTORS & LEVELS
# ==========================================
FACTORS = {
    'POP_SIZE': [30, 50, 100],         # Factor A: ขนาดประชากร
    'MAX_GEN': [50, 100, 200],         # Factor B: จำนวนรุ่น
    'T_SIZE': [5, 10, 15],             # Factor C: ขนาดเพื่อนบ้าน (Neighborhood Size)
    'MUT_RATE': [0.05, 0.1, 0.2]       # Factor D: อัตราการกลายพันธุ์
}

# L9 Orthogonal Array (แถว=การทดลองที่ 1-9, คอลัมน์=A, B, C, D)
L9_ARRAY = [
    [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2], 
    [1, 0, 1, 2], [1, 1, 2, 0], [1, 2, 0, 1], 
    [2, 0, 2, 1], [2, 1, 0, 2], [2, 2, 1, 0]  
]

# ใช้ไฟล์เดิมจาก Group B เพื่อความยุติธรรมในการเปรียบเทียบ
TEST_FILE = r'C:\GALBP\Hybrid VNS\GALBP_B\buxey_c=36.alb' 
NUM_REPS = 3 

# ==========================================
# 2. RUN EXPERIMENTS
# ==========================================
results = []

print("🚀 Starting Taguchi L9 Parameter Tuning for MOEA/D...")
print(f"📄 Test Instance: {os.path.basename(TEST_FILE)}\n")

for exp_idx, levels in enumerate(L9_ARRAY):
    pop = FACTORS['POP_SIZE'][levels[0]]
    gen = FACTORS['MAX_GEN'][levels[1]]
    t_size = FACTORS['T_SIZE'][levels[2]]
    mut = FACTORS['MUT_RATE'][levels[3]]
    
    # ล็อค Crossover Rate ไว้ที่ 1.0 (จากบทเรียน NSGA-II)
    moead.CROSSOVER_RATE = 1.0
    
    # อัปเดตพารามิเตอร์ของ MOEA/D
    moead.POPULATION_SIZE = pop
    moead.MAX_GENERATIONS = gen
    moead.NEIGHBORHOOD_SIZE = t_size
    moead.MUTATION_RATE = mut
    
    print(f"[Exp {exp_idx+1}/9] POP={pop}, GEN={gen}, T={t_size}, MUT={mut}")
    
    exp_scores = []
    for rep in range(NUM_REPS):
        pareto_front, duration = moead.run_moead_for_instance(TEST_FILE)
        
        if pareto_front:
            min_z1 = min(sol['Z1'] for sol in pareto_front)
            avg_z2 = sum(sol['Z2'] for sol in pareto_front) / len(pareto_front)
            score = (min_z1 * 50000) + avg_z2
            exp_scores.append(score)
        else:
            exp_scores.append(99999999)
            
    avg_score = np.mean(exp_scores)
    sn_ratio = -10 * math.log10(sum(s**2 for s in exp_scores) / NUM_REPS)
    
    results.append({
        'Exp': exp_idx + 1,
        'POP_SIZE': pop,
        'MAX_GEN': gen,
        'T_SIZE': t_size,
        'MUT_RATE': mut,
        'Avg_Score': round(avg_score, 2),
        'SN_Ratio': round(sn_ratio, 2)
    })
    print(f"    -> Avg Score: {avg_score:.2f} | S/N Ratio: {sn_ratio:.2f}\n")

# ==========================================
# 3. SAVE L9 RESULTS
# ==========================================
df_results = pd.DataFrame(results)
output_path = r'C:\GALBP\Hybrid VNS\Taguchi_MOEAD_Results.csv'
df_results.to_csv(output_path, index=False)
print("✅ Taguchi Tuning Completed!")

# ==========================================
# 4. CALCULATE MAIN EFFECTS FOR PLOTTING
# ==========================================
print("\n📊 Calculating Main Effects for S/N Ratios...")
main_effects = []

for factor_name in ['POP_SIZE', 'MAX_GEN', 'T_SIZE', 'MUT_RATE']:
    for level_idx, level_val in enumerate(FACTORS[factor_name]):
        subset = df_results[df_results[factor_name] == level_val]
        avg_sn = subset['SN_Ratio'].mean()
        avg_score = subset['Avg_Score'].mean()
        
        main_effects.append({
            'Factor': factor_name,
            'Level': f"Level {level_idx + 1}",
            'Value': level_val,
            'Mean_Score': round(avg_score, 2),
            'Avg_SN_Ratio': round(avg_sn, 2)
        })

df_main_effects = pd.DataFrame(main_effects)
main_effects_path = r'C:\GALBP\Hybrid VNS\Taguchi_Main_Effects_MOEAD.csv'
df_main_effects.to_csv(main_effects_path, index=False)

print(f"✅ Main Effects summary saved to: {main_effects_path}")
print(df_main_effects.to_string(index=False))