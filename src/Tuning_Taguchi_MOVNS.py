# -*- coding: utf-8 -*-
import os
import math
import pandas as pd
import numpy as np

# นำเข้า MO-VNS ที่เราเพิ่งอัปเดตไป
import MOVNS_soft_constraint as movns

# ==========================================
# ⚙️ 1. SETUP FACTORS & LEVELS
# ==========================================
FACTORS = {
    'MAX_NO_IMPROVE': [30, 50, 100],   # Factor A: จุดหยุดรัน
    'INIT_ARCHIVE': [10, 20, 50],      # Factor B: จำนวนประชากรตั้งต้น
    'SHAKING': [1, 2, 3],              # Factor C: ความแรงในการเขย่า
    'LS_LIMIT': [50, 100, 200]         # Factor D: จำนวนการค้นหาย่านใกล้เคียงต่อ 1 รอบ
}

L9_ARRAY = [
    [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2], 
    [1, 0, 1, 2], [1, 1, 2, 0], [1, 2, 0, 1], 
    [2, 0, 2, 1], [2, 1, 0, 2], [2, 2, 1, 0]  
]

# ใช้ไฟล์เดิมจาก Group B เพื่อความยุติธรรม
TEST_FILE = r'C:\GALBP\Hybrid VNS\GALBP_B\buxey_c=36.alb' 
NUM_REPS = 3 

# ==========================================
# 2. RUN EXPERIMENTS
# ==========================================
results = []

print("🚀 Starting Taguchi L9 Parameter Tuning for MO-VNS...")
print(f"📄 Test Instance: {os.path.basename(TEST_FILE)}\n")

for exp_idx, levels in enumerate(L9_ARRAY):
    max_no = FACTORS['MAX_NO_IMPROVE'][levels[0]]
    init_arch = FACTORS['INIT_ARCHIVE'][levels[1]]
    shake = FACTORS['SHAKING'][levels[2]]
    ls_limit = FACTORS['LS_LIMIT'][levels[3]]
    
    # อัปเดตพารามิเตอร์ของ MO-VNS
    movns.MAX_NO_IMPROVE = max_no
    movns.INIT_ARCHIVE_SIZE = init_arch
    movns.SHAKING_STRENGTH = shake
    movns.LS_LIMIT = ls_limit
    
    print(f"[Exp {exp_idx+1}/9] NO_IMPROVE={max_no}, INIT={init_arch}, SHAKE={shake}, LS={ls_limit}")
    
    exp_scores = []
    for rep in range(NUM_REPS):
        pareto_front, duration = movns.run_movns_for_instance(TEST_FILE)
        
        if pareto_front:
            # ดึงคำตอบจากคลัง (Archive) มาคำนวณคะแนน
            min_z1 = min(sol[1] for sol in pareto_front) # sol[1] คือ z1
            avg_z2 = sum(sol[2] for sol in pareto_front) / len(pareto_front) # sol[2] คือ z2
            score = (min_z1 * 50000) + avg_z2
            exp_scores.append(score)
        else:
            exp_scores.append(99999999)
            
    avg_score = np.mean(exp_scores)
    sn_ratio = -10 * math.log10(sum(s**2 for s in exp_scores) / NUM_REPS)
    
    results.append({
        'Exp': exp_idx + 1,
        'MAX_NO_IMPROVE': max_no,
        'INIT_ARCHIVE': init_arch,
        'SHAKING': shake,
        'LS_LIMIT': ls_limit,
        'Avg_Score': round(avg_score, 2),
        'SN_Ratio': round(sn_ratio, 2)
    })
    print(f"    -> Avg Score: {avg_score:.2f} | S/N Ratio: {sn_ratio:.2f}\n")

# ==========================================
# 3. SAVE L9 RESULTS
# ==========================================
df_results = pd.DataFrame(results)
output_path = r'C:\GALBP\Hybrid VNS\Taguchi_MOVNS_Results.csv'
df_results.to_csv(output_path, index=False)
print("✅ Taguchi Tuning Completed!")

# ==========================================
# 4. CALCULATE MAIN EFFECTS FOR PLOTTING
# ==========================================
print("\n📊 Calculating Main Effects for S/N Ratios...")
main_effects = []

for factor_name in ['MAX_NO_IMPROVE', 'INIT_ARCHIVE', 'SHAKING', 'LS_LIMIT']:
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
main_effects_path = r'C:\GALBP\Hybrid VNS\Taguchi_Main_Effects_MOVNS.csv'
df_main_effects.to_csv(main_effects_path, index=False)

print(f"✅ Main Effects summary saved to: {main_effects_path}")
print(df_main_effects.to_string(index=False))