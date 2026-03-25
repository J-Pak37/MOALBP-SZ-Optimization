# -*- coding: utf-8 -*-
import os
import math
import pandas as pd
import numpy as np

# นำเข้า NSGA-II ที่เราเพิ่งเขียนไป (อ้างอิงชื่อไฟล์ให้ตรงกันนะครับ)
import NSGA2_soft_constraint as nsga2

# ==========================================
# ⚙️ 1. SETUP FACTORS & LEVELS
# ==========================================
# เราจะจูน 4 ปัจจัย ปัจจัยละ 3 ระดับ
FACTORS = {
    'POP_SIZE': [30, 50, 100],         # Factor A: ขนาดประชากร
    'MAX_GEN': [50, 100, 200],         # Factor B: จำนวนรุ่น (Generation)
    'CX_RATE': [0.8, 0.9, 1.0],        # Factor C: อัตราการข้ามสายพันธุ์
    'MUT_RATE': [0.05, 0.1, 0.2]       # Factor D: อัตราการกลายพันธุ์
}

# L9 Orthogonal Array (แถว=การทดลองที่ 1-9, คอลัมน์=A, B, C, D)
# เลข 0, 1, 2 คือ Index ของ List ใน FACTORS
L9_ARRAY = [
    [0, 0, 0, 0], # Exp 1: A1, B1, C1, D1
    [0, 1, 1, 1], # Exp 2: A1, B2, C2, D2
    [0, 2, 2, 2], # Exp 3: A1, B3, C3, D3
    [1, 0, 1, 2], # Exp 4: A2, B1, C2, D3
    [1, 1, 2, 0], # Exp 5: A2, B2, C3, D1
    [1, 2, 0, 1], # Exp 6: A2, B3, C1, D2
    [2, 0, 2, 1], # Exp 7: A3, B1, C3, D2
    [2, 1, 0, 2], # Exp 8: A3, B2, C1, D3
    [2, 2, 1, 0]  # Exp 9: A3, B3, C2, D1
]

# ไฟล์ตัวแทนที่ใช้รัน (ควรเลือกไฟล์ระดับกลางจาก Group B หรือ A)
TEST_FILE = r'C:\GALBP\Hybrid VNS\GALBP_B\buxey_c=36.alb' 
NUM_REPS = 3 # รันซ้ำ 3 รอบต่อ 1 การทดลอง เพื่อหาค่าเฉลี่ยที่เสถียร

# ==========================================
# 2. RUN EXPERIMENTS
# ==========================================
results = []

print("🚀 Starting Taguchi L9 Parameter Tuning for NSGA-II...")
print(f"📄 Test Instance: {os.path.basename(TEST_FILE)}\n")

for exp_idx, levels in enumerate(L9_ARRAY):
    # ดึงค่าพารามิเตอร์ตาม L9 Array
    pop = FACTORS['POP_SIZE'][levels[0]]
    gen = FACTORS['MAX_GEN'][levels[1]]
    cx = FACTORS['CX_RATE'][levels[2]]
    mut = FACTORS['MUT_RATE'][levels[3]]
    
    # อัปเดตค่าพารามิเตอร์เข้าไปในโมดูล NSGA-II ดื้อๆ เลย
    nsga2.POPULATION_SIZE = pop
    nsga2.MAX_GENERATIONS = gen
    nsga2.CROSSOVER_RATE = cx
    nsga2.MUTATION_RATE = mut
    
    print(f"[Exp {exp_idx+1}/9] POP={pop}, GEN={gen}, CX={cx}, MUT={mut}")
    
    exp_scores = []
    
    # รันซ้ำเพื่อลดความแกว่ง (Stochastic nature of GA)
    for rep in range(NUM_REPS):
        pareto_front, duration = nsga2.run_nsga2_for_instance(TEST_FILE)
        
        # --- ประเมินคุณภาพของ Pareto Front แบบเร็วๆ (Simplified Metric) ---
        # ในเปเปอร์เราจะใช้ HVR แต่นี่คือการประเมินเพื่อหา Parameter ที่ดีที่สุด
        # เราอยากได้ "สถานีน้อยๆ" และ "Cost ต่ำๆ"
        if pareto_front:
            # ดึงค่าที่สถานีน้อยที่สุด (Best Z1)
            min_z1 = min(sol['Z1'] for sol in pareto_front)
            # หาค่าเฉลี่ยของ Z2 ใน Pareto นั้นๆ
            avg_z2 = sum(sol['Z2'] for sol in pareto_front) / len(pareto_front)
            
            # สร้างสมการรวบยอด (Score = (Min Z1 * 50000) + Avg Z2)
            # ยิ่ง Score ต่ำ ยิ่งแปลว่าเส้น Pareto พุ่งเข้าหาจุดอุดมคติ (0,0) ได้ดี
            score = (min_z1 * 50000) + avg_z2
            exp_scores.append(score)
        else:
            exp_scores.append(99999999) # พัง
            
    # คำนวณ S/N Ratio แบบ Smaller-is-Better (ค่าน้อยยิ่งดี)
    avg_score = np.mean(exp_scores)
    sn_ratio = -10 * math.log10(sum(s**2 for s in exp_scores) / NUM_REPS)
    
    results.append({
        'Exp': exp_idx + 1,
        'POP_SIZE': pop,
        'MAX_GEN': gen,
        'CX_RATE': cx,
        'MUT_RATE': mut,
        'Avg_Score': round(avg_score, 2),
        'SN_Ratio': round(sn_ratio, 2)
    })
    print(f"    -> Avg Score: {avg_score:.2f} | S/N Ratio: {sn_ratio:.2f}\n")

# ==========================================
# 3. SAVE AND DISPLAY
# ==========================================
df_results = pd.DataFrame(results)
output_path = r'C:\GALBP\Hybrid VNS\Taguchi_NSGA2_Results.csv'
df_results.to_csv(output_path, index=False)

print("✅ Taguchi Tuning Completed!")
print(df_results.to_string(index=False))

# หาการทดลองที่ดีที่สุด (S/N Ratio สูงสุด แปลว่าเสถียรและเก่งสุด)
best_exp = df_results.loc[df_results['SN_Ratio'].idxmax()]
print("\n🏆 BEST PARAMETER COMBINATION:")
print(f"POP_SIZE: {best_exp['POP_SIZE']}")
print(f"MAX_GEN: {best_exp['MAX_GEN']}")
print(f"CX_RATE: {best_exp['CX_RATE']}")
print(f"MUT_RATE: {best_exp['MUT_RATE']}")

# ==========================================
# 4. CALCULATE MAIN EFFECTS FOR PLOTTING
# ==========================================
print("\n📊 Calculating Main Effects for S/N Ratios...")

main_effects = []

# วนลูปหาค่าเฉลี่ย S/N Ratio และ Score ของแต่ละ Factor (A, B, C, D) ในแต่ละ Level (1, 2, 3)
for factor_name in ['POP_SIZE', 'MAX_GEN', 'CX_RATE', 'MUT_RATE']:
    for level_idx, level_val in enumerate(FACTORS[factor_name]):
        
        # กรองข้อมูลเฉพาะแถวที่ Factor นี้มีค่าตรงกับ Level ปัจจุบัน
        subset = df_results[df_results[factor_name] == level_val]
        
        # คำนวณค่าเฉลี่ย
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

# เซฟไฟล์สำหรับเอาไป Plot กราฟ
main_effects_path = r'C:\GALBP\Hybrid VNS\Taguchi_Main_Effects_NSGA2.csv'
df_main_effects.to_csv(main_effects_path, index=False)

print(f"✅ Main Effects summary saved to: {main_effects_path}")
print(df_main_effects.to_string(index=False))