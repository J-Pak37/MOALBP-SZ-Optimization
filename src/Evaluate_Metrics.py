# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = r'C:\GALBP\Hybrid VNS'
TARGET_FOLDER = 'GALBP_C'
OUTPUT_METRICS_FILE = f'Comparative_Metrics_{TARGET_FOLDER}.csv'

# ==========================================
# 1. CORE MATH FUNCTIONS (HV & Epsilon)
# ==========================================
def calculate_2d_hypervolume(front, ref_point=[1.1, 1.1]):
    if not front: return 0.0
    front_sorted = sorted(front, key=lambda x: x[0])
    hv = 0.0
    last_z2 = ref_point[1]
    for point in front_sorted:
        z1, z2 = point[0], point[1]
        if z1 > ref_point[0] or z2 > ref_point[1]:
            continue 
        width = ref_point[0] - z1
        height = last_z2 - z2
        hv += width * height
        last_z2 = z2
    return hv

def calculate_additive_epsilon(alg_front, ref_front):
    if not alg_front or not ref_front: return float('inf')
    epsilons = []
    for r in ref_front:
        min_diff = min(max(a[0] - r[0], a[1] - r[1]) for a in alg_front)
        epsilons.append(min_diff)
    return max(epsilons)

def get_non_dominated_front(points):
    unique_points = list({(p[0], p[1]) for p in points})
    pareto = []
    for p in unique_points:
        is_dominated = False
        for q in unique_points:
            if (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(list(p))
    return pareto

# ==========================================
# 2. MAIN EVALUATION SCRIPT
# ==========================================
def evaluate_all_algorithms():
    print(f"📊 Starting Evaluation for {TARGET_FOLDER}...")
    
    search_pattern = os.path.join(BASE_DIR, f"Result_*_Soft_{TARGET_FOLDER}.csv")
    result_files = glob.glob(search_pattern)
    
    if not result_files:
        print("❌ No result files found!")
        return
        
    all_data = []
    alg_names = []
    
    for file in result_files:
        alg_name = os.path.basename(file).split('_')[1]
        if alg_name not in alg_names: alg_names.append(alg_name)
        
        df = pd.read_csv(file)
        
        # 🌟 จุดที่แก้ไข: ปรับชื่อคอลัมน์ของ Hybrid/VNS ให้ตรงกับตระกูล MO
        if 'Z1_Best' in df.columns:
            df = df.rename(columns={
                'Z1_Best': 'Z1_Pareto', 
                'Z2_Best': 'Z2_Pareto', 
                'Time_Avg': 'Time_Sec'
            })
            
        df['Algorithm'] = alg_name
        all_data.append(df)
        
    combined_df = pd.concat(all_data, ignore_index=True)
    instances = combined_df['Instance'].unique()
    
    metrics_results = []
    
    for instance in instances:
        instance_data = combined_df[combined_df['Instance'] == instance]
        
        # ดึงค่ามาคำนวณ Min/Max โดยตัด NaN ทิ้ง (กันเหนียว)
        all_z1 = instance_data['Z1_Pareto'].dropna().values
        all_z2 = instance_data['Z2_Pareto'].dropna().values
        
        if len(all_z1) == 0 or len(all_z2) == 0:
            continue
            
        min_z1, max_z1 = all_z1.min(), all_z1.max()
        min_z2, max_z2 = all_z2.min(), all_z2.max()
        
        range_z1 = (max_z1 - min_z1) if max_z1 > min_z1 else 1.0
        range_z2 = (max_z2 - min_z2) if max_z2 > min_z2 else 1.0
        
        def normalize(z1, z2):
            return [(z1 - min_z1) / range_z1, (z2 - min_z2) / range_z2]
            
        # สร้าง Reference Set (True Pareto) จากทุกอัลกอริทึม
        all_normalized_points = []
        for _, row in instance_data.iterrows():
            if pd.notna(row['Z1_Pareto']) and pd.notna(row['Z2_Pareto']):
                all_normalized_points.append(normalize(row['Z1_Pareto'], row['Z2_Pareto']))
                
        reference_front = get_non_dominated_front(all_normalized_points)
        hv_ref = calculate_2d_hypervolume(reference_front)
        
        for alg in alg_names:
            alg_data = instance_data[instance_data['Algorithm'] == alg]
            if alg_data.empty: continue
                
            alg_norm_points = []
            for _, row in alg_data.iterrows():
                if pd.notna(row['Z1_Pareto']) and pd.notna(row['Z2_Pareto']):
                    alg_norm_points.append(normalize(row['Z1_Pareto'], row['Z2_Pareto']))
                    
            if not alg_norm_points:
                continue
                
            alg_front = get_non_dominated_front(alg_norm_points)
            
            hv_alg = calculate_2d_hypervolume(alg_front)
            hvr = hv_alg / hv_ref if hv_ref > 0 else 1.0
            epsilon = calculate_additive_epsilon(alg_front, reference_front)
            avg_time = alg_data['Time_Sec'].mean()
            
            metrics_results.append({
                'Instance': instance,
                'Algorithm': alg,
                'Pareto_Size': len(alg_front),
                'HVR': round(hvr, 4),
                'Epsilon': round(epsilon, 4),
                'Time_Sec': round(avg_time, 4)
            })
            
    df_metrics = pd.DataFrame(metrics_results)
    output_path = os.path.join(BASE_DIR, OUTPUT_METRICS_FILE)
    df_metrics.to_csv(output_path, index=False)
    
    print(f"\n✅ Evaluation Completed! Saved to: {OUTPUT_METRICS_FILE}")
    
    summary = df_metrics.groupby('Algorithm')[['HVR', 'Epsilon', 'Time_Sec']].mean().reset_index()
    print("\n🏆 SUMMARY OF ALGORITHM PERFORMANCE (AVERAGE):")
    print("-" * 50)
    print(summary.to_string(index=False))
    print("-" * 50)

if __name__ == "__main__":
    evaluate_all_algorithms()