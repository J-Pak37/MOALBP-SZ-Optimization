import pandas as pd

# ระบุรายการไฟล์ข้อมูลและกลุ่มที่ต้องการเปรียบเทียบ
files = [
    ('Group A', 'Hard', 'Result_Hybrid_VNS_Hard_GALBP_A.csv'),
    ('Group A', 'Soft', 'Result_Hybrid_VNS_Soft_GALBP_A.csv'),
    ('Group B', 'Hard', 'Result_Hybrid_VNS_Hard_GALBP_B.csv'),
    ('Group B', 'Soft', 'Result_Hybrid_VNS_Soft_GALBP_B.csv'),
    ('Group C', 'Hard', 'Result_Hybrid_VNS_Hard_GALBP_C.csv'),
    ('Group C', 'Soft', 'Result_Hybrid_VNS_Soft_GALBP_C.csv')
]

results = []

for group, const, file in files:
    try:
        # อ่านไฟล์ CSV
        df = pd.read_csv(file)
        
        # ทำความสะอาดข้อมูลคอลัมน์ Feasibility_Rate (ตัดเครื่องหมาย % และแปลงเป็นตัวเลข)
        if df['Feasibility_Rate'].dtype == object:
            df['Feasibility_Rate'] = df['Feasibility_Rate'].str.replace('%', '').astype(float)
            
        # คำนวณค่าสถิติ
        total = len(df)
        avg_feas = df['Feasibility_Rate'].mean()
        avg_time = df['Time_Avg'].mean()
        
        # นับจำนวนสถานะของการประมวลผล
        success = len(df[df['Feasibility_Rate'] == 100])
        deadlock = len(df[df['Feasibility_Rate'] == 0])
        partial = len(df[(df['Feasibility_Rate'] > 0) & (df['Feasibility_Rate'] < 100)])
        
        results.append({
            'Problem Group': group,
            'Constraint Type': const,
            'Instances Count': total,
            'Avg Feasibility (%)': round(avg_feas, 2),
            'Avg Time (s)': round(avg_time, 4),
            'Success (100%)': success,
            'Partial Feasible (<100%)': partial,
            'Deadlock (0%)': deadlock
        })
    except Exception as e:
        print(f"Error processing {file}: {e}")

# สร้าง DataFrame และแสดงผล
summary_df = pd.DataFrame(results)
print(summary_df.to_markdown(index=False))

# ส่งออกเป็นไฟล์ CSV (ถ้าต้องการ)
summary_df.to_csv("Summary_Hard_vs_Soft.csv", index=False)