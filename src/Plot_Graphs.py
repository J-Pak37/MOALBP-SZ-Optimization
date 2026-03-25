import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# ⚙️ CONFIGURATION & DATA LOADING
# ==========================================
# อ่านข้อมูลจากไฟล์ CSV ทั้ง 3 Group
df_A = pd.read_csv('Comparative_Metrics_GALBP_A.csv')
df_B = pd.read_csv('Comparative_Metrics_GALBP_B.csv')
df_C = pd.read_csv('Comparative_Metrics_GALBP_C.csv')

# คำนวณค่าเฉลี่ยของแต่ละ Algorithm ในแต่ละ Group
mean_A = df_A.groupby('Algorithm')[['HVR', 'Epsilon', 'Time_Sec']].mean().reset_index()
mean_B = df_B.groupby('Algorithm')[['HVR', 'Epsilon', 'Time_Sec']].mean().reset_index()
mean_C = df_C.groupby('Algorithm')[['HVR', 'Epsilon', 'Time_Sec']].mean().reset_index()

# จัดเรียงลำดับชื่อ Algorithm ให้เหมือนกันทุกกราฟ
alg_order = ['Exact', 'Hybrid', 'MOEAD', 'MOVNS', 'NSGA2', 'Pure']

for df in [mean_A, mean_B, mean_C]:
    df['Algorithm'] = pd.Categorical(df['Algorithm'], categories=alg_order, ordered=True)
    df.sort_values('Algorithm', inplace=True)

# กำหนดสไตล์ของกราฟ
sns.set_theme(style="whitegrid")
colors = sns.color_palette("Set2", len(alg_order))

# ==========================================
# 📊 กราฟที่ 1: Bar Chart เปรียบเทียบ HVR (3 รูป)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
groups = zip([mean_A, mean_B, mean_C], ['Group A (Small)', 'Group B (Medium)', 'Group C (Large)'])

for i, (mean_df, group_name) in enumerate(groups):
    sns.barplot(x='Algorithm', y='HVR', data=mean_df, ax=axes[i], palette=colors, order=alg_order)
    axes[i].set_title(f'Hypervolume Ratio - {group_name}', fontsize=14, fontweight='bold')
    axes[i].set_ylim(0, 1.05)
    axes[i].set_ylabel('HVR (Higher is better)', fontsize=12)
    axes[i].set_xlabel('Algorithm', fontsize=12)
    
    # ใส่ตัวเลขค่า HVR ไว้บนแท่งกราฟ
    for p in axes[i].patches:
        height = p.get_height()
        if not np.isnan(height):
            axes[i].annotate(f"{height:.3f}", 
                             (p.get_x() + p.get_width() / 2., height), 
                             ha='center', va='bottom', fontsize=11, color='black', 
                             xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('Chart_1_HVR_Comparison.png', dpi=300) # บันทึกรูปภาพความละเอียดสูง
print("✅ บันทึกรูปภาพ: Chart_1_HVR_Comparison.png")
plt.close()

# ==========================================
# 📈 กราฟที่ 2: Line Graph เปรียบเทียบเวลา (Time_Sec)
# ==========================================
# เตรียมข้อมูลสำหรับ Line Graph
mean_A['Group'] = 'Group A'
mean_B['Group'] = 'Group B'
mean_C['Group'] = 'Group C'
combined_time = pd.concat([mean_A, mean_B, mean_C])

plt.figure(figsize=(10, 6))
# วาดกราฟเส้น
sns.lineplot(data=combined_time, x='Group', y='Time_Sec', hue='Algorithm', 
             marker='o', linewidth=2.5, markersize=10, palette="Set1")

plt.title('Computational Time Across Problem Sizes', fontsize=16, fontweight='bold')
plt.ylabel('Time (Seconds) - Log Scale', fontsize=12)
plt.xlabel('Problem Size', fontsize=12)

# 💡 จุดสำคัญ: เราใช้ Log Scale แกน Y เพื่อให้เห็นความต่างของเวลาหลัก 0.001 วิ ไปจนถึง 500 วิ ได้ชัดเจน
plt.yscale('log')  
plt.grid(True, which="both", ls="--", alpha=0.5)

# ย้าย Legend ออกไปไว้นอกกราฟเพื่อไม่ให้บังเส้น
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
plt.tight_layout()

plt.savefig('Chart_2_Computational_Time.png', dpi=300) # บันทึกรูปภาพความละเอียดสูง
print("✅ บันทึกรูปภาพ: Chart_2_Computational_Time.png")
plt.close()