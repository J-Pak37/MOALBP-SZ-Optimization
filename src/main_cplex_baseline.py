
# ==========================================
# 0. SETUP & LIBRARIES
# ==========================================
import os
import glob
import time
import pandas as pd
import collections
from docplex.mp.model import Model
import re 

# ==========================================
# 1. PARSER (Robust Version - แก้ Error อ่าน Tag ผิด)
# ==========================================
def parse_new_benchmark_format(filepath):
    with open(filepath, 'r') as f:
        # ใช้ splitlines เพื่อตัดปัญหาเรื่อง \r\n หรือ \n
        lines = f.read().splitlines()
    
    iterator = iter(lines)
    data = {
        'num_tasks': 0,
        'cycle_time': 0,
        'task_modes': collections.defaultdict(list), 
        'zoning': [],
        'precedence': [],
        'preds': collections.defaultdict(list),
        'succs': collections.defaultdict(list)
    }
    
    current_section = None
    
    # --- Helper Function: อ่านค่าตัวเลขแบบปลอดภัย ---
    # ฟังก์ชันนี้จะวนหาบรรทัดถัดไปเรื่อยๆ จนกว่าจะเจอตัวเลข 
    # ถ้าไปเจอ Tag อื่นก่อน (<...) แสดงว่าไม่มีค่า ก็จะคืน 0 ไม่ Error
    def get_value_robust(current_line, iterator):
        # 1. ลองดูในบรรทัดเดียวกับ Tag ก่อน (เช่น <number of tasks> 50)
        temp = re.sub(r'<[^>]+>', '', current_line).strip()
        if temp.isdigit():
            return int(temp)
        
        # 2. ถ้าไม่มี ให้วนหาในบรรทัดถัดไป
        while True:
            line = next(iterator, None)
            if line is None: return 0 # จบไฟล์
            line = line.strip()
            if not line: continue # ข้ามบรรทัดว่าง
            
            if line.isdigit():
                return int(line) # เจอตัวเลขแล้ว!
            
            # ถ้าเจอ Tag อื่น (เช่น <optimal...>) แสดงว่าค่าหายไป ให้หยุดหา
            if line.startswith('<'): 
                return 0

    try:
        while True:
            line = next(iterator, None)
            if line is None: break
            line = line.strip()
            if not line: continue
            
            # --- Detect Headers ---
            # ใช้ get_value_robust แทน int(next(...))
            if '<number of tasks>' in line:
                data['num_tasks'] = get_value_robust(line, iterator)
            
            elif '<cycle time>' in line:
                data['cycle_time'] = get_value_robust(line, iterator)
                
            elif '<task times>' in line:
                current_section = 'ignore' 
                
            elif '<task process alternatives>' in line:
                current_section = 'alternatives'
                
            elif '<incompatible tasks>' in line:
                current_section = 'zoning'
                
            elif '<precedence relations>' in line:
                current_section = 'precedence'
                
            elif '<end>' in line:
                break
                
            else:
                # --- Process Data (ใส่ Try-Except เพื่อกัน Error รายบรรทัด) ---
                try:
                    if current_section == 'alternatives':
                        # รองรับ Format ที่อาจเพี้ยน เช่น ไม่มี : หรือ ;
                        # เปลี่ยนตัวคั่นทั้งหมดเป็นช่องว่าง แล้วแยกคำ
                        clean_line = line.replace(':', ' ').replace(';', ' ')
                        parts = clean_line.split()
                        
                        # ต้องมีอย่างน้อย 3 ส่วน (TaskID, Time1, Cost1)
                        if len(parts) >= 3 and parts[0].isdigit():
                            t_id = int(parts[0])
                            # ข้อมูลที่เหลือคือคู่ Time, Cost
                            modes_data = parts[1:]
                            
                            # วนลูปทีละคู่อย่างปลอดภัย
                            for m_str in modes_data:
                                if ',' in m_str:
                                    val = m_str.split(',')
                                    if len(val) == 2:
                                        t_val = int(val[0])
                                        c_val = int(val[1])
                                        data['task_modes'][t_id].append({'time': t_val, 'cost': c_val})

                    elif current_section == 'zoning':
                        parts = line.replace(',', ' ').split()
                        if len(parts) >= 2 and parts[0].isdigit():
                            t1, t2 = int(parts[0]), int(parts[1])
                            data['zoning'].append((t1, t2, 'NZ'))

                    elif current_section == 'precedence':
                        parts = line.replace(',', ' ').split()
                        # กรองเฉพาะที่เป็นตัวเลขจริงๆ
                        valid_nums = [int(x) for x in parts if x.isdigit()]
                        if len(valid_nums) >= 2:
                            i, j = valid_nums[0], valid_nums[1]
                            data['precedence'].append((i, j))
                            data['preds'][j].append(i)
                            data['succs'][i].append(j)
                            
                except ValueError:
                    continue # ถ้าบรรทัดไหนแปลงค่าไม่ได้ ให้ข้ามไปเลย (ไม่ Crash)
                    
    except StopIteration:
        pass
        
    return data

# ==========================================
# 2. CPLEX MILP SOLVER (Using docplex)
# ==========================================
def Solve_GALBP_CPLEX(data, time_limit=3600):
    num_tasks = data['num_tasks']
    C = data['cycle_time']
    tasks = range(1, num_tasks + 1)
    
    # Upper Bound ของสถานี (Worst case = จำนวนงาน)
    max_stations = num_tasks 
    stations = range(1, max_stations + 1)
    
    mdl = Model(name='GALBP_Exact')
    
    # -- Variables --
    x = {} # x[i,k,m]
    for i in tasks:
        for k in stations:
            for m_idx, mode in enumerate(data['task_modes'][i]):
                x[i, k, m_idx] = mdl.binary_var(name=f'x_{i}_{k}_{m_idx}')
    
    y = {k: mdl.binary_var(name=f'y_{k}') for k in stations} # y[k] สถานีถูกเปิดใช้
    
    # -- Constraints --
    # 1. Assignment (ทำ 1 ครั้ง)
    for i in tasks:
        mdl.add_constraint(
            mdl.sum(x[i, k, m_idx] for k in stations for m_idx, _ in enumerate(data['task_modes'][i])) == 1
        )
        
    # 2. Cycle Time
    for k in stations:
        station_load = mdl.sum(
            data['task_modes'][i][m_idx]['time'] * x[i, k, m_idx]
            for i in tasks for m_idx, _ in enumerate(data['task_modes'][i])
        )
        mdl.add_constraint(station_load <= C * y[k])
        
    # 3. Precedence
    for (i, j) in data['precedence']:
        st_i = mdl.sum(k * x[i, k, m_idx] for k in stations for m_idx, _ in enumerate(data['task_modes'][i]))
        st_j = mdl.sum(k * x[j, k, m_idx] for k in stations for m_idx, _ in enumerate(data['task_modes'][j]))
        mdl.add_constraint(st_i <= st_j)
        
    # 4. Zoning
    for (i, j, z_type) in data['zoning']:
        if z_type == 'NZ':
            for k in stations:
                sum_i = mdl.sum(x[i, k, m_idx] for m_idx, _ in enumerate(data['task_modes'][i]))
                sum_j = mdl.sum(x[j, k, m_idx] for m_idx, _ in enumerate(data['task_modes'][j]))
                mdl.add_constraint(sum_i + sum_j <= 1)

    # 5. Symmetry Breaking (เปิดสถานีเรียงกัน)
    for k in range(1, max_stations):
        mdl.add_constraint(y[k] >= y[k+1])

    # -- Optimization Phase 1: Minimize Z1 (Stations) --
    mdl.minimize(mdl.sum(y[k] for k in stations))
    mdl.set_time_limit(time_limit)
    
    sol1 = mdl.solve() # log_output=True ถ้าอยากเห็น log ใน Terminal
    
    if not sol1:
        return 999, 999999, "Infeasible"
        
    optimal_z1 = int(sol1.get_objective_value())
    
    # -- Optimization Phase 2: Minimize Z2 (Cost) --
    
    # 1. ล็อค Z1 (ใช้ค่า Optimal จาก Phase 1)
    # หมายเหตุ: ต้องมั่นใจว่า optimal_z1 มีค่าตัวเลขที่ถูกต้องจาก Phase 1
    mdl.add_constraint(mdl.sum(y[k] for k in stations) <= optimal_z1)
    
    # 2. สร้างสมการ Total Cost
    # (Editor อาจจะขึ้นเหลืองตรงนี้ ไม่ต้องสนใจครับ รันได้แน่นอน)
    total_cost_expr = mdl.sum(
        data['task_modes'][i][m_idx]['cost'] * x[i, k, m_idx]
        for i in tasks for k in stations for m_idx, _ in enumerate(data['task_modes'][i])
    )
    
    # 3. สั่ง Minimize
    mdl.minimize(total_cost_expr)
    
    # 4. สั่ง Solve รอบที่ 2
    sol2 = mdl.solve()
    
    # 5. ตรวจสอบผลลัพธ์และสถานะ (สำคัญมากสำหรับกลุ่ม C)
    if sol2:
        # ดึงค่า Cost ออกมา
        optimal_z2 = int(sol2.get_objective_value())
        
        # ตรวจสอบสถานะจริง (Status Check)
        # วิธีนี้ปลอดภัยสุด ไม่ต้อง import JobSolveStatus เพิ่ม
        solve_details = mdl.get_solve_details()
        status_code = solve_details.status_code
        status_str = "Optimal" # ค่า default
        
        # ถ้าสถานะไม่ใช่ Optimal (เช่น เวลาหมด หรือ Gap ยังเหลือ)
        # โค้ด 101, 102 มักจะคือ Optimal ใน CPLEX, ส่วน Time Limit จะเป็นเลขอื่น
        if "OPTIMAL" not in str(solve_details.status).upper():
             status_str = f"Feasible ({solve_details.status})"
             
        # (Optional) เก็บค่า Gap เพื่อดูว่าห่างจากค่าดีสุดแค่ไหน
        try:
            gap = solve_details.mip_relative_gap
            if gap > 0.0001: # ถ้า Gap > 0.01%
                status_str += f" [Gap {gap:.2%}]"
        except:
            pass

        return optimal_z1, optimal_z2, status_str
        
    else:
        # กรณีหาคำตอบไม่ได้เลยใน Phase 2 (แปลกมากถ้า Phase 1 ผ่าน)
        return optimal_z1, 999999, "Infeasible_Phase2"
    
# ==========================================
# 3. BATCH RUNNER
# ==========================================
def Batch_Run_CPLEX(input_folder, output_csv, time_limit=3600):
    results = []
    # ค้นหาไฟล์ .alb ในโฟลเดอร์
    files = glob.glob(os.path.join(input_folder, '*.alb')) 
    files = sorted(files)
    
    print(f"🏭 CPLEX Baseline Running...")
    print(f"📂 Folder: {input_folder}")
    print(f"📄 Found: {len(files)} files")
    print("-" * 50)
    
    for idx, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(files)}] Solving: {filename} ...", end=" ")
        
        try:
            start_time = time.time()
            data = parse_new_benchmark_format(filepath)
            
            # เรียก CPLEX Solve
            z1, z2, status = Solve_GALBP_CPLEX(data, time_limit)
            
            duration = time.time() - start_time
            
            results.append({
                'Instance': filename,
                'Tasks': data['num_tasks'],
                'Optimal_Z1': z1,
                'Optimal_Z2': z2,
                'Time_Sec': round(duration, 2),
                'Status': status
            })
            print(f"Done! (Z1={z1}, Z2={z2}, Time={duration:.2f}s)")
            
        except Exception as e:
            print(f"Error: {e}")
            
    # Save CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Saved results to: {output_csv}")

# ==========================================
# 4. EXECUTION ZONE (แก้ไข Path ให้ชัวร์)
# ==========================================
if __name__ == "__main__":
    
    # กำหนด Path ตรงๆ ไปที่ C:\GALBP เลย จะได้ไม่มีปัญหาเรื่องหาไฟล์ไม่เจอ
    BASE_DIR = r'C:\GALBP'  # ใช้ r นำหน้าเพื่อบอกว่าเป็น Raw string (กัน path error บน Windows)
    
    target_group = 'GALBP_A'  # <--- เปลี่ยนเป็น GALBP_B หรือ C ได้ที่นี่
    
    # สร้าง Path เต็มไปยังโฟลเดอร์ข้อมูล
    input_path = os.path.join(BASE_DIR, target_group)
    
    # ไฟล์ผลลัพธ์จะถูกเซฟไว้ที่ C:\GALBP เช่นกัน
    output_csv = os.path.join(BASE_DIR, f'Baseline_Results_{target_group}.csv')
    
    # Debug: ปริ้นท์เช็ค Path ก่อนรัน
    print(f"Target Input Path: {input_path}")
    
    if os.path.exists(input_path):
        # รัน CPLEX (ตั้งเวลา 3600 วินาที = 1 ชม. สำหรับ Baseline มาตรฐาน)
        Batch_Run_CPLEX(input_path, output_csv, time_limit=3600) 
    else:
        print(f"❌ Error: ไม่พบโฟลเดอร์ที่ระบุ")
        print(f"โปรดตรวจสอบว่ามีโฟลเดอร์ชื่อ {target_group} อยู่ใน {BASE_DIR} หรือไม่")