# -*- coding: utf-8 -*-
import os
import glob
import time
import collections
import pandas as pd
from docplex.mp.model import Model

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = r'C:\GALBP\Hybrid VNS'
TARGET_FOLDER = 'GALBP_C'  # ลองรัน Group A ก่อน
OUTPUT_FILENAME = f'Result_Exact_Docplex_{TARGET_FOLDER}.csv'
TIME_LIMIT = 600  # 1 ชั่วโมง (วินาที)

# ==========================================
# 1. PARSER (ตัวอ่านไฟล์คงเดิม)
# ==========================================
def parse_new_benchmark_format(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: lines = f.read().splitlines()
    except:
        with open(filepath, 'r', encoding='latin-1') as f: lines = f.read().splitlines()
    iterator = iter(lines)
    data = {'num_tasks': 0, 'cycle_time': 0, 'task_modes': collections.defaultdict(list), 'zoning': [], 'precedence': []}
    current_section = None

    def get_value_robust(current_line, iterator):
        if current_line.strip() == "" or current_line.startswith('<'):
            try:
                next_line = next(iterator)
                while next_line.strip() == "": next_line = next(iterator)
                return next_line.strip()
            except StopIteration: return None
        else: return current_line.strip()

    try:
        for line in iterator:
            line = line.strip()
            if not line: continue
            if line.startswith('<'):
                current_section = line
                if line == '<number of tasks>':
                    val = get_value_robust(line, iterator)
                    if val: data['num_tasks'] = int(val)
                elif line == '<cycle time>':
                    val = get_value_robust(line, iterator)
                    if val: data['cycle_time'] = int(val)
                continue
            if current_section == '<task process alternatives>':
                if ':' in line:
                    t_part, modes_part = line.split(':', 1)
                    t_id = int(t_part.strip())
                    modes = modes_part.strip().split(';')
                    mode_list = []
                    for idx, m_str in enumerate(modes):
                        m_str = m_str.strip()
                        if ',' in m_str: mode_list.append({'time': int(m_str.split(',')[0]), 'cost': int(m_str.split(',')[1]), 'mode_id': idx})
                        else: mode_list.append({'time': int(m_str), 'cost': int(m_str)*10, 'mode_id': idx})
                    data['task_modes'][t_id] = mode_list
            elif current_section == '<incompatible tasks>':
                parts = line.replace(' ', '').split(',')
                if len(parts) == 2: data['zoning'].append((int(parts[0]), int(parts[1])))
            elif current_section == '<precedence relations>':
                parts = line.replace(' ', '').split(',')
                if len(parts) == 2: data['precedence'].append((int(parts[0]), int(parts[1])))
    except Exception as e: return None
    return data

# ==========================================
# 2. EXACT MILP SOLVER (NATIVE DOCPLEX)
# ==========================================
def solve_exact_docplex(filepath):
    data = parse_new_benchmark_format(filepath)
    if data is None or data['num_tasks'] == 0: return None, 0, "Error"
    
    start_time = time.time()
    
    I = list(range(1, data['num_tasks'] + 1))
    K = list(range(1, data['num_tasks'] + 1)) 
    P = data['precedence']
    # กรองข้อมูล Zoning ซ้ำซ้อน และจัดเรียง (น้อย, มาก) เสมอ เพื่อไม่ให้เกิด (45,96) และ (96,45) ซ้ำกัน
    Z = list(set([(min(i, j), max(i, j)) for i, j in data['zoning']]))
    C = data['cycle_time']
    # 1. สร้างโมเดล Docplex
    mdl = Model(name='GALBP_Soft_Constraint')
    
    # 2. ประกาศตัวแปรตัดสินใจ (Decision Variables)
    x = {}
    for i in I:
        for k in K:
            for m in data['task_modes'][i]:
                x[i, k, m['mode_id']] = mdl.binary_var(name=f'x_{i}_{k}_{m["mode_id"]}')
                
    y = {k: mdl.binary_var(name=f'y_{k}') for k in K}
    v = {(i, j, k): mdl.binary_var(name=f'v_{i}_{j}_{k}') for (i, j) in Z for k in K}
    
    # 3. สมการเป้าหมาย (Objective Function)
    total_stations = mdl.sum(y[k] * 10000000 for k in K)
    total_cost = mdl.sum(m['cost'] * x[i, k, m['mode_id']] for i in I for k in K for m in data['task_modes'][i])
    total_penalty = mdl.sum(100000 * v[i, j, k] for (i, j) in Z for k in K)
    
    mdl.minimize(total_stations + total_cost + total_penalty)
    
    # 4. ข้อจำกัด (Constraints)
    # 4.1) Occurrence
    for i in I:
        mdl.add_constraint(mdl.sum(x[i, k, m['mode_id']] for k in K for m in data['task_modes'][i]) == 1)
        
    # 4.2) Cycle Time
    for k in K:
        mdl.add_constraint(mdl.sum(m['time'] * x[i, k, m['mode_id']] for i in I for m in data['task_modes'][i]) <= C * y[k])
        
    # 4.3) Precedence
    for (i, j) in P:
        mdl.add_constraint(
            mdl.sum(k * x[i, k, m['mode_id']] for k in K for m in data['task_modes'][i]) <= \
            mdl.sum(k * x[j, k, m['mode_id']] for k in K for m in data['task_modes'][j])
        )
        
    # 4.4) Soft Zoning Constraint
    for (i, j) in Z:
        for k in K:
            mdl.add_constraint(
                mdl.sum(x[i, k, m['mode_id']] for m in data['task_modes'][i]) + \
                mdl.sum(x[j, k, m['mode_id']] for m in data['task_modes'][j]) - v[i, j, k] <= 1
            )
            
    # 4.5) Station Ordering
    for k in range(1, len(K)):
        mdl.add_constraint(y[k] >= y[k+1])
        
    # ==================================================
    # 🌟 ตั้งค่าและรัน CPLEX
    # ==================================================
    mdl.parameters.timelimit = TIME_LIMIT # ตั้งเวลา Time Limit
    mdl.parameters.threads = 0            # ให้ CPLEX ดึง CPU มาใช้เต็มสูบ (ทุกคอร์)
    
    solution = mdl.solve(log_output=False) # เปลี่ยนเป็น True ถ้าอยากดู Log การตัดกิ่งของ CPLEX
    
    duration = time.time() - start_time
    
    if solution:
        z1 = sum(round(solution.get_value(y[k])) for k in K)
        cost = sum(m['cost'] * round(solution.get_value(x[i, k, m['mode_id']])) for i in I for k in K for m in data['task_modes'][i])
        penalty = sum(100000 * round(solution.get_value(v[i, j, k])) for (i, j) in Z for k in K)
        z2 = cost + penalty
        
        # ตรวจสอบว่าคำตอบนี้เป็น Optimal หรือเปล่า
        status = "Optimal" if mdl.solve_details.status == "integer optimal solution" else "Feasible (Timeout)"
        return {"Z1_Pareto": int(z1), "Z2_Pareto": int(z2)}, duration, status
    else:
        return None, duration, "Infeasible / Timeout"

# ==========================================
# 3. RUNNER
# ==========================================
if __name__ == "__main__":
    dataset_path = os.path.join(BASE_DIR, TARGET_FOLDER)
    all_files = glob.glob(os.path.join(dataset_path, "*.alb"))
    print(f"🚀 เริ่มรัน Native CPLEX (Docplex) สำหรับ {len(all_files)} ไฟล์ (Time Limit: {TIME_LIMIT}s/file)")
    
    results = []
    for idx, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(all_files)}] กำลังรัน {filename}...", end=" ", flush=True)
        
        try:
            ans, duration, status = solve_exact_docplex(filepath)
            if ans:
                print(f"-> {status} (Z1: {ans['Z1_Pareto']}, Z2: {ans['Z2_Pareto']}, Time: {duration:.2f}s)")
                results.append({
                    'Instance': filename,
                    'Algorithm': 'CPLEX',
                    'Z1_Pareto': ans['Z1_Pareto'],
                    'Z2_Pareto': ans['Z2_Pareto'],
                    'Status': status,
                    'Time_Sec': round(duration, 4)
                })
            else:
                print(f"-> {status}")
        except Exception as e:
            print(f"-> Error: {e}")
            
    if results:
        pd.DataFrame(results).to_csv(os.path.join(BASE_DIR, OUTPUT_FILENAME), index=False)
        print("\n✅ เสร็จสิ้น! บันทึกผลลัพธ์แล้ว")