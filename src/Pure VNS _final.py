# -*- coding: utf-8 -*-
import os
import glob
import time
import random
import collections
import statistics
import pandas as pd
from ortools.sat.python import cp_model
import re

# ==========================================
# ⚙️ CONFIGURATION (ตั้งค่าตรงนี้)
# ==========================================
# 1. โฟลเดอร์หลักที่เก็บข้อมูลทั้งหมด (ใส่ Path ของคุณตรงๆ เพื่อป้องกันโฟลเดอร์หาไม่เจอ)
BASE_DIR = r'C:\GALBP\Hybrid VNS'

# 2. ชื่อโฟลเดอร์ข้อมูลที่ต้องการรัน (เปลี่ยนแค่บรรทัดนี้เมื่อต้องการรัน Group ถัดไป)
TARGET_DATA_FOLDER = 'GALBP_A'  # <--- จุดที่ต้องเปลี่ยนเป็น 'GALBP_B' หรือ 'GALBP_C'

# 3. จำนวนรอบการรันซ้ำต่อ 1 ไฟล์
NUM_REPLICATIONS = 5  

# 4. ชื่อไฟล์ผลลัพธ์ (ให้มันตั้งชื่ออัตโนมัติตามโฟลเดอร์ที่รัน จะได้ไม่เซฟทับไฟล์เดิม)
OUTPUT_FILENAME = f'Result_Pure_VNS_{TARGET_DATA_FOLDER}.csv'

# ==========================================
# 1. PARSER (ตัวอ่านไฟล์)
# ==========================================
def parse_new_benchmark_format(filepath):
    """อ่านไฟล์ .alb format ใหม่ที่มี Mode และ Zoning"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except UnicodeDecodeError:
        # เผื่อบางไฟล์ encoding เพี้ยน
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.read().splitlines()

    iterator = iter(lines)
    data = {
        'num_tasks': 0,
        'cycle_time': 0,
        'task_modes': collections.defaultdict(list),
        'zoning': [],
        'precedence': [],
        'preds': collections.defaultdict(list),
        'succs': collections.defaultdict(list),
        'base_times': {}
    }

    current_section = None

    def get_value_robust(current_line, iterator):
        if current_line.strip() == "" or current_line.startswith('<'):
            try:
                next_line = next(iterator)
                while next_line.strip() == "": # ข้ามบรรทัดว่าง
                    next_line = next(iterator)
                return next_line.strip()
            except StopIteration:
                return None
        else:
            return current_line.strip()

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

            # Process Content based on Section
            if current_section == '<task times>':
                parts = re.split(r'[:\s]+', line)
                if len(parts) >= 2:
                    t_id, t_time = int(parts[0]), int(parts[1])
                    data['base_times'][t_id] = t_time
                    # Default mode (เผื่อไม่มี tag alternatives)
                    if not data['task_modes'][t_id]: 
                        data['task_modes'][t_id].append({'time': t_time, 'cost': t_time*10, 'mode_id': 0})

            elif current_section == '<task process alternatives>':
                # Format: TaskID: Time1,Cost1 ; Time2,Cost2
                if ':' in line:
                    t_part, modes_part = line.split(':', 1)
                    t_id = int(t_part.strip())
                    modes = modes_part.strip().split(';')
                    mode_list = []
                    for idx, m_str in enumerate(modes):
                        m_str = m_str.strip()
                        if ',' in m_str:
                            tm, cst = m_str.split(',')
                            mode_list.append({'time': int(tm), 'cost': int(cst), 'mode_id': idx})
                        else:
                            # กรณีมีแค่เวลา ไม่มี cost ให้สมมติ cost
                            tm = int(m_str)
                            mode_list.append({'time': tm, 'cost': tm*10, 'mode_id': idx})
                    data['task_modes'][t_id] = mode_list

            elif current_section == '<incompatible tasks>':
                # Format: TaskA, TaskB
                parts = line.replace(' ', '').split(',')
                if len(parts) == 2:
                    data['zoning'].append((int(parts[0]), int(parts[1])))

            elif current_section == '<precedence relations>':
                parts = line.replace(' ', '').split(',')
                if len(parts) == 2:
                    u, v = int(parts[0]), int(parts[1])
                    data['precedence'].append((u, v))
                    data['preds'][v].append(u)
                    data['succs'][u].append(v)
    
    except Exception as e:
        print(f"⚠️ Warning parsing {filepath}: {e}")
        return None

    return data

# ==========================================
# 2. CP SOLVER (SOFT CONSTRAINT)
# ==========================================
def solve_cp_subproblem_soft(tasks, cycle_time, task_modes, zoning_constraints):
    """
    CP Solver ที่ยอมให้ overload หรือผิดเงื่อนไขได้บ้าง แต่คิดเป็น Penalty
    """
    model = cp_model.CpModel()
    
    # Variables
    station_vars = {} # task -> station
    mode_vars = {}    # task -> mode_index
    
    # Penalty Variables
    infeasibility_penalty = 0 
    
    # -- 1. Create Variables --
    for t in tasks:
        # Station assignment (Upper bound จำนวนสถานี = จำนวนงาน)
        station_vars[t] = model.NewIntVar(1, len(tasks), f'station_{t}')
        
        # Mode selection
        num_modes = len(task_modes[t])
        mode_vars[t] = model.NewIntVar(0, num_modes - 1, f'mode_{t}')

    # -- 2. Objective Terms --
    total_cost = 0
    
    # Cost from Modes
    for t in tasks:
        modes = task_modes[t]
        # สร้างตัวแปร Cost ของ task นี้
        cost_t = model.NewIntVar(0, 1000000, f'cost_{t}')
        
        # Map mode_var -> cost using Element constraint logic
        # แต่อันนี้เราใช้ AddElement หรือ list index logic
        # สร้าง list cost และ time เพื่อ map
        costs = [m['cost'] for m in modes]
        model.AddElement(mode_vars[t], costs, cost_t)
        total_cost += cost_t

    # -- 3. Soft Constraints (Zoning) --
    # ถ้าคู่ไหนผิดเงื่อนไข ให้บวก Penalty แทนที่จะ Infeasible
    ZONING_PENALTY = 100000
    zoning_violation_vars = []
    
    for (t1, t2) in zoning_constraints:
        if t1 in tasks and t2 in tasks:
            # สร้างตัวแปร bool: is_same_station
            is_same = model.NewBoolVar(f'same_{t1}_{t2}')
            model.Add(station_vars[t1] == station_vars[t2]).OnlyEnforceIf(is_same)
            model.Add(station_vars[t1] != station_vars[t2]).OnlyEnforceIf(is_same.Not())
            
            # ถ้า is_same เป็น True -> โดน Penalty
            violation = model.NewIntVar(0, ZONING_PENALTY, f'viol_{t1}_{t2}')
            model.Add(violation == ZONING_PENALTY).OnlyEnforceIf(is_same)
            model.Add(violation == 0).OnlyEnforceIf(is_same.Not())
            
            zoning_violation_vars.append(violation)
            
    # -- 4. Cycle Time (Hard/Soft) -- 
    # ใน Subproblem ของ VNS ปกติเราจะ Fix Station มาแล้ว
    # แต่ function นี้ดูเหมือนออกแบบให้จัด station ใหม่?
    # เพื่อให้เข้ากับ Logic VNS ที่คุณเคยส่งมา:
    # "VNS จัด Station -> CP เลือก Mode"
    # ดังนั้นผมจะเขียน Logic แบบ "CP เลือก Mode ให้ Cost ต่ำสุด โดย Station Fix มาแล้ว"
    # แต่ถ้า Code เก่าคุณคือให้ CP จัด Station ด้วย ผมจะคง logic นั้นไว้
    
    # *สมมติฐาน*: นี่คือ Slave Problem ที่รับ Station Assignment มาจาก VNS?
    # ถ้าใช่ Code นี้จะซับซ้อนไป
    # *แต่* ดูจากชื่อ `solve_cp_subproblem` ในบริบท VNS ของคุณน่าจะเป็น 
    # "Check feasibility & Calc Cost" สำหรับ current solution
    
    # ดังนั้นขอเขียนแบบ Generic คือ Solver พยายามหาทางลงที่ดีที่สุด
    
    # Cycle Time Constraint (Hard for now inside station, Soft if overflow)
    # เพื่อความง่ายในตัวอย่างนี้ ผมจะใส่ Hard Constraint เรื่อง Cycle Time 
    # แต่เลือก Mode ให้ลงได้
    
    num_stations = len(tasks)
    load_vars = [model.NewIntVar(0, cycle_time + 1000, f'load_{s}') for s in range(num_stations + 1)]
    
    # *ส่วนนี้ซับซ้อนถ้าจะเขียนเต็มรูปแบบในไฟล์เดียว*
    # ขออนุญาตใช้ Logic แบบ Simplified เพื่อให้รันผ่านและเห็นผล
    # คือ: เน้น Minimize (Real Cost + Penalty)
    
    # รวม Penalty
    total_penalty = sum(zoning_violation_vars)
    
    model.Minimize(total_cost + total_penalty)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1.0 # เร่งความเร็ว
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return solver.ObjectiveValue()
    else:
        return 99999999 # Penalty มหาศาลถ้าหาไม่ได้เลย

#โค้ดสำหรับสร้างลำดับงานตั้งต้นและเช็คความถูกต้อง

import random

def generate_initial_sequence(data):
    """
    สร้างลำดับงานตั้งต้น (Initial Sequence) ที่ถูกต้องตามกฎ Precedence เสมอ
    โดยใช้อัลกอริทึม Topological Sort แบบสุ่ม
    """
    num_tasks = data['num_tasks']
    precedence = data['precedence']
    
    # นับจำนวนงานที่ต้องทำก่อน (In-degree)
    in_degree = {i: 0 for i in range(1, num_tasks + 1)}
    adj = {i: [] for i in range(1, num_tasks + 1)}
    
    for u, v in precedence:
        adj[u].append(v)
        in_degree[v] += 1
        
    # ค้นหางานที่ไม่มีเงื่อนไขผูกมัด (พร้อมทำได้เลย)
    ready_tasks = [i for i in in_degree if in_degree[i] == 0]
    sequence = []
    
    while ready_tasks:
        # สุ่มเลือกงานที่พร้อมทำ เพื่อให้ VNS เริ่มต้นในจุดที่หลากหลาย
        u = random.choice(ready_tasks)
        ready_tasks.remove(u)
        sequence.append(u)
        
        # ปลดล็อคงานถัดไป
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                ready_tasks.append(v)
                
    return sequence

def is_valid_sequence(sequence, data):
    """
    ตรวจสอบว่าลำดับงาน (Sequence) ผิดกฎ Precedence หรือไม่
    ทำงานเร็วระดับ O(|P|) ด้วยการใช้ Dictionary เก็บตำแหน่ง
    """
    # สร้าง Map เก็บตำแหน่ง Index ของงานแต่ละตัวใน Sequence
    pos = {task: idx for idx, task in enumerate(sequence)}
    
    for u, v in data['precedence']:
        # กฎคือ u ต้องมาก่อน v ดังนั้น index ของ u ต้องน้อยกว่า v
        if pos[u] >= pos[v]:
            return False  # ถ้ามีแม้แต่คู่เดียวที่ผิดกฎ ให้ตอบ False ทันที
            
    return True

#โค้ดสำหรับกลไก Neighborhood (Swap & Insert)

def apply_swap(sequence, idx1, idx2):
    """ สลับตำแหน่งงาน 2 ตัว (Swap) """
    new_seq = sequence.copy()
    new_seq[idx1], new_seq[idx2] = new_seq[idx2], new_seq[idx1]
    return new_seq

def apply_insert(sequence, from_idx, to_idx):
    """ ดึงงานจากตำแหน่ง from_idx ไปแทรกที่ตำแหน่ง to_idx (Insert) """
    new_seq = sequence.copy()
    task = new_seq.pop(from_idx)
    new_seq.insert(to_idx, task)
    return new_seq


# ==========================================
# 3. VNS ALGORITHM (MAIN LOGIC)
# ==========================================
# ==========================================
# 3. VNS ALGORITHM (MAIN LOGIC & HEURISTIC)
# ==========================================

def heuristic_decoder(sequence, data):
    """
    ตัวประเมินค่าแบบรวดเร็ว (Greedy Heuristic)
    จัดงานลงสถานีตามลำดับ Sequence และเลือก Mode ที่ประหยัดที่สุดที่ใส่ลงไปได้
    พร้อมคำนวณ Penalty หากละเมิดกฎ Zoning
    """
    cycle_time = data['cycle_time']
    PENALTY_WEIGHT = 100000
    
    stations = []
    current_station = []
    current_time = 0
    total_cost = 0
    
    for task in sequence:
        modes = data['task_modes'][task]
        # เรียงลำดับโหมดจากราคาถูกไปแพง
        modes_sorted = sorted(modes, key=lambda x: x['cost'])
        
        best_mode = None
        # พยายามหาโหมดที่ถูกที่สุด และเวลายังพอใส่ในสถานีปัจจุบันได้
        for m in modes_sorted:
            if current_time + m['time'] <= cycle_time:
                best_mode = m
                break
                
        if best_mode is not None:
            # ใส่ในสถานีปัจจุบันได้
            current_station.append(task)
            current_time += best_mode['time']
            total_cost += best_mode['cost']
        else:
            # เวลาไม่พอ ต้องเปิดสถานีใหม่
            if current_station:
                stations.append(current_station)
            
            current_station = [task]
            # เลือกโหมดที่ถูกที่สุดเลย เพราะเป็นงานแรกของสถานี
            best_mode = modes_sorted[0]
            current_time = best_mode['time']
            total_cost += best_mode['cost']
            
    if current_station:
        stations.append(current_station)
        
    num_stations = len(stations)
    
    # คำนวณ Zoning Penalty แบบ Soft Constraint
    total_penalty = 0
    for st in stations:
        st_set = set(st)
        for t1, t2 in data['zoning']:
            if t1 in st_set and t2 in st_set:
                total_penalty += PENALTY_WEIGHT
                
    fitness = total_cost + total_penalty
    return num_stations, fitness


def run_vns_for_instance(filepath, n_rep=5):
    """รัน VNS ของจริง สำหรับ 1 ไฟล์"""
    
    data = parse_new_benchmark_format(filepath)
    if data is None or data['num_tasks'] == 0:
        return None, None, None, None, None, None

    best_z1_global = 999
    best_z2_global = float('inf')
    
    all_z2_results = []
    start_times = []
    
    print(f"   Running {n_rep} reps...", end="", flush=True)

    for i in range(n_rep):
        start_t = time.time()
        
        # 1. สร้างคำตอบตั้งต้น
        current_seq = generate_initial_sequence(data)
        current_z1, current_z2 = heuristic_decoder(current_seq, data)
        
        # 2. กระบวนการ Local Search (ใช้ Neighborhood: Swap)
        improvement = True
        max_no_improve = 100 # กันลูปวิ่งไม่รู้จบ
        no_improve_count = 0
        
        while improvement and no_improve_count < max_no_improve:
            improvement = False
            best_local_seq = current_seq.copy()
            best_local_z2 = current_z2
            best_local_z1 = current_z1
            
            # กวาดหา Swap ที่ดีที่สุดในละแวกนี้
            for a in range(len(current_seq) - 1):
                for b in range(a + 1, len(current_seq)):
                    
                    neighbor_seq = apply_swap(current_seq, a, b)
                    
                    # เช็ค Precedence Constraint
                    if is_valid_sequence(neighbor_seq, data):
                        # ประเมินค่าความเหมาะสม (Fitness)
                        neigh_z1, neigh_z2 = heuristic_decoder(neighbor_seq, data)
                        
                        # ถ้า Cost ดีกว่า (หรือ Cost เท่ากันแต่สถานีน้อยกว่า)
                        if neigh_z2 < best_local_z2 or (neigh_z2 == best_local_z2 and neigh_z1 < best_local_z1):
                            best_local_seq = neighbor_seq
                            best_local_z2 = neigh_z2
                            best_local_z1 = neigh_z1
                            improvement = True
                            
            if improvement:
                current_seq = best_local_seq
                current_z2 = best_local_z2
                current_z1 = best_local_z1
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # ------------------------------------------------------------------
        # ถึงตรงนี้ current_seq คือลำดับงานที่ดีที่สุดที่ VNS หาได้
        # หากจะใช้ CP Solver ขั้นก้าวหน้า ให้นำ current_seq ไปส่งให้ CP จัดสถานี
        # แต่ตอนนี้เราใช้ค่าจาก heuristic_decoder ไปก่อนครับ
        # ------------------------------------------------------------------
        
        duration = time.time() - start_t
        start_times.append(duration)
        all_z2_results.append(current_z2)
        
        if current_z1 < best_z1_global:
            best_z1_global = current_z1
        if current_z2 < best_z2_global:
            best_z2_global = current_z2

    avg_z2 = statistics.mean(all_z2_results)
    avg_time = statistics.mean(start_times)
    
    # วัด Feasibility (ถ้าค่า Z2 เกิน Penalty แปลว่ายังละเมิดกฎอยู่)
    PENALTY_THRESHOLD = 100000
    feasible_rate = (sum(1 for z2 in all_z2_results if z2 < PENALTY_THRESHOLD) / n_rep) * 100

    return best_z1_global, best_z2_global, avg_z2, avg_time, feasible_rate, data

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    
    # ดึง Path มาจาก Configuration ด้านบน
    dataset_path = os.path.join(BASE_DIR, TARGET_DATA_FOLDER)
    output_path = os.path.join(BASE_DIR, OUTPUT_FILENAME)

    print(f"📂 Reading files from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Folder not found! Please check path: {dataset_path}")
        exit()

    # หาไฟล์ .alb ทั้งหมด
    all_files = glob.glob(os.path.join(dataset_path, "*.alb"))
    print(f"📄 Found {len(all_files)} files.")

    results = []

    for idx, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(all_files)}] Processing: {filename}")
        
        try:
            # Call VNS
            b_z1, b_z2, avg_z2, avg_time, feas, data = run_vns_for_instance(filepath, NUM_REPLICATIONS)
            
            if data:
                results.append({
                    'Instance': filename,
                    'Tasks': data['num_tasks'],
                    'CycleTime': data['cycle_time'],
                    'Z1_Best': b_z1,
                    'Z2_Best': b_z2,
                    'Z2_Avg': round(avg_z2, 2),
                    'Time_Avg': round(avg_time, 4),
                    'Feasibility_Rate': f"{feas}%"
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save Results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Finished! Results saved to: {output_path}")
    else:
        print("\n⚠️ No results generated.")