# -*- coding: utf-8 -*-
import os
import glob
import time
import random
import collections
import pandas as pd
import re

# ==========================================
# ⚙️ CONFIGURATION (ตั้งค่าตรงนี้)
# ==========================================
BASE_DIR = r'C:\GALBP\Hybrid VNS'
TARGET_DATA_FOLDER = 'GALBP_C'  # ทดสอบกับ Group A ก่อน

# สำหรับ MO-VNS เราไม่จำเป็นต้องรันหลาย Replication เพื่อหาค่าเฉลี่ย
# แต่เราจะให้มันรันยาวๆ 1 รอบเพื่อขยายคลังคำตอบ (Archive) ให้ใหญ่ที่สุด
MAX_NO_IMPROVE = 50  # จำนวนรอบสูงสุดที่คลังคำตอบไม่ถูกอัปเดตแล้วให้หยุดรัน

# MO-VNS Parameters (เตรียมไว้จูน)
MAX_NO_IMPROVE = 50        # Factor A: จำนวนรอบสูงสุดที่คลังไม่พัฒนาแล้วหยุด
INIT_ARCHIVE_SIZE = 10     # Factor B: จำนวนคำตอบสุ่มตั้งต้นในคลัง
SHAKING_STRENGTH = 3       # Factor C: ความแรงในการเขย่า (สุ่มสลับกี่ครั้ง)
LS_LIMIT = 50              # Factor D: ขอบเขตการค้นหาย่านใกล้เคียงสูงสุดต่อรอบ

OUTPUT_FILENAME = f'Result_MOVNS_Soft_{TARGET_DATA_FOLDER}.csv'

# ==========================================
# 1. PARSER (ตัวอ่านไฟล์ - ใช้ของเดิม 100%)
# ==========================================
def parse_new_benchmark_format(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.read().splitlines()

    iterator = iter(lines)
    data = {
        'num_tasks': 0, 'cycle_time': 0, 'task_modes': collections.defaultdict(list),
        'zoning': [], 'precedence': [], 'preds': collections.defaultdict(list),
        'succs': collections.defaultdict(list), 'base_times': {}
    }
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

            if current_section == '<task times>':
                parts = re.split(r'[:\s]+', line)
                if len(parts) >= 2:
                    t_id, t_time = int(parts[0]), int(parts[1])
                    data['base_times'][t_id] = t_time
                    if not data['task_modes'][t_id]: 
                        data['task_modes'][t_id].append({'time': t_time, 'cost': t_time*10, 'mode_id': 0})
            elif current_section == '<task process alternatives>':
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
                            tm = int(m_str)
                            mode_list.append({'time': tm, 'cost': tm*10, 'mode_id': idx})
                    data['task_modes'][t_id] = mode_list
            elif current_section == '<incompatible tasks>':
                parts = line.replace(' ', '').split(',')
                if len(parts) == 2: data['zoning'].append((int(parts[0]), int(parts[1])))
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
# 2. SEQUENCE VALIDATION & HEURISTIC DECODER
# ==========================================
def generate_initial_sequence(data):
    num_tasks = data['num_tasks']
    precedence = data['precedence']
    in_degree = {i: 0 for i in range(1, num_tasks + 1)}
    adj = {i: [] for i in range(1, num_tasks + 1)}
    for u, v in precedence:
        adj[u].append(v)
        in_degree[v] += 1
    ready_tasks = [i for i in in_degree if in_degree[i] == 0]
    sequence = []
    while ready_tasks:
        u = random.choice(ready_tasks)
        ready_tasks.remove(u)
        sequence.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0: ready_tasks.append(v)
    return sequence

def is_valid_sequence(sequence, data):
    pos = {task: idx for idx, task in enumerate(sequence)}
    for u, v in data['precedence']:
        if pos[u] >= pos[v]: return False
    return True

def apply_swap(sequence, idx1, idx2):
    new_seq = sequence.copy()
    new_seq[idx1], new_seq[idx2] = new_seq[idx2], new_seq[idx1]
    return new_seq

def heuristic_decoder(sequence, data):
    cycle_time = data['cycle_time']
    PENALTY_WEIGHT = 100000
    stations, current_station = [], []
    current_time, total_cost = 0, 0
    
    for task in sequence:
        modes_sorted = sorted(data['task_modes'][task], key=lambda x: x['cost'])
        best_mode = next((m for m in modes_sorted if current_time + m['time'] <= cycle_time), None)
                
        if best_mode is not None:
            current_station.append(task)
            current_time += best_mode['time']
            total_cost += best_mode['cost']
        else:
            if current_station: stations.append(current_station)
            current_station = [task]
            best_mode = modes_sorted[0]
            current_time = best_mode['time']
            total_cost += best_mode['cost']
            
    if current_station: stations.append(current_station)
    
    total_penalty = sum(PENALTY_WEIGHT for st in stations for t1, t2 in data['zoning'] if t1 in set(st) and t2 in set(st))
    return len(stations), total_cost + total_penalty

# ==========================================
# 3. MULTI-OBJECTIVE MECHANISMS (เพิ่มใหม่สำหรับ MO-VNS)
# ==========================================
def dominates(z1_a, z2_a, z1_b, z2_b):
    """เช็คว่าคำตอบ A ข่ม B หรือไม่ (A ต้องดีกว่าหรือเท่ากับ B ทุกด้าน และดีกว่าอย่างน้อย 1 ด้าน)"""
    return (z1_a <= z1_b and z2_a <= z2_b) and (z1_a < z1_b or z2_a < z2_b)

def update_archive(archive, new_seq, new_z1, new_z2):
    """
    อัปเดตคลังคำตอบ (Pareto Archive)
    ถ้าเอาตัวใหม่เข้าคลังได้ คืนค่า True, ถ้าโดนตัวเก่าข่ม คืนค่า False
    """
    # 1. เช็คว่าตัวใหม่โดนตัวเก่าในคลังข่มไหม
    for arch_seq, arch_z1, arch_z2 in archive:
        if dominates(arch_z1, arch_z2, new_z1, new_z2):
            return False 
        # เพื่อไม่ให้คลังมีคำตอบซ้ำซ้อนกันมากเกินไป (เอาแค่ objective ไม่ซ้ำก็พอ)
        if arch_z1 == new_z1 and arch_z2 == new_z2:
            return False 

    # 2. คัดกรองตัวเก่าที่โดนตัวใหม่ข่มทิ้งไป
    non_dominated_archive = [(s, z1, z2) for s, z1, z2 in archive if not dominates(new_z1, new_z2, z1, z2)]
    
    # 3. เอาตัวใหม่ใส่เข้าไป
    non_dominated_archive.append((new_seq, new_z1, new_z2))
    
    # อัปเดตคลังหลัก
    archive.clear()
    archive.extend(non_dominated_archive)
    return True

# ==========================================
# 4. MO-VNS ALGORITHM
# ==========================================
def run_movns_for_instance(filepath):
    data = parse_new_benchmark_format(filepath)
    if data is None or data['num_tasks'] == 0:
        return None, 0

    start_t = time.time()
    archive = []
    
    # 1. Initialization: สร้างคำตอบตั้งต้นตาม INIT_ARCHIVE_SIZE
    for _ in range(INIT_ARCHIVE_SIZE):
        init_seq = generate_initial_sequence(data)
        init_z1, init_z2 = heuristic_decoder(init_seq, data)
        update_archive(archive, init_seq, init_z1, init_z2)
        
    no_improve_count = 0
    
    # 2. Search Loop
    while no_improve_count < MAX_NO_IMPROVE:
        improvement_in_this_round = False
        
        # เลือกคำตอบตั้งต้น
        base_seq, _, _ = random.choice(archive)
        current_seq = base_seq.copy()
        
        # 3. Shaking Phase: เขย่าคำตอบเพื่อหนี Local Optima
        for _ in range(SHAKING_STRENGTH):
            for _ in range(10): # ลองเขย่า 10 ครั้ง หาอันที่ถูกกฎ
                idx1, idx2 = random.sample(range(len(current_seq)), 2)
                temp = apply_swap(current_seq, idx1, idx2)
                if is_valid_sequence(temp, data):
                    current_seq = temp
                    break

        # 4. Local Search Phase (สุ่มตรวจย่านใกล้เคียง ไม่เกิน LS_LIMIT)
        indices = [(a, b) for a in range(len(current_seq)-1) for b in range(a+1, len(current_seq))]
        random.shuffle(indices) # สับเปลี่ยนเพื่อไม่ให้ค้นหาแค่ส่วนหัวของงาน
        
        for a, b in indices[:LS_LIMIT]:
            neighbor_seq = apply_swap(current_seq, a, b)
            if is_valid_sequence(neighbor_seq, data):
                neigh_z1, neigh_z2 = heuristic_decoder(neighbor_seq, data)
                is_added = update_archive(archive, neighbor_seq, neigh_z1, neigh_z2)
                if is_added:
                    improvement_in_this_round = True
                        
        if improvement_in_this_round:
            no_improve_count = 0 
        else:
            no_improve_count += 1
            
    duration = time.time() - start_t
    return archive, duration

# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    
    dataset_path = os.path.join(BASE_DIR, TARGET_DATA_FOLDER)
    output_path = os.path.join(BASE_DIR, OUTPUT_FILENAME)

    print(f"📂 Reading files from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Folder not found! Please check path: {dataset_path}")
        exit()

    all_files = glob.glob(os.path.join(dataset_path, "*.alb"))
    print(f"📄 Found {len(all_files)} files. Running MO-VNS (Pareto Front Archive)...")

    results = []

    for idx, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(all_files)}] Processing: {filename} ", end="", flush=True)
        
        try:
            archive, duration = run_movns_for_instance(filepath)
            
            if archive:
                print(f"-> Found {len(archive)} Pareto solutions (Time: {duration:.2f}s)")
                # บันทึกทุกคำตอบที่อยู่ใน Pareto Front ของไฟล์นี้
                for sol_idx, (seq, z1, z2) in enumerate(archive):
                    results.append({
                        'Instance': filename,
                        'Solution_ID': sol_idx + 1,  # ลำดับของคำตอบใน Pareto Front
                        'Tasks': len(seq),
                        'Z1_Pareto': z1,
                        'Z2_Pareto': z2,
                        'Time_Sec': round(duration, 4)
                    })
            else:
                print("-> No solutions found.")
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    # Save Results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Finished! Results saved to: {output_path}")
        print(f"💡 Total Non-dominated solutions found across all instances: {len(df)}")
    else:
        print("\n⚠️ No results generated.")