# -*- coding: utf-8 -*-
import os
import glob
import time
import random
import collections
import pandas as pd
import math
import re

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = r'C:\GALBP\Hybrid VNS'
TARGET_DATA_FOLDER = 'GALBP_C'

# MOEA/D Parameters (Tuned by Taguchi Method)
POPULATION_SIZE = 100     # จำนวนประชากร (และจำนวนเวกเตอร์น้ำหนัก)
NEIGHBORHOOD_SIZE = 10    # ตัวแปร T: ขนาดของเพื่อนบ้านที่จะแลกเปลี่ยนข้อมูลกัน
MAX_GENERATIONS = 200
CROSSOVER_RATE = 1.0  # ล็อคไว้ที่ 1.0 ตามที่ตกลงกันไว้
MUTATION_RATE = 0.2

OUTPUT_FILENAME = f'Result_MOEAD_Soft_{TARGET_DATA_FOLDER}.csv'

# ==========================================
# 1. PARSER & HEURISTIC (ใช้ของเดิม)
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

def is_valid_sequence(sequence, data):
    pos = {task: idx for idx, task in enumerate(sequence)}
    for u, v in data['precedence']:
        if pos[u] >= pos[v]: return False
    return True

def generate_initial_sequence(data):
    num_tasks = data['num_tasks']
    in_degree = {i: 0 for i in range(1, num_tasks + 1)}
    adj = {i: [] for i in range(1, num_tasks + 1)}
    for u, v in data['precedence']:
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
# 2. GENETIC OPERATORS (PPX & MUTATION)
# ==========================================
def ppx_crossover(parent1, parent2):
    n = len(parent1)
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    template1 = [random.choice([1, 2]) for _ in range(n)]
    template2 = [1 if t == 2 else 2 for t in template1]
    def make_child(p1, p2, template):
        child, used = [], set()
        p1_idx, p2_idx = 0, 0
        for source in template:
            if source == 1:
                while p1[p1_idx] in used: p1_idx += 1
                task = p1[p1_idx]
            else:
                while p2[p2_idx] in used: p2_idx += 1
                task = p2[p2_idx]
            child.append(task)
            used.add(task)
        return child
    return make_child(parent1, parent2, template1), make_child(parent1, parent2, template2)

def mutate(sequence, data):
    if random.random() > MUTATION_RATE: return sequence
    new_seq = sequence.copy()
    for _ in range(10): 
        idx1, idx2 = random.sample(range(len(new_seq)), 2)
        temp_seq = new_seq.copy()
        temp_seq[idx1], temp_seq[idx2] = temp_seq[idx2], temp_seq[idx1]
        if is_valid_sequence(temp_seq, data): return temp_seq
    return sequence

# ==========================================
# 3. MOEA/D CORE COMPONENTS
# ==========================================
def get_euclidean_distance(w1, w2):
    return math.sqrt(sum((w1[i] - w2[i])**2 for i in range(len(w1))))

def generate_weight_vectors(pop_size):
    """สร้าง Weight Vectors แบบ Uniform กระจายตัว (สำหรับ 2 Objectives)"""
    weights = []
    for i in range(pop_size):
        w1 = i / (pop_size - 1)
        w2 = 1.0 - w1
        # เลี่ยงค่า 0 ดิบๆ ป้องกันการหารด้วยศูนย์ในฟังก์ชัน Tchebycheff
        w1 = max(w1, 1e-6)
        w2 = max(w2, 1e-6)
        weights.append((w1, w2))
    return weights

def get_neighborhoods(weights, T):
    """หาเพื่อนบ้านที่ใกล้ที่สุด T ลำดับแรกของแต่ละ Weight Vector"""
    B = []
    for i in range(len(weights)):
        distances = [(j, get_euclidean_distance(weights[i], weights[j])) for j in range(len(weights))]
        distances.sort(key=lambda x: x[1])
        B.append([idx for idx, dist in distances[:T]])
    return B

def calc_tchebycheff(z1, z2, weight, ideal_point):
    """คำนวณค่า Tchebycheff (ค่ายิ่งน้อยยิ่งดี)"""
    part1 = weight[0] * abs(z1 - ideal_point[0])
    part2 = weight[1] * abs(z2 - ideal_point[1])
    return max(part1, part2)

# ==========================================
# 4. MAIN MOEA/D ALGORITHM
# ==========================================
def run_moead_for_instance(filepath):
    data = parse_new_benchmark_format(filepath)
    if data is None or data['num_tasks'] == 0:
        return None, 0
    start_t = time.time()
    
    # 1. Setup Weights & Neighborhoods
    weights = generate_weight_vectors(POPULATION_SIZE)
    B = get_neighborhoods(weights, NEIGHBORHOOD_SIZE)
    
    # 2. Initialization & Ideal Point (Z*) Update
    population = []
    ideal_point = [float('inf'), float('inf')]
    
    for i in range(POPULATION_SIZE):
        seq = generate_initial_sequence(data)
        z1, z2 = heuristic_decoder(seq, data)
        population.append({'seq': seq, 'z1': z1, 'z2': z2})
        
        # อัปเดตจุดอุดมคติ (ค่าน้อยที่สุดที่เคยเจอ)
        if z1 < ideal_point[0]: ideal_point[0] = z1
        if z2 < ideal_point[1]: ideal_point[1] = z2

    # เก็บผลรวมของ Pareto Front เพื่อโชว์ตอนจบ (External Archive)
    external_archive = []

    # 3. Evolution Loop
    for gen in range(MAX_GENERATIONS):
        for i in range(POPULATION_SIZE):
            # สุ่มเลือกพ่อแม่จากกลุ่มเพื่อนบ้าน
            p1_idx, p2_idx = random.sample(B[i], 2)
            parent1 = population[p1_idx]['seq']
            parent2 = population[p2_idx]['seq']
            
            # Crossover & Mutation (เอาลูกคนแรกไปใช้)
            child_seq, _ = ppx_crossover(parent1, parent2)
            child_seq = mutate(child_seq, data)
            
            child_z1, child_z2 = heuristic_decoder(child_seq, data)
            
            # อัปเดต Ideal Point
            if child_z1 < ideal_point[0]: ideal_point[0] = child_z1
            if child_z2 < ideal_point[1]: ideal_point[1] = child_z2
                
            # Update Neighborhood Solutions
            for j in B[i]:
                current_tch = calc_tchebycheff(population[j]['z1'], population[j]['z2'], weights[j], ideal_point)
                new_tch = calc_tchebycheff(child_z1, child_z2, weights[j], ideal_point)
                
                # ถ้าลูกใหม่ทำค่า Tchebycheff ได้ดีกว่า ให้แทนที่ตัวเก่าของเพื่อนบ้านเลย
                if new_tch <= current_tch:
                    population[j] = {'seq': child_seq, 'z1': child_z1, 'z2': child_z2}
                    
            # แอบเก็บเข้า External Archive 
            external_archive.append((child_z1, child_z2))
            
    # คัดเฉพาะ Non-dominated solutions จาก External Archive
    def dominates_simple(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])
        
    unique_sols = list(set(external_archive))
    pareto_front = []
    for sol_p in unique_sols:
        is_dominated = False
        for sol_q in unique_sols:
            if dominates_simple(sol_q, sol_p):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append({"Z1": sol_p[0], "Z2": sol_p[1]})
            
    duration = time.time() - start_t
    return pareto_front, duration

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    dataset_path = os.path.join(BASE_DIR, TARGET_DATA_FOLDER)
    output_path = os.path.join(BASE_DIR, OUTPUT_FILENAME)

    print(f"📂 Reading files from: {dataset_path}")
    if not os.path.exists(dataset_path): exit()

    all_files = glob.glob(os.path.join(dataset_path, "*.alb"))
    print(f"📄 Found {len(all_files)} files. Running MOEA/D...")

    results = []
    for idx, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(all_files)}] Processing: {filename} ", end="", flush=True)
        try:
            pareto_front, duration = run_moead_for_instance(filepath)
            if pareto_front:
                print(f"-> Found {len(pareto_front)} Pareto solutions (Time: {duration:.2f}s)")
                for sol_idx, sol in enumerate(pareto_front):
                    results.append({
                        'Instance': filename,
                        'Solution_ID': sol_idx + 1,
                        'Z1_Pareto': sol['Z1'],
                        'Z2_Pareto': sol['Z2'],
                        'Time_Sec': round(duration, 4)
                    })
            else:
                print("-> No solutions.")
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"\n✅ Finished! Results saved to: {output_path}")