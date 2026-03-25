
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

# NSGA-II Parameters (Tuned by Taguchi Method)
POPULATION_SIZE = 50               # Factor A: ขนาดประชากร
MAX_GENERATIONS = 100              # Factor B: จำนวนรุ่น (Generation)
CROSSOVER_RATE = 1.0               # Factor C: อัตราการข้ามสายพันธุ์
MUTATION_RATE = 0.05               # Factor D: อัตราการกลายพันธุ์

OUTPUT_FILENAME = f'Result_NSGA2_Soft_{TARGET_DATA_FOLDER}.csv'

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
# 2. NSGA-II CORE COMPONENTS
# ==========================================
class Individual:
    def __init__(self, sequence):
        self.sequence = sequence
        self.z1 = 0
        self.z2 = 0
        self.rank = 0
        self.crowding_distance = 0

def dominates(ind1, ind2):
    return (ind1.z1 <= ind2.z1 and ind1.z2 <= ind2.z2) and (ind1.z1 < ind2.z1 or ind1.z2 < ind2.z2)

def fast_nondominated_sort(population):
    fronts = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in population:
            if dominates(p, q):
                p.dominated_solutions.append(q)
            elif dominates(q, p):
                p.domination_count += 1
        if p.domination_count == 0:
            p.rank = 1
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

def calculate_crowding_distance(front):
    l = len(front)
    if l == 0: return
    for ind in front: ind.crowding_distance = 0
    if l <= 2:
        for ind in front: ind.crowding_distance = float('inf')
        return
        
    for obj in ['z1', 'z2']:
        front.sort(key=lambda x: getattr(x, obj))
        front[0].crowding_distance = float('inf')
        front[l-1].crowding_distance = float('inf')
        
        obj_min = getattr(front[0], obj)
        obj_max = getattr(front[l-1], obj)
        obj_range = obj_max - obj_min
        if obj_range == 0: continue
            
        for i in range(1, l-1):
            front[i].crowding_distance += (getattr(front[i+1], obj) - getattr(front[i-1], obj)) / obj_range

def tournament_selection(population, k=2):
    best = random.choice(population)
    for _ in range(k-1):
        contender = random.choice(population)
        if contender.rank < best.rank:
            best = contender
        elif contender.rank == best.rank and contender.crowding_distance > best.crowding_distance:
            best = contender
    return best

# ==========================================
# 3. GENETIC OPERATORS (PPX & MUTATION)
# ==========================================
def ppx_crossover(parent1, parent2):
    n = len(parent1)
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
        
    template1 = [random.choice([1, 2]) for _ in range(n)]
    template2 = [1 if t == 2 else 2 for t in template1]
    
    def make_child(p1, p2, template):
        child = []
        used = set()
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
    if random.random() > MUTATION_RATE:
        return sequence
    new_seq = sequence.copy()
    for _ in range(10): # ให้โอกาสลองสุ่มหาสลับที่ถูกกฎ 10 ครั้ง
        idx1, idx2 = random.sample(range(len(new_seq)), 2)
        temp_seq = new_seq.copy()
        temp_seq[idx1], temp_seq[idx2] = temp_seq[idx2], temp_seq[idx1]
        if is_valid_sequence(temp_seq, data):
            return temp_seq
    return sequence

# ==========================================
# 4. MAIN NSGA-II ALGORITHM
# ==========================================
def run_nsga2_for_instance(filepath):
    data = parse_new_benchmark_format(filepath)
    if data is None or data['num_tasks'] == 0:
        return None, 0
    start_t = time.time()
    
    # 1. Initialization
    population = []
    for _ in range(POPULATION_SIZE):
        seq = generate_initial_sequence(data)
        ind = Individual(seq)
        ind.z1, ind.z2 = heuristic_decoder(seq, data)
        population.append(ind)
        
    fronts = fast_nondominated_sort(population)
    for front in fronts: calculate_crowding_distance(front)
    
    # 2. Evolution Loop
    for gen in range(MAX_GENERATIONS):
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            
            c1_seq, c2_seq = ppx_crossover(p1.sequence, p2.sequence)
            c1_seq = mutate(c1_seq, data)
            c2_seq = mutate(c2_seq, data)
            
            ind1, ind2 = Individual(c1_seq), Individual(c2_seq)
            ind1.z1, ind1.z2 = heuristic_decoder(c1_seq, data)
            ind2.z1, ind2.z2 = heuristic_decoder(c2_seq, data)
            
            offspring.extend([ind1, ind2])
            
        # Elitism (รวมประชากรเก่าและใหม่ แล้วคัดเฉพาะหัวกะทิ)
        combined_population = population + offspring
        fronts = fast_nondominated_sort(combined_population)
        
        population = []
        for front in fronts:
            calculate_crowding_distance(front)
            if len(population) + len(front) <= POPULATION_SIZE:
                population.extend(front)
            else:
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                population.extend(front[:POPULATION_SIZE - len(population)])
                break
                
    duration = time.time() - start_t
    
    # ดึงเฉพาะคำตอบที่ไม่ถูกข่มเลย (Rank 1) เป็นผลลัพธ์สุดท้าย
    pareto_front = fast_nondominated_sort(population)[0]
    
    # กรองคำตอบซ้ำ
    unique_pareto = []
    seen = set()
    for ind in pareto_front:
        obj_pair = (ind.z1, ind.z2)
        if obj_pair not in seen:
            seen.add(obj_pair)
            unique_pareto.append({"Z1": ind.z1, "Z2": ind.z2})
            
    return unique_pareto, duration

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    dataset_path = os.path.join(BASE_DIR, TARGET_DATA_FOLDER)
    output_path = os.path.join(BASE_DIR, OUTPUT_FILENAME)

    print(f"📂 Reading files from: {dataset_path}")
    if not os.path.exists(dataset_path): exit()

    all_files = glob.glob(os.path.join(dataset_path, "*.alb"))
    print(f"📄 Found {len(all_files)} files. Running NSGA-II...")

    results = []
    for idx, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(all_files)}] Processing: {filename} ", end="", flush=True)
        try:
            pareto_front, duration = run_nsga2_for_instance(filepath)
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