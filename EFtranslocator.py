#!/usr/bin/env python3
import os
import sys
import urllib.request
import argparse
import re
import subprocess
import pandas as pd
import numpy as np
import math
import shutil
import tarfile
import zipfile
import logging
import gzip
import io
import contextlib
import traceback
import difflib
from Bio import SeqIO
from collections import defaultdict

# Safely force headless backend for Colab/Setonix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

try:
    from upsetplot import from_memberships, plot as upset_plot
except ImportError:
    print("[Setup] 'upsetplot' not found. Installing automatically...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "upsetplot"], check=False, stdout=subprocess.DEVNULL)
    try:
        from upsetplot import from_memberships, plot as upset_plot
    except:
        upset_plot = None

# Kyte-Doolittle Hydropathy scale
KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

class DummyRecord:
    def __init__(self, id, seq, desc):
        self.id = id
        self.seq = seq
        self.description = desc

class TeeLogger:
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def make_executable(path):
    if not os.path.exists(path): return False
    try:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    p = os.path.join(root, f)
                    try: os.chmod(p, 0o755)
                    except: pass
        elif os.path.isfile(path):
            os.chmod(path, 0o755)
        return True
    except: return False

def run_logged_command(cmd, tool_name, outdir, env, cwd=None, mode='w', shell=False):
    log_out = os.path.join(outdir, f"{tool_name}_stdout.log")
    log_err = os.path.join(outdir, f"{tool_name}_stderr.log")
    with open(log_out, mode) as f_out, open(log_err, mode) as f_err:
        return subprocess.run(cmd, stdout=f_out, stderr=f_err, env=env, text=True, cwd=cwd, shell=shell)

def get_y_val(x_val, profile):
    try:
        idx = int(x_val) - 1
        if idx < 0 or idx >= len(profile): return 0.0
        y = profile[idx]
        return y if not np.isnan(y) else 0.0
    except: return 0.0

def calculate_kd_profile(sequence, window_size=21):
    seq_kd = [KD_SCALE.get(aa.upper(), 0.0) for aa in sequence]
    if len(seq_kd) < window_size: return np.array(seq_kd, dtype=float)
    profile = np.convolve(seq_kd, np.ones(window_size)/window_size, mode='valid')
    pad_start = window_size // 2
    pad_end = window_size - 1 - pad_start
    return np.pad(profile, (pad_start, pad_end), mode='edge')

def generate_polynomial_model():
    x = np.arange(1, 41)
    return -3.6107e-09*(x**6) + 3.8384e-07*(x**5) - 2.2462e-05*(x**4) + 8.0176e-04*(x**3) - 1.0307e-02*(x**2) - 3.1747e-02*x - 4.1017e-01

def get_random_forest_model():
    return np.array([
        -0.78215, -0.61565, -0.4853, -0.40685, -0.5311, -0.5162, -0.6248, -0.90385, -0.8529, -0.8105, 
        -0.75055, -0.9783, -1.0009, -0.96215, -0.8243, -0.77645, -0.71545, -0.5015, -0.2571, -0.07945, 
        -0.2892, -0.53705, -0.69715, -0.54385, -0.2392, -0.2486, -0.15525, -0.16675, -0.17765, -0.32, 
        -0.22875, 0.0741, 0.28285, 0.3341, 0.3352, 0.24265, 0.2893, 0.04435, -0.17495, -0.2114, -0.2379
    ][:40])

def calculate_r_squared(obs, pred):
    ss_tot = np.sum((obs - np.mean(obs))**2)
    return 0.0 if ss_tot == 0 else 1 - (np.sum((obs - pred)**2) / ss_tot)

def calculate_rmse(obs, pred):
    return np.sqrt(np.mean((obs - pred)**2))

def scan_translocator(profile, sequence, poly_model, rf_model, window_size, anchor_intervals, sp_len):
    pad_start = window_size // 2
    valid_profile = profile[pad_start : len(profile) - (window_size - 1 - pad_start)]
    if len(valid_profile) < 40: return [], -999.0, float('inf')
    passed_windows = []
    max_poly_r2, best_dist_seen = -999.0, float('inf')
    
    for i in range(len(valid_profile) - 40 + 1):
        window = valid_profile[i:i+40]
        poly_r2 = calculate_r_squared(window, poly_model)
        rf_r2 = calculate_r_squared(window, rf_model)
        poly_rmse = calculate_rmse(window, poly_model)
        rf_rmse = calculate_rmse(window, rf_model)
        
        seq_start, seq_end = i + pad_start, i + pad_start + 40
        if sp_len > 0 and (min(seq_end, sp_len) - max(seq_start + 1, 1) + 1) >= 16: continue
            
        min_dist = float('inf')
        for ds, de in anchor_intervals:
            d = ds - seq_end if seq_end <= ds else (seq_start - de if de <= seq_start else 0)
            if d < min_dist: min_dist = d

        if poly_r2 > max_poly_r2: max_poly_r2, best_dist_seen = poly_r2, min_dist
        if (poly_r2 >= 0.5 or rf_r2 >= 0.5) and min_dist <= 10:
            m_type = "Both" if (poly_r2 >= 0.5 and rf_r2 >= 0.5) else ("Poly" if poly_r2 >= 0.5 else "RF")
            passed_windows.append({'start': seq_start, 'end': seq_end, 'poly_r2': poly_r2, 'poly_rmse': poly_rmse, 'rf_r2': rf_r2, 'rf_rmse': rf_rmse, 'match_type': m_type})
    
    passed_windows.sort(key=lambda x: max(x['poly_r2'], x['rf_r2']), reverse=True)
    filtered_windows = []
    for w in passed_windows:
        if not any(not (w['end'] <= f['start'] or w['start'] >= f['end']) for f in filtered_windows):
            filtered_windows.append(w)
            
    filtered_windows.sort(key=lambda x: x['start'])
    return filtered_windows, max_poly_r2, best_dist_seen

def find_drek_motifs(sequence):
    drek_chars = set('DREK')
    active_indices = set()
    for i in range(len(sequence) - 3):
        if sum(1 for aa in sequence[i:i+4] if aa in drek_chars) >= 2: active_indices.update(range(i, i+4))
    if not active_indices: return []
    sorted_idx = sorted(list(active_indices))
    blocks, curr = [], [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if idx == curr[-1] + 1: curr.append(idx)
        else: blocks.append(curr); curr = [idx]
    blocks.append(curr)
    
    results = []
    for block in blocks:
        while block and sequence[block[0]] not in drek_chars: block.pop(0)
        while block and sequence[block[-1]] not in drek_chars: block.pop()
        if not block: continue
        seq_sub = sequence[block[0]:block[-1]+1]
        if len(seq_sub) >= 3 or (len(seq_sub) == 2 and all(aa in drek_chars for aa in seq_sub)): 
            results.append((block[0], block[-1]+1, seq_sub))
    return results

def find_electrostatic_patches(sequence):
    patches = []
    for i in range(len(sequence) - 39):
        window = sequence[i:i+40]
        if sum(1 for a in window if a in 'DREK') > 20:
            if np.mean([KD_SCALE.get(a.upper(), 0.0) for a in window]) <= -1.0:
                ed, kr = sum(1 for a in window if a in 'ED'), sum(1 for a in window if a in 'KR')
                if ed + kr > 0:
                    if ed / (ed + kr) > 0.6: patches.append([i, i+40, 'Polyacidic Patch'])
                    elif kr / (ed + kr) >= 0.6: patches.append([i, i+40, 'Polybasic Patch'])
                        
    merged = []
    for p_type in ['Polyacidic Patch', 'Polybasic Patch']:
        type_p = [p for p in patches if p[2] == p_type]
        if not type_p: continue
        type_p.sort(key=lambda x: x[0])
        curr = type_p[0]
        for nxt in type_p[1:]:
            if nxt[0] <= curr[1]: curr[1] = max(curr[1], nxt[1])
            else:
                merged.append({'category': 'Electrostatic Patch', 'name': p_type, 'start': curr[0]+1, 'end': curr[1], 'cut_pos': None, 'type': 'regex', 'raw': sequence[curr[0]:curr[1]], 'evalue': None, 'pfam_id': '', 'motif': '', 'description': 'Electrostatic property match', 'notes': '', 'matching_seq': sequence[curr[0]:curr[1]]})
                curr = nxt
        merged.append({'category': 'Electrostatic Patch', 'name': p_type, 'start': curr[0]+1, 'end': curr[1], 'cut_pos': None, 'type': 'regex', 'raw': sequence[curr[0]:curr[1]], 'evalue': None, 'pfam_id': '', 'motif': '', 'description': 'Electrostatic property match', 'notes': '', 'matching_seq': sequence[curr[0]:curr[1]]})
    return merged

def find_kep_repeats(mature_seq, sp_len, mature_cuts, threshold):
    kex2_pos = [c[0] for c in mature_cuts if c[2] == "Kex2"]
    if not kex2_pos: return []
    boundaries = [0] + sorted(kex2_pos) + [len(mature_seq)]
    fragments = []
    for i in range(len(boundaries)-1):
        s, e = boundaries[i], boundaries[i+1]
        seq = mature_seq[s:e]
        if len(seq) > 3: fragments.append({'start': s + sp_len + 1, 'end': e + sp_len, 'seq': seq})
            
    groups = []
    for f in fragments:
        added = False
        for g in groups:
            rep_seq = g[0]['seq']
            if difflib.SequenceMatcher(None, f['seq'], rep_seq).ratio() >= threshold:
                g.append(f)
                added = True
                break
        if not added: groups.append([f])
            
    kep_repeats = []
    for g in groups:
        if len(g) >= 2:
            max_len = max(len(x['seq']) for x in g)
            regex = ""
            for i in range(max_len):
                chars = set(x['seq'][i] for x in g if i < len(x['seq']))
                valid_chars = [c for c in chars if c in KD_SCALE]
                if len(valid_chars) > 1: regex += "[" + "".join(sorted(valid_chars)) + "]"
                elif len(valid_chars) == 1: regex += valid_chars[0]
            if regex:
                kep_repeats.append({'regex': regex, 'instances': g, 'freq': len(g)})
            
    return kep_repeats

# ================= DYNAMIC DOMAIN SCANNER =================
def fetch_pfam_hmm(pfam_id, hmm_dir):
    os.makedirs(hmm_dir, exist_ok=True)
    out_path = os.path.join(hmm_dir, f"{pfam_id}.hmm")
    if not os.path.exists(out_path):
        url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam_id}/?annotation=hmm"
        try:
            req = urllib.request.Request(url, headers={'Accept': '*/*'})
            with urllib.request.urlopen(req) as response:
                content = response.read()
                if content.startswith(b'\x1f\x8b'):
                    try:
                        with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as tar:
                            for member in tar.getmembers():
                                if member.name.endswith('.hmm'):
                                    content = tar.extractfile(member).read()
                                    break
                    except:
                        content = gzip.decompress(content)
                if b'HMMER3/f' not in content[:50]: return None
                with open(out_path, 'wb') as out_file: out_file.write(content)
        except: return None
    return out_path

def get_csv_col(df, possible_names):
    lower_cols = {c.strip().lower(): c for c in df.columns}
    for p in possible_names:
        if p.lower() in lower_cols: return lower_cols[p.lower()]
    return None

def filter_and_merge_domains(domains):
    pfams = [d for d in domains if d['type'] == 'hmm']
    non_pfams = [d for d in domains if d['type'] != 'hmm']
    
    pfam_groups = defaultdict(list)
    for p in pfams: pfam_groups[p['pfam_id']].append(p)
    
    filtered_pfams = []
    for pid, grp in pfam_groups.items():
        grp.sort(key=lambda x: x['start'])
        resolved = []
        for h in grp:
            overlap_idx = -1
            for i, rh in enumerate(resolved):
                if not (h['end'] < rh['start'] or h['start'] > rh['end']):
                    overlap_idx = i; break
            if overlap_idx != -1:
                rh = resolved[overlap_idx]
                if (h['end'] - h['start']) > (rh['end'] - rh['start']) or ((h['end'] - h['start']) == (rh['end'] - rh['start']) and h['evalue'] < rh['evalue']):
                    resolved[overlap_idx] = h
            else:
                resolved.append(h)
        filtered_pfams.extend(resolved)
        
    all_domains = filtered_pfams + non_pfams
    conflated = []
    type_groups = defaultdict(list)
    for d in all_domains: type_groups[(d['category'], d['name'])].append(d)
        
    for k, grp in type_groups.items():
        grp.sort(key=lambda x: x['start'])
        curr = grp[0]
        for nxt in grp[1:]:
            if nxt['start'] <= curr['end']:
                curr['end'] = max(curr['end'], nxt['end'])
                curr['matching_seq'] = curr['raw']
            else:
                conflated.append(curr)
                curr = nxt
        conflated.append(curr)
        
    seen = set()
    final_domains = []
    for d in conflated:
        tup = (d['category'], d['name'], d['start'], d['end'])
        if tup not in seen:
            seen.add(tup)
            final_domains.append(d)
            
    return final_domains

def scan_domains(fasta_file, records, data_dir, env, outdir):
    print("  -> Scanning Domains and Pfams...", flush=True)
    domain_results = {rec.id: [] for rec in records}
    csv_path = os.path.join(data_dir, "domains.csv")
    hmm_dir = os.path.join(data_dir, "domains.hmm")
    logs_dir = os.path.join(outdir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    log_out = os.path.join(logs_dir, "hmmsearch_stdout.log")
    log_err = os.path.join(logs_dir, "hmmsearch_stderr.log")
    open(log_out, 'w').close(); open(log_err, 'w').close()
    
    if not os.path.exists(csv_path): return domain_results
    try: df = pd.read_csv(csv_path)
    except: return domain_results
    
    name_col = get_csv_col(df, ['Domain Name', 'Domain', 'Name', 'Motif', 'Domain/Motif', 'Feature'])
    desc_col = get_csv_col(df, ['Description', 'Desc'])
    notes_col = get_csv_col(df, ['Notes', 'Note'])
    
    for _, row in df.iterrows():
        cat = str(row.get('Category', '')).strip()
        pfam = str(row.get('Pfam ID', 'N/A')).strip()
        motif = str(row.get('Motif Pattern', str(row.get('Motif', 'N/A')))).strip()
        
        name = str(row[name_col]).strip() if name_col else ""
        if name in ['Unknown', 'nan', 'None', '']: name = pfam if pfam != 'N/A' and pfam != 'nan' else motif
        desc = str(row[desc_col]).strip() if desc_col and str(row[desc_col]) != 'nan' else ""
        notes = str(row[notes_col]).strip() if notes_col and str(row[notes_col]) != 'nan' else ""
        
        if pfam != 'N/A' and pfam != 'nan':
            pfam_list = set(re.findall(r'PF\d{5}', pfam))
            for single_pfam in pfam_list:
                hmm_path = fetch_pfam_hmm(single_pfam, hmm_dir)
                if hmm_path and os.path.exists(hmm_path) and shutil.which('hmmsearch'):
                    dom_out = os.path.join(data_dir, f"{single_pfam}_tmp.domtbl")
                    run_logged_command(['hmmsearch', '--noali', '--domtblout', dom_out, hmm_path, fasta_file], 'hmmsearch', logs_dir, env, mode='a')
                    if os.path.exists(dom_out):
                        with open(dom_out, 'r') as f:
                            for line in f:
                                if line.startswith('#'): continue
                                parts = line.split()
                                if len(parts) > 16:
                                    seq_id = parts[0]
                                    evalue = float(parts[6]) 
                                    if evalue < 1e-3 and seq_id in domain_results:
                                        s_start, s_end = int(parts[15]), int(parts[16])
                                        rec = next((r for r in records if r.id == seq_id), None)
                                        m_seq = rec.seq[s_start-1:s_end] if rec else ""
                                        domain_results[seq_id].append({
                                            'category': cat, 'name': name, 'start': s_start, 'end': s_end,
                                            'evalue': evalue, 'type': 'hmm', 'raw': f"PFAM:{single_pfam}", 'cut_pos': None,
                                            'pfam_id': single_pfam, 'motif': '', 'description': desc, 'notes': notes,
                                            'matching_seq': m_seq
                                        })
                        os.remove(dom_out)
        
        if motif != 'N/A' and motif != 'nan':
            motif_clean = motif.replace('X', '.').replace('x', '.')
            cleavage_mode = '/' in motif_clean
            safe_regex = f"({motif_clean.split('/')[0]})({motif_clean.split('/')[1]})" if cleavage_mode else f"({motif_clean})"
            for rec in records:
                for match in re.finditer(safe_regex, rec.seq, re.IGNORECASE):
                    domain_results[rec.id].append({
                        'category': cat, 'name': name, 'start': match.start() + 1, 'end': match.end(),
                        'cut_pos': match.end(1) if cleavage_mode else None, 'type': 'regex', 'raw': match.group(0), 'evalue': None,
                        'pfam_id': '', 'motif': motif_clean, 'description': desc, 'notes': notes, 'matching_seq': match.group(0)
                    })
                    
    for rec in records:
        domain_results[rec.id].extend(find_electrostatic_patches(rec.seq))
        domain_results[rec.id] = filter_and_merge_domains(domain_results[rec.id])
        
    return domain_results

# ================= EXTERNAL TOOLS & PRE-CHECKS =================

def patch_rippminer_paths(rm_dir, abs_bin=None):
    # Detect system path for blastp and obabel if possible, fallback to /usr/local/bin
    blast_exe = shutil.which('blastp')
    blast_dir = os.path.dirname(blast_exe) if blast_exe else "/usr/local/bin"
    openbabel_exe = shutil.which('obabel')
    openbabel_dir = os.path.dirname(openbabel_exe) if openbabel_exe else "/usr/local/bin"

    # 1. Update paths file
    path_file = os.path.join(rm_dir, "path")
    if os.path.exists(path_file):
        try:
            with open(path_file, 'w') as f:
                f.write(f"openbabel={openbabel_dir}\nblast={blast_dir}\n")
        except: pass
        
    # Download and install 64-bit Linux svm_light, svm_multiclass, and Weka binaries if missing
    scripts_dir = os.path.join(rm_dir, 'scripts')
    svm_classify_path = os.path.join(scripts_dir, 'svm_classify')
    svm_multiclass_path = os.path.join(scripts_dir, 'svm_multiclass_classify')
    weka_jar_path = os.path.join(scripts_dir, 'weka-3-6-14', 'weka.jar')
    
    if os.path.exists(rm_dir) and (not os.path.exists(svm_classify_path) or not os.path.exists(svm_multiclass_path) or not os.path.exists(weka_jar_path)):
        try:
            # 1. svm_light
            if not os.path.exists(svm_classify_path):
                print("     [RiPPMiner]: Downloading and installing 64-bit svm_light binary...", flush=True)
                url_svm = "http://download.joachims.org/svm_light/current/svm_light_linux64.tar.gz"
                req = urllib.request.Request(url_svm, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    with tarfile.open(fileobj=io.BytesIO(response.read()), mode='r:gz') as tar:
                        tar.extractall(path=scripts_dir)
                make_executable(svm_classify_path)
            
            # 2. svm_multiclass
            if not os.path.exists(svm_multiclass_path):
                print("     [RiPPMiner]: Downloading and installing 64-bit svm_multiclass binary...", flush=True)
                url_multi = "http://download.joachims.org/svm_multiclass/current/svm_multiclass_linux64.tar.gz"
                req = urllib.request.Request(url_multi, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    with tarfile.open(fileobj=io.BytesIO(response.read()), mode='r:gz') as tar:
                        tar.extractall(path=scripts_dir)
                make_executable(svm_multiclass_path)
            
            # 3. Weka
            if not os.path.exists(weka_jar_path):
                # Try using local weka zip from Localizer first to save bandwidth and work offline
                local_weka_zip = None
                if abs_bin:
                    local_weka_zip = os.path.join(os.path.abspath(abs_bin), "LOCALIZER-1.0.5", "Scripts", "weka-3-6-12.zip")
                
                extracted_locally = False
                if local_weka_zip and os.path.exists(local_weka_zip):
                    print("     [RiPPMiner]: Extracting Weka locally from LOCALIZER package...", flush=True)
                    try:
                        with zipfile.ZipFile(local_weka_zip) as zip_ref:
                            zip_ref.extractall(path=scripts_dir)
                        # Rename extracted weka-3-6-12 to weka-3-6-14
                        extracted_dir = os.path.join(scripts_dir, "weka-3-6-12")
                        target_dir = os.path.join(scripts_dir, "weka-3-6-14")
                        if os.path.exists(extracted_dir):
                            if os.path.exists(target_dir):
                                shutil.rmtree(target_dir)
                            os.rename(extracted_dir, target_dir)
                            extracted_locally = True
                            print("     [RiPPMiner]: Weka locally cloned and renamed successfully.", flush=True)
                    except Exception as e:
                        print(f"     [RiPPMiner]: Local Weka copy failed ({e}). Falling back to download...", flush=True)
                
                if not extracted_locally:
                    print("     [RiPPMiner]: Downloading Weka 3.6.14...", flush=True)
                    url_weka = "https://sourceforge.net/projects/weka/files/weka-3-6/3.6.14/weka-3-6-14.zip/download"
                    zip_path = os.path.join(scripts_dir, "weka-3-6-14.zip")
                    
                    download_success = False
                    if shutil.which('wget'):
                        try:
                            subprocess.run(['wget', '--no-check-certificate', '-q', '-O', zip_path, url_weka], check=True)
                            download_success = True
                        except: pass
                    
                    if not download_success and shutil.which('curl'):
                        try:
                            subprocess.run(['curl', '-L', '-s', '-o', zip_path, url_weka], check=True)
                            download_success = True
                        except: pass
                    
                    if not download_success:
                        try:
                            req = urllib.request.Request(url_weka, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req) as response:
                                with open(zip_path, 'wb') as f:
                                    f.write(response.read())
                            download_success = True
                        except Exception as e:
                            raise RuntimeError(f"Failed to download Weka: {e}")
                    
                    if download_success and os.path.exists(zip_path):
                        with zipfile.ZipFile(zip_path) as zip_ref:
                            zip_ref.extractall(path=scripts_dir)
                        os.remove(zip_path)
                        print("     [RiPPMiner]: Weka 3.6.14 successfully installed.", flush=True)
        except Exception as e:
            print(f"     [RiPPMiner Error]: Failed to download/install svm/weka tools: {e}", flush=True)

    # 2. Prevent "Can't exec ''" and fix system call in chech_fasta_input.pl
    script1 = os.path.join(rm_dir, "scripts", "chech_fasta_input.pl")
    if os.path.exists(script1):
        try:
            with open(script1, 'r') as f: content = f.read()
            content = content.replace('system(`rm -f $ARGV[0]/cyclizationInput.fasta`);', 'unlink("$ARGV[0]/cyclizationInput.fasta");')
            with open(script1, 'w') as f: f.write(content)
        except: pass

    # 2b. Fix extract_core.pl system call bug
    script_ec = os.path.join(rm_dir, "scripts", "extract_core.pl")
    if os.path.exists(script_ec):
        try:
            with open(script_ec, 'r') as f: content = f.read()
            content = content.replace('system(`rm -f $ARGV[0]/cyclizationInput.fasta $ARGV[0]/cyclizationInput1.fasta`);', 'unlink("$ARGV[0]/cyclizationInput.fasta", "$ARGV[0]/cyclizationInput1.fasta");')
            with open(script_ec, 'w') as f: f.write(content)
        except: pass

    # 2c. Fix calculateKmer.pl system call bug
    script_ck = os.path.join(rm_dir, "scripts", "calculateKmer.pl")
    if os.path.exists(script_ck):
        try:
            with open(script_ck, 'r') as f: content = f.read()
            content = content.replace('system(`rm -f kmerlist`);', 'unlink("kmerlist");')
            with open(script_ck, 'w') as f: f.write(content)
        except: pass
        
    # 3. Eliminate empty variable dependency for blast and strictly enforce pure system calls
    script3 = os.path.join(rm_dir, "sequence_similarity_search.pl")
    if os.path.exists(script3):
        try:
            with open(script3, 'r') as f: content = f.read()
            content = re.sub(r'\$Path_to_Blast_bin\s*=\s*[\'"].*?[\'"];', '$Path_to_Blast_bin = "";', content)
            content = re.sub(r'[\'"]?\$Path_to_Blast_bin[\'"]?\s*\.?\s*[\'"]/*blastp[\'"]?', '"blastp"', content)
            content = re.sub(r'[\'"]?\$Path_to_Blast_bin[\'"]?\s*\.?\s*[\'"]/*makeblastdb[\'"]?', '"makeblastdb"', content)
            content = re.sub(r'\$Path_to_Blast_bin(?=\s+-query)', 'blastp', content)
            content = re.sub(r'\$Path_to_Blast_bin(?=\s+-in)', 'makeblastdb', content)
            with open(script3, 'w') as f: f.write(content)
        except: pass

    # 4. Patch OpenBabel structure search
    script4 = os.path.join(rm_dir, "closest_structure_search.pl")
    if os.path.exists(script4):
        try:
            with open(script4, 'r') as f: content = f.read()
            content = re.sub(r'\$Path_to_openbabel_bin\s*=\s*[\'"].*?[\'"];', f'$Path_to_openbabel_bin = "{openbabel_dir}";', content)
            with open(script4, 'w') as f: f.write(content)
        except: pass

    # 5. Repair the prediction loop truncation caused by previous faulty string replaces
    rm_script = os.path.join(rm_dir, "run_rippminer.pl")
    if os.path.exists(rm_script):
        try:
            with open(rm_script, 'r') as f: content = f.read()
            content = re.sub(r'cat\s+prediction\.out\s*>>\s*prediction\.(outputput|output|out)', 'cat prediction.out >> prediction.output', content)
            with open(rm_script, 'w') as f: f.write(content)
        except: pass


def precheck_binaries(bin_dir, outdir):
    abs_bin = os.path.abspath(bin_dir)
    print("\n--- PRE-FLIGHT BINARY VERIFICATION ---", flush=True)
    warnings = []
    os.makedirs(abs_bin, exist_ok=True)
    logs_dir = os.path.join(outdir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # TargetP Check
    tp = os.path.join(abs_bin, 'targetp', 'TargetP-2.0', 'bin', 'targetp')
    if not os.path.exists(tp):
        warnings.append("TargetP: Missing. Please place the targetp-2.0.Linux.tar.gz archive in the bin/ folder and extract it.")
    else: make_executable(os.path.join(abs_bin, 'targetp'))
        
    # WoLF PSORT Check
    wp_dir = os.path.join(abs_bin, "WoLFPSort")
    if not os.path.exists(wp_dir):
        warnings.append("WoLF PSORT: Missing. Clone/download into bin/WoLFPSort.")
    else: make_executable(wp_dir)

    # MultiLoc2 Configure Check
    ml2_dir = os.path.join(abs_bin, "MultiLoc2-master", "MultiLoc2")
    ml2_config = os.path.join(ml2_dir, "configureML2.py")
    ml2_pred = os.path.join(ml2_dir, "src", "multiloc2_prediction.py")
    ml2_configured = False
    
    if os.path.exists(ml2_pred):
        try:
            with open(ml2_pred, 'r') as f:
                content = f.read()
                if re.search(r'src_path\s*=\s*["\'].+["\']', content):
                    ml2_configured = True
        except: pass

    if os.path.exists(ml2_config) and not ml2_configured:
        print(f"     [MultiLoc2]: Running configureML2.py with python2...", flush=True)
        py2_exec = shutil.which('python2') or shutil.which('python2.7') or 'python2'
        try:
            ml2_env = os.environ.copy()
            if 'PYTHONPATH' in ml2_env:
                del ml2_env['PYTHONPATH']
            run_logged_command([py2_exec, "configureML2.py"], 'multiloc2_config', logs_dir, ml2_env, cwd=ml2_dir)
        except Exception as e:
            warnings.append(f"MultiLoc2 Config failed (is python2 installed?): {e}")

    # Programmatic patch for MultiLoc2's svm_phyloloc.py to use batch BLAST (speed up by ~120x)
    ml2_phylo = os.path.join(ml2_dir, "src", "svm_phyloloc.py")
    if os.path.exists(ml2_phylo):
        try:
            with open(ml2_phylo, 'r') as f:
                phylo_content = f.read()
            if "[PhyloLoc Progress]" not in phylo_content:
                print(f"     [MultiLoc2]: Patching svm_phyloloc.py with batch BLAST optimization...", flush=True)
                predict_idx = phylo_content.find("def predict(")
                if predict_idx != -1:
                    new_predict_code = r"""def predict(origin,table,path,data,model,libsvm_path,blast_path,genome_path, id=1):
    model=str(model)

    proteins = util.parse_fasta_file(data)
    if not proteins:
        return []

    file_path = tmpfile_path+"/"+str(id)
    
    # 1. Write all proteins to a single batch fasta file
    batch_fasta_path = "%squery_batch.fasta" % file_path
    bf = open(batch_fasta_path, "w")
    for prot in proteins:
        bf.write(">" + prot['id'] + "\n" + prot['sequence'] + "\n")
    bf.close()
    
    # 2. Build database of the batch fasta file
    cmd = 'nice -n 19 "%s/makeblastdb" -in "%s" -dbtype prot' % (blast_path, batch_fasta_path)
    os.system(cmd)
    
    # 3. Blast batch fasta file against itself to get self bit scores
    self_blast_out = "%sself_blast.txt" % file_path
    cmd = 'nice -n 19 "%s/blastp" -db "%s" -query "%s" -outfmt 7 -out "%s"' % (blast_path, batch_fasta_path, batch_fasta_path, self_blast_out)
    os.system(cmd)
    
    # 4. Parse self bit scores
    protein_self_bit_score_map = {}
    if os.path.exists(self_blast_out):
        sbf = open(self_blast_out, "r")
        for line in sbf:
            if line.startswith("#") or not line.strip():
                continue
            tokens = line.split("\t")
            if len(tokens) >= 12 and tokens[0] == tokens[1]:
                if tokens[0] not in protein_self_bit_score_map:
                    protein_self_bit_score_map[tokens[0]] = float(tokens[11])
        sbf.close()
        try: os.remove(self_blast_out)
        except: pass
        
    # Clean up batch database files
    for ext in [".phr", ".pin", ".psq"]:
        if os.path.exists(batch_fasta_path + ext):
            try: os.remove(batch_fasta_path + ext)
            except: pass
            
    # 5. Load genomeList
    genomeList = []
    lfile=open(genome_path+"/ordered_ncbi_taxIDs.dat",'r')
    while 1:
        line=lfile.readline()
        if not line: break
        line=re.sub("\n","",line) + ".faa"
        genomeList.append(line)
    lfile.close()
    lfile=open(genome_path+"/ordered_ncbi_taxIDs_archaea.dat",'r')
    while 1:
        line=lfile.readline()
        if not line: break
        line=re.sub("\n","",line) + ".faa"
        genomeList.append(line)
    lfile.close()
    lfile=open(genome_path+"/ordered_ncbi_taxIDs_eukaryota.dat",'r')
    while 1:
        line=lfile.readline()
        if not line: break
        line=re.sub("\n","",line) + ".faa"
        genomeList.append(line)
    lfile.close()
    
    # Initialize profile scores dictionary for all proteins
    proteins2 = {}
    for prot in proteins:
        proteins2[prot['id']] = [0.0] * len(genomeList)
        
    # 6. Run BLAST against reference genomes in batch
    total_genomes_to_run = 0
    for i in range(len(genomeList)):
        if i>24 and i <400:
            continue
        total_genomes_to_run += 1
        
    current_genome_idx = 0
    
    for i in range(len(genomeList)):
        if i>24 and i <400:
            continue
            
        current_genome_idx += 1
        
        db_path = genome_path+"/genomes/Bacteria/all/" + genomeList[i]
        if i >=400 and i <433:
            db_path = genome_path+"/genomes/Archaea/" + genomeList[i]
        if i >=433:
            db_path = genome_path+"/genomes/Eukaryota/" + genomeList[i]
            
        if not os.path.exists(db_path):
            print("[PhyloLoc Warning] Genome database file %s not found. Skipping." % db_path)
            sys.stdout.flush()
            continue
            
        # Ensure genome database is formatted
        if os.path.exists(db_path + ".pin") == False:
            cmd = 'nice -n 19 "%s/makeblastdb" -in "%s" -dbtype prot' % (blast_path, db_path)
            os.system(cmd)
            
        # Run blastp
        blastoutput_path = "%s_phylo_blast_%s.txt" % (file_path, genomeList[i])
        
        print("[PhyloLoc Progress] Genome %d/%d (%s) against %d sequences..." % (current_genome_idx, total_genomes_to_run, genomeList[i], len(proteins)))
        sys.stdout.flush()
        
        cmd = 'nice -n 19 "%s/blastp" -db "%s" -query "%s" -outfmt 7 -out "%s"' % (blast_path, db_path, batch_fasta_path, blastoutput_path)
        os.system(cmd)
        
        # Parse blast results for this genome
        if os.path.exists(blastoutput_path):
            bf = open(blastoutput_path, "r")
            seen_queries = set()
            for line in bf:
                if line.startswith("#") or not line.strip():
                    continue
                tokens = line.split("\t")
                if len(tokens) >= 12:
                    q_id = tokens[0]
                    if q_id not in seen_queries:
                        seen_queries.add(q_id)
                        bit_score_raw = float(tokens[11])
                        if q_id in protein_self_bit_score_map:
                            bit_score = bit_score_raw / protein_self_bit_score_map[q_id]
                            proteins2[q_id][i] = bit_score
            bf.close()
            try: os.remove(blastoutput_path)
            except: pass
            
    # Remove batch fasta file
    if os.path.exists(batch_fasta_path):
        try: os.remove(batch_fasta_path)
        except: pass
        
    # Write feature vectors to test_svm.dat
    input_file = open("%stest_svm.dat" % file_path, 'w')
    no_fv_proteins = []
    for prot in proteins:
        pid = prot['id']
        if pid not in protein_self_bit_score_map:
            no_fv_proteins.append(pid)
            continue
            
        evalues = proteins2[pid]
        featurevector = ""
        for i in range(0, len(evalues)):
            if i>24 and i <400:
                continue
            if featurevector == "":
                featurevector="%s:%s" %(i+1,evalues[i])
            else:
                featurevector=featurevector + " %s:%s" %(i+1,evalues[i])
        
        if featurevector != "":
            input_file.write("0 " + featurevector + "\n")
        else:
            no_fv_proteins.append(pid)
    input_file.close()
    
    return util.predict_one_vs_one(table,origin,model,path,libsvm_path,tmpfile_path,id,proteins,no_fv_proteins)

def animal_predict(table,path,data,model,libsvm_path,blast_path,genome_path, id=1):
	return predict("animal",table,path,data,model,libsvm_path,blast_path,genome_path, id)

def fungi_predict(table,path,data,model,libsvm_path,blast_path,genome_path, id=1):
	return predict("fungi",table,path,data,model,libsvm_path,blast_path,genome_path, id)

def plant_predict(table,path,data,model,libsvm_path,blast_path,genome_path, id=1):
	return predict("plant",table,path,data,model,libsvm_path,blast_path,genome_path, id)
"""
                    patched_content = phylo_content[:predict_idx] + new_predict_code
                    with open(ml2_phylo, 'w') as f:
                        f.write(patched_content)
                    print(f"     [MultiLoc2]: Patch applied successfully.", flush=True)
        except Exception as patch_err:
            print(f"     [MultiLoc2 Patch Error]: {patch_err}", flush=True)

    # RiPPMiner Check & Auto-Extract/Patch
    rm_dir = os.path.join(abs_bin, 'rippminer_standalone')
    rm = os.path.join(rm_dir, 'run_rippminer.pl')
    if not os.path.exists(rm):
        tar_path = os.path.join(abs_bin, 'rippminer_standalone.tar.gz')
        if os.path.exists(tar_path):
            print(f"     [RiPPMiner]: Found archive {tar_path}. Extracting and patching...", flush=True)
            try:
                with tarfile.open(tar_path, "r:gz") as tar: tar.extractall(path=abs_bin)
                patch_rippminer_paths(rm_dir, abs_bin)
                make_executable(rm_dir)
            except Exception as e: warnings.append(f"RiPPMiner: Failed to extract tarball: {e}")
        else:
            warnings.append("RiPPMiner: Missing. Please place rippminer_standalone.tar.gz into bin/")
    else: 
        patch_rippminer_paths(rm_dir, abs_bin)
        make_executable(rm_dir)

    # DeepLoc2 Setup Check
    dl_dir = os.path.join(abs_bin, 'deeploc2_package')
    tar_path_dl = os.path.join(abs_bin, 'deeploc-2.1-All.tar.gz')
    if os.path.exists(tar_path_dl):
        if not os.path.exists(dl_dir):
            print(f"     [DeepLoc2]: Forced extraction of tarball {tar_path_dl}...", flush=True)
            try:
                with tarfile.open(tar_path_dl, "r:gz") as tar: tar.extractall(path=abs_bin)
                if os.path.exists(os.path.join(abs_bin, 'deeploc2')) and not os.path.exists(dl_dir):
                    os.rename(os.path.join(abs_bin, 'deeploc2'), dl_dir)
            except Exception as e:
                 warnings.append(f"DeepLoc2: Failed to extract tarball: {e}")
            
    if os.path.exists(dl_dir):
        dl_env_dir = os.path.join(abs_bin, 'deeploc_env')
        if not os.path.exists(os.path.join(dl_env_dir, 'esm')):
            print(f"     [DeepLoc2]: Installing isolated fair-esm to {dl_env_dir}...", flush=True)
            os.makedirs(dl_env_dir, exist_ok=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', dl_env_dir, 'fair-esm'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        deeploc_installed = shutil.which('deeploc2') or os.path.exists(os.path.join(dl_dir, 'bin', 'deeploc2')) or os.path.exists(os.path.join(sys.prefix, 'bin', 'deeploc2'))
        if not deeploc_installed:
            print(f"     [DeepLoc2]: Forced running pip install . in {dl_dir}...", flush=True)
            run_logged_command([sys.executable, '-m', 'pip', 'install', '.'], 'deeploc_install', logs_dir, os.environ.copy(), cwd=dl_dir)
            if not shutil.which('deeploc2') and not os.path.exists(os.path.join(dl_dir, 'bin', 'deeploc2')):
                pip_bin = os.path.join(sys.prefix, 'bin', 'deeploc2')
                if not os.path.exists(pip_bin): warnings.append("DeepLoc2: Installed but executable not found in path.")
    else:
        warnings.append("DeepLoc2: Missing. Place deeploc-2.1-All.tar.gz in bin/.")

    # AIUPred Check
    ai_dir = os.path.join(abs_bin, 'aiupred')
    ai_exe_local = os.path.join(ai_dir, 'bin', 'aiupred')
    ai_exe_local_win = os.path.join(ai_dir, 'Scripts', 'aiupred.exe')
    if not os.path.exists(ai_exe_local) and not os.path.exists(ai_exe_local_win) and not shutil.which("aiupred"):
        print(f"     [AIUPred]: Installing via pip from ZIP archive to {ai_dir}...", flush=True)
        os.makedirs(ai_dir, exist_ok=True)
        run_logged_command([sys.executable, '-m', 'pip', 'install', '--target', ai_dir, 'https://github.com/doszilab/AIUPred/archive/refs/heads/master.zip'], 'aiupred_install', logs_dir, os.environ.copy())

 # ApoplastP Check & Auto-Extract/Patch
    apo_dir = os.path.join(abs_bin, "ApoplastP_1.0.1")
    tar_path_apo = os.path.join(abs_bin, "ApoplastP_1.0.1.tar.gz")
    if os.path.exists(tar_path_apo) and not os.path.exists(apo_dir):
        print(f"     [ApoplastP]: Found archive {tar_path_apo}. Extracting and patching...", flush=True)
        try:
            with tarfile.open(tar_path_apo, "r:gz") as tar:
                tar.extractall(path=abs_bin)
        except Exception as e:
            print(f"     [ApoplastP]: Failed to extract archive: {e}", flush=True)

    if os.path.exists(apo_dir): 
        apo_scripts = os.path.join(apo_dir, "Scripts")
        for file in ['ApoplastP.py', 'functions.py']:
            script_path = os.path.join(apo_scripts, file)
            if os.path.exists(script_path):
                try:
                    with open(script_path, 'r') as f: content = f.read()
                    original_content = content
                    
                    # 1. Repair corruption from previous faulty regex patches that caused unterminated string literals
                    content = content.replace("print('''\n", "print()\n")
                    content = content.replace('print("""\n', "print()\n")
                    
                    # 2. Fix exceptions for Python 3
                    content = re.sub(r'(\s*)except IOError as \(errno,\s*strerror\):', r'\1except IOError as e:\n\1    errno, strerror = e.args[:2] if len(e.args) >= 2 else (0, str(e))', content)
                    content = content.replace("PEPSTATS_PATH = SCRIPT_PATH + '/EMBOSS-6.5.7/emboss/'", "PEPSTATS_PATH = '/usr/bin/'")
                    
                    # 3. Cleanly wrap remaining Python 2 prints in parentheses
                    lines = content.split('\n')
                    for i in range(len(lines)):
                        m = re.match(r'^(\s*)print\s+(?![\(])(.*?)\s*$', lines[i])
                        if m:
                            lines[i] = f"{m.group(1)}print({m.group(2)})"
                    content = '\n'.join(lines)
                    content = re.sub(r'^(\s*)print\s*$', r'\1print()', content, flags=re.MULTILINE)

                    # 4. Additional fixes for Python 3 compatibility and SyntaxWarnings
                    content = content.replace("xrange(", "range(")
                    content = content.replace("    '''\n    print(\"Usage for ApoplastP: \"),", "    print(\"Usage for ApoplastP: \"),")
                    content = content.replace('print(str(err) # will print something like "option -a not recognized")', 'print(str(err)) # will print something like "option -a not recognized"')
                    content = content.replace('re.findall("\\d+.\\d+",', 're.findall(r"\\d+.\\d+",')
                    content = content.replace('re.findall("[-+]?\\d+.\\d+",', 're.findall(r"[-+]?\\d+.\\d+",')
                    content = content.replace("open(output_file, 'wb')", "open(output_file, 'w', encoding='utf-8')")

                    if content != original_content:
                        with open(script_path, 'w') as f: f.write(content)
                except Exception as e: print(f"     [ApoplastP Patch Error]: {e}", flush=True)
        make_executable(apo_scripts)

    # Localizer Check
    loc_dir = os.path.join(abs_bin, "LOCALIZER-1.0.5")
    if os.path.exists(loc_dir):
        loc_scripts = os.path.join(loc_dir, "Scripts")
        
        weka_zip = os.path.join(loc_scripts, "weka-3-6-12.zip")
        weka_target = os.path.join(loc_scripts, "weka-3-6-12")
        weka_jar = os.path.join(weka_target, "weka.jar")
        if os.path.exists(weka_zip) and not os.path.exists(weka_jar):
            print(f"     [Localizer]: Unzipping weka-3-6-12.zip...", flush=True)
            try:
                with zipfile.ZipFile(weka_zip, 'r') as zip_ref: zip_ref.extractall(loc_scripts)
            except Exception as e: print(f"     [Localizer]: Zip extract failed: {e}")
            
        loc_script = os.path.join(loc_scripts, "LOCALIZER.py")
        if os.path.exists(loc_script):
            try:
                with open(loc_script, 'r') as f: content = f.read()
                if "PEPSTATS_PATH = '/usr/bin/'" not in content:
                    content = content.replace("PEPSTATS_PATH = SCRIPT_PATH + '/EMBOSS-6.5.7/emboss/'", "PEPSTATS_PATH = '/usr/bin/'") 
                    with open(loc_script, 'w') as f: f.write(content)
            except: pass
        make_executable(loc_dir)

    # SignalP Check
    sp = os.path.join(abs_bin, 'signalp_env', 'bin', 'signalp6')
    if not os.path.exists(sp):
        warnings.append("SignalP 6.0: Missing. Place your authorized SignalP tarball in bin/.")
    else: make_executable(sp)
        
    if warnings:
        print("⚠️ WARNING: The following tools will fail or be skipped:", flush=True)
        for w in warnings: print(f"   - {w}", flush=True)
        print("Please resolve these permissions or missing files to run the full pipeline.\n", flush=True)
    else:
        print("✅ All primary binaries verified and permissions configured.\n", flush=True)

def ensure_dbcan(bin_dir, env):
    dbcan_cmd = shutil.which('run_dbcan') or os.path.join(os.path.dirname(sys.executable), 'run_dbcan')
    if not os.path.exists(dbcan_cmd): return None
    db_dir = os.path.join(os.path.abspath(bin_dir), "dbcan", "db")
    if not os.path.exists(os.path.join(db_dir, "dbCAN.hmm")):
        print(f"  -> Downloading dbCAN database to {db_dir}...", flush=True)
        os.makedirs(db_dir, exist_ok=True)
        subprocess.run([sys.executable, dbcan_cmd, 'database', '--db_dir', db_dir, '--aws_s3'], check=False, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dbcan_cmd

def try_reuse(reuse_dir, outdir, dirs_to_copy=None, files_to_copy=None):
    if not reuse_dir or not os.path.exists(reuse_dir): return False
    
    if dirs_to_copy:
        for d in dirs_to_copy:
            if not os.path.exists(os.path.join(reuse_dir, d)): return False
    if files_to_copy:
        for f in files_to_copy:
            if not os.path.exists(os.path.join(reuse_dir, f)): return False
            
    if dirs_to_copy:
        for d in dirs_to_copy:
            src = os.path.join(reuse_dir, d)
            dst = os.path.join(outdir, d)
            if os.path.exists(dst): shutil.rmtree(dst)
            elif not os.path.exists(os.path.dirname(dst)): os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
    if files_to_copy:
        for f in files_to_copy:
            src = os.path.join(reuse_dir, f)
            dst = os.path.join(outdir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    return True

def scan_dbcan(fasta_file, outdir, bin_dir, env, reuse_dir=None):
    res = {}
    try:
        dbcan_out = os.path.join(outdir, "dbcan_out")
        if try_reuse(reuse_dir, outdir, dirs_to_copy=["dbcan_out"]):
            print("  -> Reusing previous dbCAN data...", flush=True)
        else:
            print("  -> Running dbCAN...", flush=True)
            dbcan_cmd = ensure_dbcan(bin_dir, env)
            if dbcan_cmd:
                db_dir = os.path.join(os.path.abspath(bin_dir), "dbcan", "db")
                os.makedirs(dbcan_out, exist_ok=True)
                run_logged_command([sys.executable, dbcan_cmd, 'CAZyme_annotation', '--input_raw_data', fasta_file, '--mode', 'protein', '--output_dir', dbcan_out, '--db_dir', db_dir], 'dbcan', os.path.join(outdir, "logs"), env)
                
        ov_file = os.path.join(dbcan_out, "overview.tsv")
        if os.path.exists(ov_file):
            df = pd.read_csv(ov_file, sep='\t')
            for _, row in df.iterrows():
                seq_id = str(row['Gene ID'])
                res[seq_id] = []
                for col in df.columns:
                    if col == 'Gene ID': continue
                    val = str(row[col])
                    if val != 'nan' and val != '-':
                        for x in val.split('+'):
                            m = re.search(r'([A-Za-z0-9_]+)\((\d+)-(\d+)\)', x.strip())
                            if m: res[seq_id].append({'name': m.group(1), 'start': int(m.group(2)), 'end': int(m.group(3))})
    except Exception as e:
        print(f"     [dbCAN Failed]: {e}", flush=True)
    return res

def run_signalp(fasta_file, outdir, bin_dir, env, reuse_dir=None):
    sp_dict = {}
    try:
        sp_out = os.path.join(outdir, "signalp_out")
        if try_reuse(reuse_dir, outdir, dirs_to_copy=["signalp_out"]):
            print("  -> Reusing previous SignalP 6.0 data...", flush=True)
        else:
            print("  -> Running SignalP 6.0...", flush=True)
            sp_bin = os.path.join(os.path.abspath(bin_dir), 'signalp_env', 'bin', 'signalp6')
            if not os.path.exists(sp_bin):
                return sp_dict
            os.makedirs(sp_out, exist_ok=True)
            run_logged_command([sys.executable, sp_bin, '--fasta', fasta_file, '--organism', 'euk', '--output_dir', sp_out, '--format', 'txt', '--mode', 'fast'], 'signalp', os.path.join(outdir, "logs"), env)
            
        res_file = os.path.join(sp_out, "prediction_results.txt")
        if os.path.exists(res_file):
            for line in open(res_file, 'r'):
                if line.startswith("#") or not line.strip(): continue
                parts = re.split(r'\s+', line.strip())
                if len(parts) > 0 and ("SP" in line or "Signal" in line):
                    m = re.search(r'CS pos:\s*(\d+)', line)
                    if m: sp_dict[parts[0]] = int(m.group(1))
    except Exception as e:
        print(f"     [SignalP Failed]: {e}", flush=True)
    return sp_dict

def run_cppsite_blast(fasta_file, cpp_fasta, cpp_csv, outdir, env, reuse_dir=None):
    cpp_dict = {}
    try:
        cpp_dir = os.path.join(outdir, 'cppsite_out')
        if try_reuse(reuse_dir, outdir, dirs_to_copy=["cppsite_out"]):
            print("  -> Reusing previous CPPSite2 data...", flush=True)
        else:
            print("  -> Running CPPSite2 BLAST Search...", flush=True)
            if not os.path.exists(cpp_fasta) or not os.path.exists(cpp_csv): 
                print("     [CPPSite2 Skipped]: Database FASTA or CSV missing", flush=True)
                return cpp_dict
            os.makedirs(cpp_dir, exist_ok=True)
            blast_out = os.path.join(cpp_dir, "cpp_blast.tsv")
            run_logged_command(['blastp', '-query', fasta_file, '-db', cpp_fasta, '-outfmt', '6 qseqid sseqid pident length evalue qstart qend sstart send', '-out', blast_out], 'cppsite', os.path.join(outdir, "logs"), env)
            
        blast_out = os.path.join(cpp_dir, "cpp_blast.tsv")
        if os.path.exists(blast_out):
            df_blast = pd.read_csv(blast_out, sep='\t', names=['qseqid', 'sseqid', 'pident', 'length', 'evalue', 'qstart', 'qend', 'sstart', 'send'])
            for _, row in df_blast.iterrows():
                qid = str(row['qseqid'])
                if qid not in cpp_dict: cpp_dict[qid] = []
                cpp_id = str(row['sseqid']).replace('CPP_', '')
                cpp_dict[qid].append((int(row['qstart']), int(row['qend']), f"CPP_{cpp_id}", float(row['pident'])))
    except Exception as e:
        print(f"     [CPPSite2 Failed]: {e}", flush=True)
    return cpp_dict

def run_deeptmhmm(fasta_file, outdir, bin_dir, env, reuse_dir=None):
    tm_dict = {}
    try:
        tm_outdir = os.path.join(outdir, 'deeptmhmm_out')
        if try_reuse(reuse_dir, outdir, dirs_to_copy=["deeptmhmm_out"]):
            print("  -> Reusing previous DeepTMHMM data...", flush=True)
        else:
            print("  -> Running DeepTMHMM...", flush=True)
            biolib_bin = shutil.which('biolib') or os.path.join(os.path.abspath(bin_dir), 'bin', 'biolib')
            if not os.path.exists(biolib_bin): 
                return tm_dict
            os.makedirs(tm_outdir, exist_ok=True)
            shutil.copy(fasta_file, os.path.join(tm_outdir, "input.fasta"))
            # Create a clean environment for biolib/deeptmhmm to avoid python path conflicts
            tm_env = env.copy()
            if 'PYTHONPATH' in tm_env:
                paths = tm_env['PYTHONPATH'].split(os.pathsep)
                # Filter out signalp_env path which contains a conflicting PyTorch version
                tm_env['PYTHONPATH'] = os.pathsep.join([p for p in paths if 'signalp_env' not in p and p.strip()])

            run_logged_command([sys.executable, biolib_bin, 'run', 'DTU/DeepTMHMM', '--fasta', 'input.fasta'], 'deeptmhmm', os.path.join(outdir, "logs"), tm_env, cwd=tm_outdir)
        gff3_path = os.path.join(tm_outdir, 'biolib_results', 'TMRs.gff3')
        if os.path.exists(gff3_path):
            for line in open(gff3_path, 'r'):
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split('\t')
                if len(parts) >= 5 and ('tmhelix' in parts[2].lower() or 'transmembrane' in parts[2].lower()):
                    seq_id = parts[0].split()[0]
                    if seq_id not in tm_dict: tm_dict[seq_id] = []
                    tm_dict[seq_id].append((int(parts[3]), int(parts[4])))
    except Exception as e:
        print(f"     [DeepTMHMM Failed]: {e}", flush=True)
    return tm_dict

def run_plicat_native(seq_dict, bin_dir, outdir, reuse_dir=None):
    import json
    plicat_data = {}
    plicat_dir = os.path.join(outdir, 'plicat_out')
    
    if try_reuse(reuse_dir, outdir, dirs_to_copy=["plicat_out"]):
        print("  -> Reusing previous PLiCat data...", flush=True)
        res_file = os.path.join(plicat_dir, 'plicat_results.json')
        if os.path.exists(res_file):
            try:
                with open(res_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"     [PLiCat Reuse Failed]: {e}", flush=True)
                
    print("  -> Running PLiCat...", flush=True)
    os.makedirs(plicat_dir, exist_ok=True)
    logs_dir = os.path.join(outdir, 'logs')
    try:
        import torch
        os.environ["HF_HOME"] = os.path.join(os.path.abspath(bin_dir), 'plicat')
        from esm.tokenization import EsmSequenceTokenizer
        sys.path.append(os.path.join(os.getcwd(), 'PLiCat'))
        from plicat_model import PLiCat
        
        with open(os.path.join(logs_dir, 'plicat_stdout.log'), 'w') as f_out, \
             open(os.path.join(logs_dir, 'plicat_stderr.log'), 'w') as f_err, \
             contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
             
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = EsmSequenceTokenizer()
            lipid_dict = {"0": "NotLipidType", "1": "Fatty Acyl", "2": "Prenol Lipid", "3": "Glycerophospholipid", "4": "Sterol Lipid", "5": "Polyketide", "6": "Glycerolipid", "7": "Sphingolipid", "8": "Saccharolipid"}
            model = PLiCat.from_pretrained("Noora68/PLiCat-0.4B").to(device)
            model.eval()
            
            for pid, seq in seq_dict.items():
                if len(seq) < 10: 
                    plicat_data[pid] = {'scores': {}, 'passed': [], 'truncated': False}
                    continue
                truncated = len(seq) > 510
                try:
                    input_ids = torch.tensor(tokenizer.encode(seq[:510])).unsqueeze(0).to(device)
                    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
                    with torch.no_grad(): outputs = model(input_ids, attention_mask)
                    probs = torch.sigmoid(outputs['logits']).squeeze().detach().cpu().numpy()
                    
                    scores, passed = {}, []
                    for i, p in enumerate(probs):
                        cat = lipid_dict[str(i)]
                        scores[cat] = float(p)
                        if p > 0.6 and cat != "NotLipidType": passed.append(cat)
                    plicat_data[pid] = {'scores': scores, 'passed': passed, 'truncated': truncated}
                except: plicat_data[pid] = {'scores': {}, 'passed': [], 'truncated': False}
        with open(os.path.join(plicat_dir, 'plicat_results.json'), 'w') as f:
            json.dump(plicat_data, f)
    except Exception as e:
        print(f"     [PLiCat Failed]: {e}", flush=True)
    return plicat_data

# ================= RIPP TOOLS =================
def run_rippminer(fasta_path, outdir, bin_dir, env, reuse_dir=None):
    res = {'class': {}, 'similarities': []}
    rm_out = os.path.join(outdir, 'rippminer_out')
    
    if try_reuse(reuse_dir, outdir, dirs_to_copy=["rippminer_out"]):
        print("  -> Reusing previous RiPPMiner data...", flush=True)
    else:
        print("  -> Running RiPPMiner...", flush=True)
        rm_bin = os.path.join(os.path.abspath(bin_dir), 'rippminer_standalone')
        os.makedirs(rm_out, exist_ok=True)
        
        if not os.path.exists(rm_bin) or not os.path.exists(os.path.join(rm_bin, 'run_rippminer.pl')): 
            print(f"     [RiPPMiner Skipped]: Executable missing at {rm_bin}.", flush=True)
            return res
        try:
            # Run classification
            run_logged_command(['perl', 'run_rippminer.pl', '-i', os.path.abspath(fasta_path)], 'rippminer_class', os.path.join(outdir, 'logs'), env, cwd=rm_bin)
            class_out = os.path.join(rm_bin, 'prediction.out')
            if not os.path.exists(class_out):
                class_out = os.path.join(rm_bin, 'prediction.output')
            if not os.path.exists(class_out):
                class_out = os.path.join(rm_bin, 'prediction.outputput')
            if not os.path.exists(class_out):
                class_out = os.path.join(rm_bin, 'class_prediction.out')
                
            if os.path.exists(class_out):
                shutil.move(class_out, os.path.join(rm_out, 'prediction.out'))

            # Run Sequence Similarity Search
            run_logged_command(['perl', 'sequence_similarity_search.pl', '-i', os.path.abspath(fasta_path)], 'rippminer_sim', os.path.join(outdir, 'logs'), env, cwd=rm_bin)
            sim_out = os.path.join(rm_bin, 'sequence_similarity.out')
            if os.path.exists(sim_out):
                shutil.move(sim_out, os.path.join(rm_out, 'sequence_similarity.out'))                
            align_out = os.path.join(rm_bin, 'all_alignment')
            if os.path.exists(align_out): shutil.move(align_out, os.path.join(rm_out, 'all_alignment'))
        except Exception as e:
            print(f"     [RiPPMiner Failed]: {e}", flush=True)
            
    # Parse classification predictions
    try:
        pred_out = os.path.join(rm_out, 'prediction.out')
        if os.path.exists(pred_out):
            curr_id = None
            for line in open(pred_out, 'r'):
                line_str = line.strip()
                if line_str.startswith('#INPUT'):
                    parts = line_str.split('\t')
                    if len(parts) >= 3:
                        curr_id = parts[2].strip()
                elif 'Class:' in line_str and curr_id:
                    parts = line_str.split(':')
                    if len(parts) >= 2:
                        res['class'][curr_id] = parts[1].strip()
    except Exception as e:
        print(f"     [RiPPMiner Class Parse Failed]: {e}", flush=True)

    # Parse similarity search results
    try:
        sim_out = os.path.join(rm_out, 'sequence_similarity.out')
        if os.path.exists(sim_out):
            curr_query = None
            for line in open(sim_out, 'r'):
                line_str = line.strip()
                if line_str.startswith('Sequences similar to'):
                    curr_query = line_str.replace('Sequences similar to', '').replace(':', '').strip()
                elif curr_query and line_str and not line_str.startswith('Subject') and not line_str.startswith('Alignment'):
                    parts = [p.strip() for p in line_str.split('\t') if p.strip()]
                    if not parts or len(parts) < 2:
                        parts = [p.strip() for p in line_str.split() if p.strip()]
                    if len(parts) >= 6:
                        res['similarities'].append({
                            'Query': curr_query,
                            'Subject': parts[0],
                            'Identity': parts[1],
                            'Length': parts[2],
                            'Start': parts[3],
                            'End': parts[4],
                            'Evalue': parts[5]
                        })
    except Exception as e:
        print(f"     [RiPPMiner Sim Parse Failed]: {e}", flush=True)
        
    return res

# ================= SUBCELLULAR PARSERS =================

def run_subcellular_tools(fasta_path, bin_dir, outdir, env, records_dict, reuse_analyses=None, reuse_dir=None, deeploc_model='Fast'):
    if reuse_analyses is None: reuse_analyses = []
    res = {'wolf': {}, 'deeploc': {}, 'multiloc': {}, 'apoplastp': {}, 'localizer': {}, 'targetp': {}, 'aiupred': {}}
    abs_bin = os.path.abspath(bin_dir)
    logs_dir = os.path.join(outdir, 'logs')
    
    try:
        tp_dir = os.path.join(outdir, 'targetp_out')
        tp_out_file = os.path.join(tp_dir, 'targetp.out')
        if 'targetp' in reuse_analyses and try_reuse(reuse_dir, outdir, dirs_to_copy=["targetp_out"]):
            print("  -> Reusing previous TargetP data...", flush=True)
        else:
            print("  -> Running TargetP...", flush=True)
            os.makedirs(tp_dir, exist_ok=True)
            tp_bin = os.path.join(abs_bin, 'targetp', 'TargetP-2.0', 'bin', 'targetp')
            if os.path.exists(tp_bin):
                run_logged_command([tp_bin, '-fasta', fasta_path, '-org', 'pl', '-format', 'txt', '-stdout'], 'targetp', logs_dir, env)
                tp_log = os.path.join(logs_dir, 'targetp_stdout.log')
                if os.path.exists(tp_log):
                    shutil.copy(tp_log, tp_out_file)
                
        if os.path.exists(tp_out_file):
            for line in open(tp_out_file):
                if line.startswith('#') or line.startswith('-') or line.startswith('Name') or not line.strip(): continue
                parts = line.split()
                if len(parts) >= 7:
                    scores = {'Chloroplast': float(parts[2]), 'Mitochondria': float(parts[3]), 'Extracellular': float(parts[4]), 'Other': float(parts[5])}
                    res['targetp'][parts[0]] = {'pred': parts[6], 'scores': scores}
    except Exception as e: print(f"     [TargetP Failed]: {e}", flush=True)

    try:
        wolf_dir = os.path.join(outdir, 'wolfpsort_out')
        if 'wolfpsort' in reuse_analyses and try_reuse(reuse_dir, outdir, dirs_to_copy=["wolfpsort_out"]):
            print("  -> Reusing previous WoLF PSORT data...", flush=True)
        else:
            print("  -> Running WoLF PSORT...", flush=True)
            os.makedirs(wolf_dir, exist_ok=True)
            wolf_path = shutil.which("runWolfPsortSummary") or os.path.join(abs_bin, "WoLFPSort", "bin", "runWolfPsortSummary")
            if os.path.exists(wolf_path):
                log_out = os.path.join(logs_dir, 'wolfpsort_stdout.log')
                cmd = f"cat '{fasta_path}' | {wolf_path} plant"
                run_logged_command(cmd, 'wolfpsort', logs_dir, env, shell=True)
                wolf_out_file = os.path.join(wolf_dir, 'wolfpsort_results.txt')
                if os.path.exists(log_out): shutil.copy(log_out, wolf_out_file)
                
        wolf_out_file = os.path.join(wolf_dir, 'wolfpsort_results.txt')
        if os.path.exists(wolf_out_file):
            for line in open(wolf_out_file):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(None, 1)
                    if len(parts) < 2: continue
                    pid = parts[0]
                    rest = parts[1]
                    if rest.startswith('details '):
                        rest = rest[8:]
                            
                    scores = {}
                    primary = 'None'
                    for chunk in rest.split(','):
                        chunk = chunk.strip()
                        if not chunk: continue
                        c_parts = chunk.split()
                        if len(c_parts) >= 2:
                            loc = c_parts[0]
                            try:
                                sc = float(c_parts[1])
                                scores[loc] = sc
                                if primary == 'None': primary = loc
                            except ValueError: pass
                                
                    if scores:
                        res['wolf'][pid] = {'pred': primary, 'scores': scores}
    except Exception as e: print(f"     [WoLF PSORT Failed]: {e}", flush=True)

    try:
        dl_dir = os.path.join(outdir, 'deeploc_out')
        if 'deeploc' in reuse_analyses and try_reuse(reuse_dir, outdir, dirs_to_copy=["deeploc_out"]):
            print("  -> Reusing previous DeepLoc2 data...", flush=True)
        else:
            print("  -> Running DeepLoc2...", flush=True)
            os.makedirs(dl_dir, exist_ok=True)
            dl_bin = shutil.which('deeploc2') or os.path.join(abs_bin, 'deeploc2_package', 'bin', 'deeploc2')
            if not os.path.exists(dl_bin): dl_bin = os.path.join(sys.prefix, 'bin', 'deeploc2')
            
            if os.path.exists(dl_bin) or shutil.which('deeploc2'):
                dl_exec = shutil.which('deeploc2') or dl_bin
                # Create a clean environment for deeploc2 to avoid python path conflicts from other tools (e.g., signalp)
                dl_env = env.copy()
                if 'PYTHONPATH' in dl_env:
                    paths = dl_env['PYTHONPATH'].split(os.pathsep)
                    # Filter out signalp_env path which contains a conflicting PyTorch version
                    dl_env['PYTHONPATH'] = os.pathsep.join([p for p in paths if 'signalp_env' not in p and p.strip()])

                dl_fasta = os.path.join(dl_dir, "deeploc_input.fasta")
                valid_dl_ids = set()
                with open(dl_fasta, 'w') as f_dl:
                    for rec in SeqIO.parse(fasta_path, "fasta"):
                        full_seq = records_dict.get(rec.id, str(rec.seq)) if records_dict is not None else str(rec.seq)
                        if len(full_seq) >= 15:
                            f_dl.write(f">{rec.id}\n{full_seq}\n")
                            valid_dl_ids.add(rec.id)
                
                if valid_dl_ids:
                    run_logged_command([dl_exec, '-m', deeploc_model, '-f', os.path.abspath(dl_fasta), '-o', os.path.abspath(dl_dir)], 'deeploc', logs_dir, dl_env)
        import glob
        res_csvs = glob.glob(os.path.join(dl_dir, 'results*.csv'))
        if res_csvs:
            res_csvs.sort(key=os.path.getmtime, reverse=True)
            res_csv = res_csvs[0]
        else:
            res_csv = os.path.join(dl_dir, 'results.csv')
        if os.path.exists(res_csv):
            df = pd.read_csv(res_csv)
            for _, row in df.iterrows():
                pid = str(row.get('Protein_ID', row.iloc[0]))
                pred = str(row.get('Localizations', 'None'))
                scores = {col: float(row[col]) for col in df.columns if col not in ['Protein_ID', 'Localizations', 'Signals', 'Membrane types'] and pd.api.types.is_numeric_dtype(df[col])}
                signals = str(row.get('Signals', 'None')) if 'Signals' in df.columns else str(row.iloc[2]) if len(row) > 2 else 'None'
                res['deeploc'][pid] = {'pred': pred, 'scores': scores, 'signals': signals}
    except Exception as e: print(f"     [DeepLoc2 Failed]: {e}", flush=True)

    try:
        ml_dir = os.path.join(outdir, 'multiloc_out')
        if 'multiloc' in reuse_analyses and try_reuse(reuse_dir, outdir, dirs_to_copy=["multiloc_out"]):
            print("  -> Reusing previous MultiLoc2 data...", flush=True)
        else:
            print("  -> Running MultiLoc2...", flush=True)
            os.makedirs(ml_dir, exist_ok=True)
            ml2_path = os.path.join(abs_bin, "MultiLoc2-master", "MultiLoc2", "src", "multiloc2_prediction.py")
            if not os.path.exists(ml2_path): ml2_path = os.path.join(abs_bin, "MultiLoc2-master", "MultiLoc2", "src", "multiloc2.py")
            if not os.path.exists(ml2_path): ml2_path = os.path.join(abs_bin, "MultiLoc2-master", "src", "multiloc2_prediction.py")
            if os.path.exists(ml2_path):
                tmp_out = os.path.join(ml_dir, "ml2_results.txt")
                py2_exec = shutil.which('python2') or shutil.which('python2.7') or 'python2'
                ml2_cwd = os.path.dirname(os.path.dirname(ml2_path))
                
                ml_fasta = os.path.join(ml_dir, "multiloc_input.fasta")
                valid_ml_ids = set()
                with open(ml_fasta, 'w') as f_ml:
                    for rec in SeqIO.parse(fasta_path, "fasta"):
                        if len(rec.seq) >= 15:
                            f_ml.write(f">{rec.id}\n{rec.seq}\n")
                            valid_ml_ids.add(rec.id)
                            
                if valid_ml_ids:
                    # Clean the environment for Python 2 to prevent Python 3 modules from causing silent crashes
                    ml2_env = env.copy()
                    if 'PYTHONPATH' in ml2_env:
                        del ml2_env['PYTHONPATH']
                    run_logged_command([py2_exec, ml2_path, f'-fasta={os.path.abspath(ml_fasta)}', '-origin=plant', '-predictor=HighRes', f'-result={os.path.abspath(tmp_out)}'], 'multiloc2', logs_dir, ml2_env, cwd=ml2_cwd)
                                
        tmp_out = os.path.join(ml_dir, "ml2_results.txt")
        if os.path.exists(tmp_out):
            ml_map = {
                'cytoplasmic': 'Cytoplasm',
                'nuclear': 'Nucleus',
                'mitochondrial': 'Mitochondria',
                'chloroplast': 'Chloroplast',
                'er': 'Endoplasmic Reticulum',
                'golgi': 'Golgi',
                'golgi apparatus': 'Golgi',
                'peroxisomal': 'Peroxisome',
                'plasma membrane': 'Plasma Membrane',
                'extracellular': 'Extracellular',
                'vacuolar': 'Vacuole',
                'lysosomal': 'Lysosome',
                'cy': 'Cytoplasm',
                'nu': 'Nucleus',
                'mi': 'Mitochondria',
                'ch': 'Chloroplast',
                'go': 'Golgi',
                'pe': 'Peroxisome',
                'pm': 'Plasma Membrane',
                'ex': 'Extracellular',
                'ly': 'Lysosome',
                'va': 'Vacuole'
            }
            is_new_format = False
            with open(tmp_out, 'r', encoding='utf-8') as f:
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    if "MultiLoc2 Prediction Result" in line or ":" in line:
                        is_new_format = True
                        break
            
            if is_new_format:
                with open(tmp_out, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("MultiLoc2 Prediction Result") or line.startswith("predictor =") or line.startswith("origin ="):
                            continue
                        parts = line.split('\t')
                        if len(parts) < 2:
                            continue
                        pid = parts[0].strip()
                        scores = {}
                        for item in parts[1:]:
                            if ':' in item:
                                loc, score_str = item.split(':', 1)
                                loc = loc.strip()
                                try:
                                    val = float(score_str.strip())
                                    std_loc = ml_map.get(loc.lower(), loc)
                                    scores[std_loc] = val
                                except ValueError:
                                    pass
                        if scores:
                            pred = max(scores, key=scores.get)
                            res['multiloc'][pid] = {'pred': pred, 'scores': scores}
            else:
                df = pd.read_csv(tmp_out, sep='\t')
                for _, row in df.iterrows():
                    pid = str(row.iloc[0])
                    pred_raw = str(row.iloc[1]) if len(row) > 1 else 'None'
                    pred = ml_map.get(pred_raw.lower(), pred_raw)
                    scores = {}
                    for col in df.columns[2:]:
                        if col.lower() in ml_map and pd.notna(row[col]):
                            scores[ml_map[col.lower()]] = float(row[col])
                    if not scores:
                        scores[pred] = 1.0
                    res['multiloc'][pid] = {'pred': pred, 'scores': scores}
    except Exception as e: print(f"     [MultiLoc2 Failed]: {e}", flush=True)

    try:
        apo_dir = os.path.join(outdir, 'apoplastp_out')
        if 'apoplastp' in reuse_analyses and try_reuse(reuse_dir, outdir, dirs_to_copy=["apoplastp_out"]):
            print("  -> Reusing previous ApoplastP data...", flush=True)
        else:
            print("  -> Running ApoplastP...", flush=True)
            os.makedirs(apo_dir, exist_ok=True)
            apo_path = os.path.join(abs_bin, "ApoplastP_1.0.1", "Scripts", "ApoplastP.py")
            if not os.path.exists(apo_path): apo_path = os.path.join(abs_bin, "ApoplastP_1.0.1", "ApoplastP.py")
            if os.path.exists(apo_path):
                tmp_out = os.path.join(apo_dir, "apoplastp_results.csv")
                run_logged_command([sys.executable, apo_path, '-i', fasta_path, '-o', tmp_out], 'apoplastp', logs_dir, env)
                
        tmp_out = os.path.join(apo_dir, "apoplastp_results.csv")
        if os.path.exists(tmp_out):
            with open(tmp_out, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        raw_id = parts[0].strip()
                        pred = parts[1].strip()
                        prob = 1.0
                        if len(parts) >= 3:
                            try:
                                prob = float(parts[2].strip())
                            except ValueError:
                                pass
                        pid = raw_id.split()[0]
                        res['apoplastp'][pid] = {'pred': pred, 'scores': {pred: prob}}
    except Exception as e: print(f"     [ApoplastP Failed]: {e}", flush=True)

    try:
        loc_dir = os.path.join(outdir, 'localizer_out')
        if 'localizer' in reuse_analyses and try_reuse(reuse_dir, outdir, dirs_to_copy=["localizer_out"]):
            print("  -> Reusing previous LOCALIZER data...", flush=True)
        else:
            print("  -> Running LOCALIZER...", flush=True)
            os.makedirs(loc_dir, exist_ok=True)
            loc_base = os.path.join(abs_bin, "LOCALIZER-1.0.5")
            loc_path = os.path.join(loc_base, "Scripts", "LOCALIZER.py")
            if not os.path.exists(loc_path): loc_path = os.path.join(loc_base, "localizer.py")
            if os.path.exists(loc_path):
                py2_exec = shutil.which('python2') or 'python2'
                run_logged_command([py2_exec, loc_path, '-e', '-M', '-i', fasta_path, '-o', loc_dir], 'localizer', logs_dir, env)                

        res_txt = os.path.join(loc_dir, 'Results.txt')
        if os.path.exists(res_txt):
            with open(res_txt, 'r') as f: lines = f.readlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.strip('\r\n').split('\t')
                    if len(parts) < 4: continue
                    # Clean ID from any appended metadata in the first column
                    pid = parts[0].split()[0]
                    locs, scores, seqs = [], {}, []
                    for idx, c in enumerate(['Chloroplast', 'Mitochondria', 'Nucleus']):
                        val = parts[idx+1].strip()
                        if val.startswith('Y'):
                            locs.append(c)
                            scores[c] = 1.0
                            m = re.search(r'Y\s*\((.*?)\)', val)
                            if m:
                                inner = m.group(1).strip()
                                m2 = re.match(r'([\d\.]+)\s*[\|:]\s*(\d+)-(\d+)', inner)
                                tag = 'cTP' if c == 'Chloroplast' else ('mTP' if c == 'Mitochondria' else 'NLS')
                                if m2:
                                    prob, start, end = m2.groups()
                                    seqs.append((tag, {'prob': float(prob), 'start': int(start), 'end': int(end)}))
                                else:
                                    for sub_seq in inner.split(','):
                                        sub_seq = sub_seq.strip()
                                        if sub_seq:
                                            seqs.append((tag, {'sequence': sub_seq}))
                        else:
                            scores[c] = 0.0
                    res['localizer'][pid] = {'pred': ",".join(locs) if locs else "None", 'scores': scores, 'sequences': seqs}
    except Exception as e: print(f"     [LOCALIZER Failed]: {e}", flush=True)

    try:
        ai_out_dir = os.path.join(outdir, 'aiupred_out')
        if 'aiupred' in reuse_analyses and try_reuse(reuse_dir, outdir, dirs_to_copy=['aiupred_out']):
            print("  -> Reusing previous AIUPred data...", flush=True)
        else:
            print("  -> Running AIUPred...", flush=True)
            ai_dir = os.path.join(abs_bin, 'aiupred')
            ai_path = shutil.which("aiupred") or os.path.join(ai_dir, 'bin', 'aiupred')

            # Override paths explicitly if targeted install
            local_env = env.copy()
            local_env['PYTHONPATH'] = ai_dir + os.pathsep + env.get('PYTHONPATH', '')
            local_env['PATH'] = os.path.join(ai_dir, 'bin') + os.pathsep + env.get('PATH', '')

            if ai_path and os.path.exists(ai_path):
                os.makedirs(ai_out_dir, exist_ok=True)
                ai_fasta = os.path.join(ai_out_dir, "aiupred_input.fasta")
                with open(ai_fasta, 'w') as f_ai:
                    for rec in SeqIO.parse(fasta_path, "fasta"):
                        full_seq = records_dict.get(rec.id, str(rec.seq)) if records_dict is not None else str(rec.seq)
                        if len(full_seq) >= 15:
                            f_ai.write(f">{rec.id}\n{full_seq}\n")
                run_logged_command([ai_path, '-i', os.path.abspath(ai_fasta), '-o', os.path.join(ai_out_dir, 'aiupred.out'), '--force-cpu', '-b', '-l'], 'aiupred', logs_dir, local_env)
                if os.path.exists(ai_fasta):
                    try: os.remove(ai_fasta)
                    except: pass

        tmp_ai_out = os.path.join(ai_out_dir, 'aiupred.out')
        if not os.path.exists(tmp_ai_out):
            tmp_ai_out = os.path.join(ai_out_dir, 'iupred2.result')
        if os.path.exists(tmp_ai_out):
            curr_id = None
            scores = {'disorder': [], 'linker': [], 'binding': []}
            for line in open(tmp_ai_out):
                if line.startswith('#>'):
                    if curr_id: res['aiupred'][curr_id] = scores
                    curr_id = line[2:].strip().split()[0]
                    scores = {'disorder': [], 'linker': [], 'binding': []}
                elif line.startswith('>'):
                    if curr_id: res['aiupred'][curr_id] = scores
                    curr_id = line[1:].strip().split()[0]
                    scores = {'disorder': [], 'linker': [], 'binding': []}
                elif line.strip() and not (line.startswith('#') and not line.startswith('#>')):
                    parts = line.split('\t') if '\t' in line else line.split()
                    if len(parts) >= 5:
                        try:
                            scores['disorder'].append(float(parts[2]))
                            scores['binding'].append(float(parts[3]))
                            scores['linker'].append(float(parts[4]))
                        except ValueError: pass
            if curr_id: res['aiupred'][curr_id] = scores
    except Exception as e: print(f"     [AIUPred Failed]: {e}", flush=True)

    return res

def generate_gff3(gff_data, out_path):
    seen = set()
    with open(out_path, 'w') as f:
        f.write("##gff-version 3\n")
        for rec in gff_data:
            if rec['start'] is None or rec['end'] is None or math.isnan(rec['start']) or math.isnan(rec['end']): continue
            s_id = sanitize_filename(rec['seqid'])
            tup = (s_id, rec['type'], int(rec['start']), int(rec['end']), rec.get('note',''))
            if tup not in seen:
                seen.add(tup)
                attrs = f"ID={rec['id']}"
                if rec.get('note'): attrs += f";{rec['note']}"
                f.write(f"{s_id}\tEFtranslocator\t{rec['type']}\t{int(rec['start'])}\t{int(rec['end'])}\t.\t+\t.\t{attrs}\n")

def get_cargo_regions(seq_len, exclusions):
    all_pos = set(range(1, seq_len + 1))
    for s, e in exclusions: all_pos -= set(range(int(s), int(e) + 1))
    if not all_pos: return []
    sorted_pos = sorted(list(all_pos))
    cargo_blocks, curr_s, curr_e = [], sorted_pos[0], sorted_pos[0]
    for p in sorted_pos[1:]:
        if p == curr_e + 1: curr_e = p
        else: cargo_blocks.append((curr_s, curr_e)); curr_s = curr_e = p
    cargo_blocks.append((curr_s, curr_e))
    return cargo_blocks

def get_overlap_level(start, end, occupied_levels):
    curr_lvl = 1
    while True:
        overlap_pos = any(not (end < s or start > e) for s, e in occupied_levels.get(curr_lvl, []))
        if not overlap_pos:
            occupied_levels.setdefault(curr_lvl, []).append((start, end))
            return curr_lvl
        overlap_neg = any(not (end < s or start > e) for s, e in occupied_levels.get(-curr_lvl, []))
        if not overlap_neg:
            occupied_levels.setdefault(-curr_lvl, []).append((start, end))
            return -curr_lvl
        curr_lvl += 1

def merge_cpp(cpp_list):
    if not cpp_list: return []
    cpp_list.sort(key=lambda x: x[0])
    merged = []
    curr = list(cpp_list[0])
    curr_ids = {curr[2]}
    for nxt in cpp_list[1:]:
        if nxt[0] <= curr[1]:
            curr[1] = max(curr[1], nxt[1])
            curr_ids.add(nxt[2])
        else:
            merged.append((curr[0], curr[1], " + ".join(sorted(curr_ids))))
            curr = list(nxt)
            curr_ids = {curr[2]}
    merged.append((curr[0], curr[1], " + ".join(sorted(curr_ids))))
    return merged

def merge_dbcan(dbcan_list):
    if not dbcan_list: return []
    dbcan_list.sort(key=lambda x: x['start'])
    merged = []
    curr = dbcan_list[0].copy()
    curr_names = {curr['name']}
    for nxt in dbcan_list[1:]:
        if nxt['start'] <= curr['end']:
            curr['end'] = max(curr['end'], nxt['end'])
            curr_names.add(nxt['name'])
        else:
            curr['name'] = " + ".join(sorted(curr_names))
            merged.append(curr)
            curr = nxt.copy()
            curr_names = {curr['name']}
    curr['name'] = " + ".join(sorted(curr_names))
    merged.append(curr)
    return merged

def parse_external_gff3(gff3_path):
    features = defaultdict(list)
    if not gff3_path or not os.path.exists(gff3_path):
        return features
    try:
        with open(gff3_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 9:
                    seqid = parts[0].split()[0]
                    ftype = parts[2]
                    try:
                        start = int(parts[3])
                        end = int(parts[4])
                    except ValueError:
                        continue
                    attributes_str = parts[8]
                    attrs = {}
                    for item in attributes_str.split(';'):
                        if '=' in item:
                            k, v = item.split('=', 1)
                            attrs[k.strip().lower()] = v.strip()
                    name = attrs.get('name') or attrs.get('id') or ftype
                    note = attrs.get('note') or ''
                    features[seqid].append({
                        'type': ftype,
                        'start': start,
                        'end': end,
                        'name': name,
                        'note': note
                    })
    except Exception as e:
        print(f"Error parsing input GFF3 file {gff3_path}: {e}", flush=True)
    return features

def parse_external_bed(bed_path):
    features = defaultdict(list)
    if not bed_path or not os.path.exists(bed_path):
        return features
    try:
        with open(bed_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    parts = line.strip().split()
                if len(parts) >= 3:
                    seqid = parts[0].split()[0]
                    try:
                        start = int(parts[1]) + 1
                        end = int(parts[2])
                    except ValueError:
                        continue
                    name = parts[3] if len(parts) > 3 else "External_Feature"
                    features[seqid].append({
                        'type': 'External_Feature',
                        'start': start,
                        'end': end,
                        'name': name,
                        'note': ''
                    })
    except Exception as e:
        print(f"Error parsing input BED file {bed_path}: {e}", flush=True)
    return features

# ================= MAIN PIPELINE =================
def process_proteins(args):
    prev_dir = None
    reuse_analyses = []
    if args.prev_data:
        if ':' in args.prev_data:
            parts = args.prev_data.split(':', 1)
            prev_dir = parts[0]
            reuse_analyses = [x.strip().lower() for x in parts[1].split(',')]
        else:
            prev_dir = args.prev_data
            reuse_analyses = ['all']
            
    if 'all' in reuse_analyses:
        reuse_analyses = ['signalp', 'plicat', 'cppsite', 'localizer', 'apoplastp', 'deeptmhmm', 'deeploc', 'multiloc', 'wolfpsort', 'targetp', 'dbcan', 'ripp', 'rippminer', 'aiupred', 'auipred']
    if 'ripp' in reuse_analyses and 'rippminer' not in reuse_analyses: reuse_analyses.append('rippminer')
    if 'auipred' in reuse_analyses and 'aiupred' not in reuse_analyses: reuse_analyses.append('aiupred')
    if prev_dir and not os.path.exists(prev_dir):
        print(f"Warning: --prev-data directory '{prev_dir}' not found. Ignoring.", flush=True)
        prev_dir = None

    base_name = os.path.splitext(os.path.basename(args.i))[0]
    outdir = os.path.abspath(f"{base_name}_EFout")
    counter = 1
    while os.path.exists(outdir):
        outdir = os.path.abspath(f"{base_name}_EFout_{counter}")
        counter += 1
    os.makedirs(outdir)
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logs_dir = os.path.join(outdir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_csv = os.path.join(outdir, f"{base_name}_results.csv")
    out_gff = os.path.join(outdir, f"{base_name}_features.gff3")
    kep_dir = os.path.join(outdir, 'kep_ripp_repeats')
    os.makedirs(kep_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, 'EFtranslocator_main.log')
    sys.stdout = TeeLogger(log_file, sys.stdout)
    sys.stderr = TeeLogger(log_file, sys.stderr)
    
    print(f"\n=======================================================", flush=True)
    print(f"Initializing EFtranslocator...", flush=True)
    print(f"Output Directory: {outdir}", flush=True)
    print(f"KEP Repeat Identity Threshold: {args.kep_identity * 100}%", flush=True)
    print(f"Max Prodomain Length: {args.max_prodomain_percent * 100}% of sequence", flush=True)
    print(f"=======================================================\n", flush=True)

    precheck_binaries(args.bin_dir, outdir)

    poly_model = generate_polynomial_model()
    rf_model = get_random_forest_model()

    env = os.environ.copy()
    abs_bin_dir = os.path.abspath(args.bin_dir)
    env['PATH'] = os.path.join(abs_bin_dir, 'svm') + os.pathsep + os.path.join(abs_bin_dir, 'rippminer_standalone', 'scripts') + os.pathsep + os.path.join(abs_bin_dir, 'diamond') + os.pathsep + os.path.join(abs_bin_dir, 'signalp_env', 'bin') + os.pathsep + os.path.join(abs_bin_dir, 'deeploc_env', 'bin') + os.pathsep + os.path.join(abs_bin_dir, 'targetp', 'TargetP-2.0', 'bin') + os.pathsep + abs_bin_dir + os.pathsep + os.path.join(abs_bin_dir, 'bin') + os.pathsep + os.path.dirname(sys.executable) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = os.path.join(abs_bin_dir, 'signalp_env') + os.pathsep + os.path.join(abs_bin_dir, 'deeploc_env') + os.pathsep + abs_bin_dir + os.pathsep + env.get('PYTHONPATH', '')

    seen_ids = {}
    records = []
    records_dict = {}
    for rec in SeqIO.parse(args.i, "fasta"):
        original_id = rec.id
        if original_id in seen_ids:
            seen_ids[original_id] += 1
            unique_id = f"{original_id}_dup{seen_ids[original_id]}"
        else:
            seen_ids[original_id] = 0
            unique_id = original_id
            
        desc = rec.description[len(original_id):].strip() if rec.description.startswith(original_id) else rec.description
        records.append(DummyRecord(unique_id, str(rec.seq), desc))
        records_dict[unique_id] = str(rec.seq)

    tmp_fasta = os.path.join(outdir, "temp_scan.fasta")
    with open(tmp_fasta, 'w') as f:
        for r in records: f.write(f">{r.id}\n{r.seq}\n")
        
    print(f"Executing external sequence predictors on {len(records)} sequences... [0%]", flush=True)
    
    sp_predictions = run_signalp(tmp_fasta, outdir, args.bin_dir, env, reuse_dir=prev_dir if 'signalp' in reuse_analyses else None)
    tmhmm_predictions = run_deeptmhmm(tmp_fasta, outdir, args.bin_dir, env, reuse_dir=prev_dir if 'deeptmhmm' in reuse_analyses else None)
    dbcan_predictions = scan_dbcan(tmp_fasta, outdir, args.bin_dir, env, reuse_dir=prev_dir if 'dbcan' in reuse_analyses else None)
    dynamic_domains = scan_domains(tmp_fasta, records, "data", env, outdir)
    rippminer_res = run_rippminer(tmp_fasta, outdir, args.bin_dir, env, reuse_dir=prev_dir if 'rippminer' in reuse_analyses else None)    
    if os.path.exists(tmp_fasta): os.remove(tmp_fasta)
        
    protease_motifs = [m.replace('!', '') for m in args.protease_motifs.split(',')]
    
    final_products_dict = {}
    record_cleavage_data = {}
    kep_data_global = {}
    
    for record in records:
        pid, full_seq = record.id, record.seq
        seq_len = len(full_seq)
        sp_len = sp_predictions.get(pid, 0)
        mature_seq = full_seq[sp_len:]
        secreted = "yes" if sp_len > 0 else "no"
        max_pro_len = int(seq_len * args.max_prodomain_percent)
        
        m = re.findall(r'[\[\(]([^\]\)]+)[\]\)]', record.description)
        seq_type = " + ".join(sorted([x.strip().upper() for x in m])) if m else "UNKNOWN"
        is_pexel_translocator = "PEXEL" in seq_type
        
        mature_cuts = []
        for dom in dynamic_domains.get(pid, []):
            if dom['category'].lower() == 'proteolytic processing' and dom['cut_pos'] is not None:
                if dom['cut_pos'] > sp_len:
                    mature_cuts.append((dom['cut_pos'] - sp_len, dom['matching_seq'], dom['name']))
        
        if secreted == "yes":
            for motif in protease_motifs:
                motif_clean = motif.replace('X', '.').replace('x', '.')
                cleavage_mode = '/' in motif_clean
                safe_regex = f"({motif_clean.split('/')[0]})({motif_clean.split('/')[1]})" if cleavage_mode else f"({motif_clean})"
                for match in re.finditer(safe_regex, mature_seq, re.IGNORECASE):
                    c_pos = match.end(1) if cleavage_mode else match.end()
                    if "EDQ" in motif.upper(): nm_tag = "PEXEL"
                    elif "LIVM" in motif.upper() or "K" in motif.upper() or "R" in motif.upper(): nm_tag = "Kex2"
                    else: nm_tag = "Protease"
                    mature_cuts.append((c_pos, match.group(0), nm_tag))
                
        mature_cuts.sort(key=lambda x: x[0])
        
        kep_repeats = find_kep_repeats(mature_seq, sp_len, mature_cuts, args.kep_identity)
        kep_data_global[pid] = kep_repeats
        
        if kep_repeats:
            primary_cut = None
            pro_len = 0
        else:
            if is_pexel_translocator:
                primary_cut = next((c for c in mature_cuts if c[2] == "PEXEL" and 10 <= c[0] <= 40 and c[0] <= max_pro_len), None)
            else:
                kex2_cuts = [c for c in mature_cuts if c[2] == "Kex2" and c[0] <= 90 and c[0] <= max_pro_len]
                if kex2_cuts:
                    primary_cut = next((c for c in kex2_cuts if 10 <= c[0]), kex2_cuts[0]) if kex2_cuts[0][0] < 10 else kex2_cuts[0]
                else:
                    primary_cut = None
            pro_len = primary_cut[0] if primary_cut else 0
            
        final_products_dict[pid] = mature_seq[pro_len:]
        record_cleavage_data[pid] = {'sp_len': sp_len, 'mature_cuts': mature_cuts, 'pro_len': pro_len, 'primary_cut': primary_cut, 'seq_type': seq_type, 'is_kep': len(kep_repeats) > 0}

    print("Executing sub-cellular localisation predictors... [50%]", flush=True)
    final_fasta = os.path.join(outdir, "final_products.fasta")
    with open(final_fasta, 'w') as f:
        for pid, seq in final_products_dict.items():
            cd = record_cleavage_data[pid]
            header_add = f" SP={cd['sp_len']} Pro={cd['pro_len']}"
            f.write(f">{pid}{header_add}\n{seq}\n")
            
    subcell = run_subcellular_tools(final_fasta, args.bin_dir, outdir, env, records_dict, reuse_analyses=reuse_analyses, reuse_dir=prev_dir, deeploc_model=args.deeploc_model)
    cpp_predictions = run_cppsite_blast(final_fasta, args.cppsite_fasta, args.cppsite_csv, outdir, env, reuse_dir=prev_dir if 'cppsite' in reuse_analyses else None)
    plicat_data = run_plicat_native(final_products_dict, args.bin_dir, outdir, reuse_dir=prev_dir if 'plicat' in reuse_analyses else None)
        
    results, plot_data, gff_records = [], [], []
    summary_stats = defaultdict(lambda: {'total': 0, 'matches': 0, 'reasons': defaultdict(int)})
    upset_tl_evidence, upset_subcell_dest, upset_categories = [], [], []
    valid_lipid_cats = ["canonical phospholipid", "curvature-sensing", "lipid transfer", "outer membrane & bacterial", "signaling & scaffolding"]
    intermediate_keps = []
    seen_proteins_csv = set()

    gff3_features = parse_external_gff3(args.gff3) if args.gff3 else defaultdict(list)
    bed_features = parse_external_bed(args.bed) if args.bed else defaultdict(list)

    print("Compiling metadata... [80%]", flush=True)
    for record in records:
        try:
            pid, full_seq = record.id, record.seq
            
            if '_dup' in pid or pid in seen_proteins_csv: continue
            seen_proteins_csv.add(pid)
            
            seq_len = len(full_seq)
            cdata = record_cleavage_data[pid]
            keps = kep_data_global[pid]
            sp_len, mature_cuts, pro_len, primary_cut, seq_type = cdata['sp_len'], cdata['mature_cuts'], cdata['pro_len'], cdata['primary_cut'], cdata['seq_type']
            full_cuts = [(pos + sp_len, seq, nm) for pos, seq, nm in mature_cuts]
            final_seq = final_products_dict[pid]
            secreted = "yes" if sp_len > 0 else "no"
            
            coord_offset = sp_len + pro_len
            cpp_hits_final = cpp_predictions.get(pid, [])
            adj_cpp_hits = [(s - 1 + coord_offset, e + coord_offset, c_id, pident) for s, e, c_id, pident in cpp_hits_final]
            dbcan_hits = dbcan_predictions.get(pid, [])
            tm_domains = tmhmm_predictions.get(pid, [])
            for i, (s, e) in enumerate(tm_domains):
                if s and e: gff_records.append({'seqid': pid, 'type': 'Transmembrane', 'start': s, 'end': e, 'id': f"{pid}_TM_{i}"})
            
            # Add external features if present
            ext_feats = []
            if args.gff3:
                ext_feats.extend(gff3_features.get(pid, []))
            if args.bed:
                ext_feats.extend(bed_features.get(pid, []))
            
            for idx, feat in enumerate(ext_feats):
                gff_records.append({
                    'seqid': pid,
                    'type': feat['type'].replace(' ', '_'),
                    'start': feat['start'],
                    'end': feat['end'],
                    'id': f"{pid}_Ext_{idx}",
                    'note': f"Name={feat['name']}" + (f";Note={feat['note']}" if feat['note'] else "")
                })
            
            full_kd_profile = calculate_kd_profile(full_seq, args.translocator_windowsize)
            exclusions_for_cargo = []
            
            if sp_len > 0:
                exclusions_for_cargo.append((1, sp_len))
                gff_records.append({'seqid': pid, 'type': 'SignalPeptide', 'start': 1, 'end': sp_len, 'id': f"{pid}_SP"})
            if pro_len > 0 and not cdata['is_kep']:
                exclusions_for_cargo.append((sp_len + 1, sp_len + pro_len))
                gff_records.append({'seqid': pid, 'type': 'Prodomain', 'start': sp_len + 1, 'end': sp_len + pro_len, 'id': f"{pid}_Prodomain"})
                
            drek_hits = find_drek_motifs(full_seq)
            for idx, (s, e, seq) in enumerate(drek_hits): 
                if s is not None and e is not None: gff_records.append({'seqid': pid, 'type': 'DREK', 'start': s+1, 'end': e, 'id': f"{pid}_DREK_{idx}", 'note': f"Note={seq}"})
            
            # Map top RiPPMiner Hit
            top_ripp_hit = ""
            hits_for_pid = [h for h in rippminer_res['similarities'] if h.get('Query') == pid]
            if hits_for_pid:
                top_ripp_hit = hits_for_pid[0]['Subject'].split('|')[-1] 
            
            for k in keps:
                intermediate_keps.append({'Protein_ID': pid, 'KEP_Regex': k['regex'], 'Frequency': k['freq'], 'Coordinates': ";".join([f"{x['start']}-{x['end']}" for x in k['instances']])})
                for idx, inst in enumerate(k['instances']):
                    gff_records.append({'seqid': pid, 'type': 'Putative_KEP_RIPP', 'start': inst['start'], 'end': inst['end'], 'id': f"{pid}_KEP_{idx}", 'note': f"Note={inst['seq']};Regex={k['regex']}"})
            
            for idx, (c_pos, seq, nm_tag) in enumerate(full_cuts):
                if c_pos is not None: gff_records.append({'seqid': pid, 'type': f"{nm_tag}_Cleavage", 'start': c_pos, 'end': c_pos, 'id': f"{pid}_{nm_tag}_{idx}", 'note': f"Note={seq}"})
            
            rgd_hits = [(m.start(), m.end()) for m in re.finditer(r'RGD', full_seq)]
            anchor_intervals = [(s, e) for s, e, _ in drek_hits] + rgd_hits
            
            for dom in dynamic_domains.get(pid, []):
                if dom['category'].lower() == 'proteolytic processing': continue
                if dom['start'] is None or dom['end'] is None: continue
                note = f"Note={dom.get('matching_seq','')}"
                if dom['evalue']: note += f";Evalue={dom['evalue']}"
                gff_records.append({'seqid': pid, 'type': dom['name'].replace(' ', '_'), 'start': dom['start'], 'end': dom['end'], 'id': f"{pid}_{dom['name']}", 'note': note})

            for idx, (s, e, c_id, _) in enumerate(adj_cpp_hits): 
                if s is not None and e is not None: gff_records.append({'seqid': pid, 'type': 'CPPSite', 'start': s+1, 'end': e, 'id': f"{pid}_{c_id}_{idx}", 'note': f"Note={full_seq[s:e]}"})

            for idx, d in enumerate(dbcan_hits):
                if d['start'] is not None and d['end'] is not None: gff_records.append({'seqid': pid, 'type': 'CAZyme', 'start': d['start'], 'end': d['end'], 'id': f"{pid}_{d['name']}_{idx}"})

            # Incorporate DeepLoc2 Signals as Plot features automatically
            dl_sig = subcell.get('deeploc', {}).get(pid, {}).get('signals', '')
            if dl_sig and dl_sig not in ['None', 'nan', '-']:
                dynamic_domains.setdefault(pid, []).append({
                    'category': 'Sorting Signal',
                    'name': f"DeepLoc2: {dl_sig}",
                    'start': 1,
                    'end': min(40, seq_len), 
                    'cut_pos': None, 'type': 'signal', 'raw': dl_sig, 'evalue': None,
                    'pfam_id': '', 'motif': '', 'description': 'DeepLoc2 Predicted Signal', 'notes': '', 'matching_seq': ''
                })

            loc_transits = []
            if 'localizer' in subcell and pid in subcell['localizer']:
                for tag, dat in subcell['localizer'][pid].get('sequences', []):
                    ts, te = None, None
                    if 'start' in dat and 'end' in dat:
                        ts = dat['start'] + coord_offset
                        te = dat['end'] + coord_offset
                    elif 'sequence' in dat:
                        seq = dat['sequence']
                        idx_match = full_seq.find(seq)
                        if idx_match != -1:
                            ts = idx_match + 1
                            te = idx_match + len(seq)
                    if ts is not None and te is not None:
                        gff_records.append({'seqid': pid, 'type': tag.replace(' ', '_'), 'start': ts, 'end': te, 'id': f"{pid}_{tag}", 'note': f"Note={dat.get('sequence', 'Localizer Signal')}"})
                        loc_transits.append({'start': ts, 'end': te, 'name': tag})

            translocator_windows, best_poly_r2, best_dist = scan_translocator(full_kd_profile, full_seq, poly_model, rf_model, args.translocator_windowsize, anchor_intervals, sp_len)
            for idx, w in enumerate(translocator_windows):
                exclusions_for_cargo.append((w['start']+1, w['end']))
                if w['start'] is not None and w['end'] is not None: gff_records.append({'seqid': pid, 'type': 'Translocator', 'start': w['start']+1, 'end': w['end'], 'id': f"{pid}_TL_{idx}"})

            cargo_regions = get_cargo_regions(seq_len, exclusions_for_cargo)
            for idx, (cs, ce) in enumerate(cargo_regions): 
                gff_records.append({'seqid': pid, 'type': 'Cargo', 'start': cs, 'end': ce, 'id': f"{pid}_Cargo_{idx}"})

            p_plicat = plicat_data.get(pid, {})
            has_lipid_domain = any(any(c in d['category'].lower() for c in valid_lipid_cats) for d in dynamic_domains.get(pid, []))
            has_poly = any("polybasic" in d['name'].lower() or "polyacidic" in d['name'].lower() or "alps" in d['name'].lower() for d in dynamic_domains.get(pid, []))
            has_rxlr = any("rxlr" in d['name'].lower() for d in dynamic_domains.get(pid, []))
            
            subcell_hits = []
            for tool in ['wolf', 'deeploc', 'multiloc', 'apoplastp', 'localizer', 'targetp']:
                res_clean = subcell.get(tool, {}).get(pid, {}).get('pred', '')
                if res_clean and res_clean != 'None':
                    start_pos = 1 if tool == 'deeploc' else coord_offset + 1
                    if start_pos > seq_len: start_pos = 1
                    note_str = f"Note=Tool={tool.upper()};Prediction={res_clean}"
                    if tool == 'deeploc':
                        if dl_sig and dl_sig not in ['None', 'nan', '-']:
                            note_str += f";Signal={dl_sig}"
                    gff_records.append({'seqid': pid, 'type': 'Subcellular_Prediction', 'start': start_pos, 'end': seq_len, 'id': f"{pid}_{tool}", 'note': note_str})
                    subcell_hits.append(res_clean)
            
            tl_reasons = []
            if any(w['match_type'] in ['Poly', 'Both'] for w in translocator_windows): tl_reasons.append("TL (Poly)")
            if any(w['match_type'] in ['RF', 'Both'] for w in translocator_windows): tl_reasons.append("TL (RF)")
            if has_lipid_domain: tl_reasons.append("Membrane/Lipid Binding Domain")
            if has_poly: tl_reasons.append("Electrostatic Patch")
            if has_rxlr: tl_reasons.append("RXLR Domain")
            if p_plicat.get('passed'): tl_reasons.append("PLiCat Pass")
            
            overall_pred = "yes" if tl_reasons else "no"
            summary_stats[seq_type]['total'] += 1
            if overall_pred == "yes": 
                summary_stats[seq_type]['matches'] += 1
                for reason in tl_reasons: summary_stats[seq_type]['reasons'][reason] += 1
                    
            upset_tl_evidence.append(tl_reasons if tl_reasons else ['No Evidence'])
            upset_subcell_dest.append(subcell_hits if subcell_hits else ['No Localization'])
            m_list = re.findall(r'[\[\(]([^\]\)]+)[\]\)]', record.description)
            upset_categories.append([x.strip().upper() for x in m_list] if m_list else ['UNKNOWN'])
            
            rm_class = rippminer_res['class'].get(pid, 'None')

            results.append({
                "Protein id": pid, "Description": record.description, "Categories": seq_type, "Full sequence": full_seq, "Mature sequence": full_seq[sp_len:],
                "Protease-processed (PP) sequence": final_seq, "Length": seq_len, "Cysteines": full_seq.count('C'),
                "DeepTMHMM TM domains number": len(tm_domains), "Signal peptide length": sp_len if secreted == "yes" else "", 
                "Prodomain length": pro_len if pro_len > 0 else "", "Prodomain start": sp_len + 1 if pro_len > 0 else "", "Prodomain end": sp_len + pro_len if pro_len > 0 else "",
                "Primary proteolytic cleavage site": f"{primary_cut[2]} ({primary_cut[1]}) at {primary_cut[0] + sp_len}" if primary_cut else "",
                "Proteolytic cut sites (Coordinates)": "; ".join([f"{nm} ({seq}) ({p})" for p, seq, nm in full_cuts]),
                "Putative KEP RIPP": "Yes" if keps else "No",
                "KEP Regexes": "; ".join([k['regex'] for k in keps]),
                "KEP Frequencies": "; ".join([str(k['freq']) for k in keps]),
                "RiPPMiner Predicted Class": rm_class,
                "RiPPMiner Top Homology": top_ripp_hit,
                "Final product length": len(final_seq), "Final product CPPsite number of matches": len(cpp_hits_final),
                "Full protein CPPsite match start,end(s)": ";".join([f"{h[0]+1}-{h[1]}" for h in adj_cpp_hits]),
                "Domain/Motif Hits (Coordinates)": "; ".join([f"{d['name']} ({d['start']}-{d['end']})" + (f" [e-val: {d['evalue']}]" if d['evalue'] else "") for d in dynamic_domains.get(pid, []) if d['category'].lower() != 'proteolytic processing']),
                "CAZyme Domains (Coordinates)": "; ".join([f"{d['name']} ({d['start']}-{d['end']})" for d in dbcan_hits]),
                "Localizer Transit Peptides": "; ".join([f"{t['name']} ({t['start']}-{t['end']})" for t in loc_transits]),
                "DeepLoc2 Signals": dl_sig,
                "Full protein DREK motif start,end": ";".join([f"{s+1}-{e}" for s, e, _ in drek_hits]),
                "Translocator Domains (Coordinates)": "; ".join([f"TL ({w['match_type']}, {w['start']+1}-{w['end']})" for w in translocator_windows]),
                "Translocator Poly R2": "; ".join([f"{w['poly_r2']:.3f}" for w in translocator_windows]),
                "Translocator Poly RMSE": "; ".join([f"{w['poly_rmse']:.3f}" for w in translocator_windows]),
                "Translocator RF R2": "; ".join([f"{w['rf_r2']:.3f}" for w in translocator_windows]),
                "Translocator RF RMSE": "; ".join([f"{w['rf_rmse']:.3f}" for w in translocator_windows]),
                "Secreted": secreted, "Translocator_Motif": "yes" if translocator_windows else "no",
                "Translocator_Overall_Predicted": overall_pred, "Translocator_Prediction_Evidence": "; ".join(tl_reasons),
                "PLiCat Scores": "; ".join([f"{k}:{v:.3f}" for k, v in p_plicat.get('scores', {}).items()]),
                "WoLF_PSORT": subcell['wolf'].get(pid, {}).get('pred', ''), "DeepLoc": subcell['deeploc'].get(pid, {}).get('pred', ''), 
                "MultiLoc2": subcell['multiloc'].get(pid, {}).get('pred', ''), "ApoplastP": subcell['apoplastp'].get(pid, {}).get('pred', ''), 
                "LOCALIZER": subcell['localizer'].get(pid, {}).get('pred', ''), "TargetP": subcell['targetp'].get(pid, {}).get('pred', '')
            })

            plot_data.append({
                'id': pid, 'title': f"{pid} {record.description}".strip(), 'full_seq': full_seq, 'profile': full_kd_profile, 'len': seq_len, 
                'cdata': cdata, 'keps': keps, 'full_cuts': full_cuts, 'dynamic_domains': dynamic_domains.get(pid, []),
                'drek_data': drek_hits, 'rgd_data': rgd_hits, 'cpp_data': adj_cpp_hits, 'dbcan_data': dbcan_hits, 'tl_data': translocator_windows, 
                'loc_transits': loc_transits, 'cargo': cargo_regions, 'cys': [idx + 1 for idx, aa in enumerate(full_seq) if aa.upper() == 'C'], 
                'tm_data': tm_domains, 'subcell': {k: v.get(pid, {}) for k, v in subcell.items() if k != 'aiupred'}, 
                'aiupred': subcell['aiupred'].get(pid, {}), 'tl_reasons': tl_reasons, 'plicat_trunc': p_plicat.get('truncated', False), 'ripp_hit': top_ripp_hit,
                'ripp_class': rippminer_res['class'].get(pid, 'None'),
                'ripp_sims': [s for s in rippminer_res['similarities'] if s.get('Query') == pid],
                'external_features': ext_feats
            })
        except Exception as e: print(f"Error processing {record.id}: {e}\n{traceback.format_exc()}", flush=True)

    if intermediate_keps: pd.DataFrame(intermediate_keps).to_csv(os.path.join(kep_dir, "intermediate_kep_repeats.csv"), index=False)
    if results: pd.DataFrame(results).to_csv(out_csv, index=False)
    if gff_records: generate_gff3(gff_records, out_gff)
    
    print(f"Generating enhanced GridSpec SVG plots for {len(plot_data)} sequences... [100%]", flush=True)
    
    std_locations = ['Extracellular /\nNon-apoplastic*', 'Plasma Membrane /\nApoplastic*', 'Cytoplasm', 'Nucleus', 'Mitochondria', 'Chloroplast', 'Endoplasmic\nReticulum', 'Golgi', 'Vacuole', 'Peroxisome']
    loc_mappings = {
        'extr': 'Extracellular /\nNon-apoplastic*',
        'Extracellular': 'Extracellular /\nNon-apoplastic*',
        'extracellular': 'Extracellular /\nNon-apoplastic*',
        'Non-apoplastic': 'Extracellular /\nNon-apoplastic*',
        'non-apoplastic': 'Extracellular /\nNon-apoplastic*',
        'Apoplast': 'Plasma Membrane /\nApoplastic*',
        'apoplast': 'Plasma Membrane /\nApoplastic*',
        'Cell membrane': 'Plasma Membrane /\nApoplastic*',
        'plas': 'Plasma Membrane /\nApoplastic*',
        'Plasma Membrane': 'Plasma Membrane /\nApoplastic*',
        'plasma membrane': 'Plasma Membrane /\nApoplastic*',
        'Apoplastic': 'Plasma Membrane /\nApoplastic*',
        'apoplastic': 'Plasma Membrane /\nApoplastic*',
        'Cytoplasm': 'Cytoplasm', 'cyto': 'Cytoplasm',
        'Nucleus': 'Nucleus', 'nucl': 'Nucleus',
        'Mitochondrion': 'Mitochondria', 'Mitochondria': 'Mitochondria', 'mito': 'Mitochondria',
        'Chloroplast': 'Chloroplast', 'chlo': 'Chloroplast', 'Plastid': 'Chloroplast',
        'Endoplasmic reticulum': 'Endoplasmic\nReticulum', 'Endoplasmic Reticulum': 'Endoplasmic\nReticulum', 'E.R.': 'Endoplasmic\nReticulum',
        'Golgi apparatus': 'Golgi', 'Golgi': 'Golgi',
        'Lysosome/Vacuole': 'Vacuole', 'vacu': 'Vacuole', 'lyso': 'Lysosome',
        'Peroxisome': 'Peroxisome', 'pero': 'Peroxisome',
        'Other': 'Cytoplasm', 'other': 'Cytoplasm'
    }

    cmap = plt.get_cmap('tab20')
    domain_colors = [mcolors.to_hex(cmap(i)) for i in range(20)]

    for p in plot_data:
        try:
            fig = plt.figure(figsize=(20, 14))
            # Subcellular left-aligned (0.64 * 0.8 = 0.512 approx), PLiCat pushed right (0.1)
            gs_main = GridSpec(3, 5, height_ratios=[5, 1.6, 3], width_ratios=[0.46, 0.07, 0.025, 0.03, 0.025], hspace=0.15, wspace=0.25)
            
            # Subgridspec for the 4 strips to make them perfectly adjacent (hspace=0.0)
            gs_strips = gs_main[1, :].subgridspec(4, 1, hspace=0.0)
            
            # --- TOP: KD PLOT ---
            ax1 = fig.add_subplot(gs_main[0, :])
            x_vals = np.arange(1, p['len'] + 1)
            ax1.plot(x_vals, p['profile'], color='black', linewidth=1.2)
            ax1.set_xlim(1, p['len'])
            ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)

            occupied_levels = {}
            cdata = p['cdata']
            
            def mark_occupied(start, end):
                occupied_levels.setdefault(0, []).append((start, end))

            if cdata['sp_len'] > 0: 
                ax1.plot(x_vals[:cdata['sp_len']], p['profile'][:cdata['sp_len']], color='#2ca02c', linewidth=18, alpha=0.5, zorder=4, label='Signal Peptide')
                sp_c_y = get_y_val(cdata['sp_len'], p['profile'])
                ax1.vlines(x=cdata['sp_len'], ymin=sp_c_y - 1.2, ymax=sp_c_y + 1.2, colors='#2ca02c', linestyles='dotted', lw=2)
                mark_occupied(1, cdata['sp_len'])

            plotted_cut_labels = set()
            
            if cdata['is_kep']:
                # Draw KEP Blocks Anchored
                for k_grp in p['keps']:
                    lbl = f"Putative KEP RIPP ({k_grp['freq']}x)" if k_grp == p['keps'][0] else ""
                    inst_centers = [inst['start'] + (inst['end'] - inst['start'])//2 for inst in k_grp['instances']]
                    for inst in k_grp['instances']:
                        ax1.plot(x_vals[inst['start']-1:inst['end']], p['profile'][inst['start']-1:inst['end']], color='#d62728', linewidth=18, alpha=0.5, zorder=3, label=lbl if inst == k_grp['instances'][0] else "")
                        mark_occupied(inst['start'], inst['end'])
                    
                    if inst_centers:
                        mean_x = np.mean(inst_centers)
                        cy_top = get_y_val(mean_x, p['profile']) + 2.5
                        if cy_top > 4.0: cy_top = get_y_val(mean_x, p['profile']) - 2.5 
                        
                        ripp_str = f" [Sim: {p['ripp_hit']}]" if p.get('ripp_hit') else ""
                        ax1.text(mean_x, cy_top, k_grp['regex'] + ripp_str, color='white', fontsize=7, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='#d62728', alpha=0.8, boxstyle='round,pad=0.3'))
                        for cx in inst_centers:
                            offset = -0.2 if cy_top > get_y_val(mean_x, p['profile']) else 0.2
                            ax1.plot([cx, mean_x], [get_y_val(cx, p['profile']) + (0.5 if offset < 0 else -0.5), cy_top + offset], color='#d62728', linestyle='-', linewidth=0.8, alpha=0.7)
                
                # Draw all Kex2 cuts as dotted lines
                for pos_rel_full, seq, nm in p['full_cuts']:
                    kx_c_y = get_y_val(pos_rel_full, p['profile'])
                    ax1.vlines(x=pos_rel_full, ymin=kx_c_y - 1.2, ymax=kx_c_y + 1.2, colors='blue', linestyles='dotted', lw=2)
                    ax1.text(pos_rel_full, kx_c_y + 1.4, seq, color='blue', fontsize=7, ha='center', va='bottom', zorder=10, fontweight='bold')
                
                # Draw RiPPMiner Class block if present
                r_class = p.get('ripp_class', 'None')
                if r_class and r_class != 'None':
                    pro_start = cdata['sp_len'] + cdata['pro_len'] + 1
                    if pro_start <= p['len']:
                        ax1.plot(x_vals[pro_start-1:], p['profile'][pro_start-1:], color='#ba55d3', linewidth=18, alpha=0.4, zorder=2, label=f"RiPPMiner Class: {r_class}")
                
                # Draw RiPPMiner Similarity Hits
                plotted_sim_count = 0
                for sim in p.get('ripp_sims', []):
                    try:
                        s_start = int(sim['Start'])
                        s_end = int(sim['End'])
                        evalue = sim['Evalue']
                        subject = sim['Subject']
                        
                        if plotted_sim_count >= 3:
                            break
                        
                        # Plot hit region on the profile
                        lbl = f"RiPPMiner Hit: {subject}" if plotted_sim_count == 0 else ""
                        ax1.plot(x_vals[s_start-1:s_end], p['profile'][s_start-1:s_end], color='#ff00ff', linewidth=10, alpha=0.6, zorder=4, label=lbl)
                        
                        # Draw annotation box
                        mid_x = (s_start + s_end) // 2
                        y_pos = get_y_val(mid_x, p['profile'])
                        text_y = y_pos - 1.8 - (plotted_sim_count * 0.7)
                        
                        ax1.text(mid_x, text_y, f"{subject}\n(e-val: {evalue})", color='black', fontsize=6, ha='center', va='top',
                                 bbox=dict(facecolor='white', edgecolor='#ff00ff', alpha=0.8, boxstyle='round,pad=0.2'))
                        ax1.plot([mid_x, mid_x], [y_pos, text_y], color='#ff00ff', linestyle=':', linewidth=0.8)
                        
                        plotted_sim_count += 1
                    except Exception as ex:
                        print(f"Error plotting RiPPMiner similarity hit: {ex}", flush=True)
            else:
                for idx, (pos_rel_full, seq, nm) in enumerate(p['full_cuts']):
                    kx_c_y = get_y_val(pos_rel_full, p['profile'])
                    is_pexel = "PEXEL" in nm.upper()
                    pt_color = '#d62728' if is_pexel else '#1f77b4'
                    legend_label = "PEXEL cleavage site" if is_pexel else "Kex2 cleavage site"
                    lbl = legend_label if legend_label not in plotted_cut_labels else ""
                    plotted_cut_labels.add(legend_label)

                    ax1.scatter([pos_rel_full], [kx_c_y], color=pt_color, marker='o', s=50, zorder=6, label=lbl)
                    ax1.text(pos_rel_full, kx_c_y + 1.4, seq, color=pt_color, fontsize=7, ha='center', va='bottom', zorder=10, fontweight='bold')
                    
                    if cdata['primary_cut'] and pos_rel_full == (cdata['primary_cut'][0] + cdata['sp_len']):
                        pro_start = cdata['sp_len'] + 1 if cdata['sp_len'] > 0 else 1
                        if pro_start < pos_rel_full:
                            ax1.plot(x_vals[pro_start-1:pos_rel_full], p['profile'][pro_start-1:pos_rel_full], color=pt_color, linewidth=18, alpha=0.5, zorder=4, label='Prodomain')
                            mark_occupied(pro_start, pos_rel_full)
                        ax1.vlines(x=pos_rel_full, ymin=kx_c_y - 1.2, ymax=kx_c_y + 1.2, colors=pt_color, linestyles='dotted', lw=2)

            for i, (s, e) in enumerate(p['tm_data']): ax1.axvspan(s-0.5, e+0.5, color='#00ced1', alpha=0.35, label='TM Domain' if i==0 else "")
            
            plotted_rgd = False
            for (s, e) in p.get('rgd_data', []):
                ax1.plot(x_vals[s:e], p['profile'][s:e], color='#32CD32', linewidth=18, alpha=0.5, zorder=2, label="RGD Motif" if not plotted_rgd else "")
                mark_occupied(s+1, e)
                plotted_rgd = True
                
            poly_patches = [d for d in p['dynamic_domains'] if 'polybasic' in d['name'].lower() or 'polyacidic' in d['name'].lower()]
            for d in poly_patches:
                ax1.plot(x_vals[d['start']-1:d['end']], p['profile'][d['start']-1:d['end']], color=domain_colors[hash(d['name']) % 20], linewidth=18, alpha=0.5, zorder=3, label=d['name'])
                mark_occupied(d['start'], d['end'])

            if not cdata['is_kep']:
                for i, (s, e, seq) in enumerate(p['drek_data']): 
                    ax1.plot(x_vals[s:e], p['profile'][s:e], color='#ff7f0e', linewidth=4, zorder=5, label='DREK Motifs' if i==0 else "")
                    mark_occupied(s+1, e)
                    cy_mid = get_y_val(s + (e - s)//2, p['profile'])
                    ax1.text(s + (e - s)//2, cy_mid + 0.5, seq, color='#ff7f0e', fontsize=6, ha='center', va='bottom', zorder=10, fontweight='bold')

            # Localizer Transit Peptides overlay
            for idx, loc_trans in enumerate(p.get('loc_transits', [])):
                ax1.plot(x_vals[loc_trans['start']-1:loc_trans['end']], p['profile'][loc_trans['start']-1:loc_trans['end']], color='#e377c2', linewidth=12, alpha=0.6, zorder=4, label=loc_trans['name'] if idx==0 else "")

            # Stagger TL ONLY if overlapping SP or Prodomain
            plotted_poly, plotted_rf = False, False
            for w in p['tl_data']:
                overlap_sp_pro = any(not (w['end'] < anc_s or w['start']+1 > anc_e) for anc_s, anc_e in occupied_levels.get(0, []))
                lvl = get_overlap_level(w['start']+1, w['end'], occupied_levels) if overlap_sp_pro else 0
                y_offset = p['profile'][w['start']:w['end']] + (lvl*0.35 if lvl > 0 else 0)
                
                if w['match_type'] in ['Poly', 'Both']:
                    ax1.plot(x_vals[w['start']:w['end']], y_offset, color='#9370DB', linewidth=18, zorder=3, alpha=0.5, label="TL (Poly)" if not plotted_poly else "")
                    plotted_poly = True
                if w['match_type'] in ['RF', 'Both']:
                    ax1.plot(x_vals[w['start']:w['end']], y_offset, color='#DA70D6', linewidth=18, zorder=3, alpha=0.5, label="TL (RF)" if not plotted_rf else "")
                    plotted_rf = True

            merged_cpp = merge_cpp(p['cpp_data'])
            merged_cpp.sort(key=lambda x: (x[1] - x[0]), reverse=True)
            for i, (s, e, c_id) in enumerate(merged_cpp):
                ax1.plot(x_vals[s:e], p['profile'][s:e], color='#17becf', linewidth=8, zorder=3, alpha=0.4, label='CPPSite Match' if i==0 else "")

            merged_dbcan = merge_dbcan(p['dbcan_data'])
            merged_dbcan.sort(key=lambda x: (x['end'] - x['start']), reverse=True)
            for i, d in enumerate(merged_dbcan):
                lvl = get_overlap_level(d['start'], d['end'], occupied_levels)
                ax1.plot(x_vals[d['start']-1:d['end']], p['profile'][d['start']-1:d['end']] + (lvl*0.35 if lvl > 0 else 0), color='#FFD700', linewidth=18, alpha=0.5, zorder=4, label=d['name'])

            plotted_transit = False
            doms_to_plot = [d for d in p['dynamic_domains'] if 'polybasic' not in d['name'].lower() and 'polyacidic' not in d['name'].lower() and d['category'].lower() != 'proteolytic processing']
            doms_to_plot.sort(key=lambda x: (x['end'] - x['start']), reverse=True)
            for dom in doms_to_plot:
                overlap_anchor = any(not (dom['end'] < anc_s or dom['start'] > anc_e) for anc_s, anc_e in occupied_levels.get(0, []))
                lvl = get_overlap_level(dom['start'], dom['end'], occupied_levels) if overlap_anchor else 0
                c = '#e377c2' if 'transit tag' in dom['category'].lower() else domain_colors[hash(dom['name']) % 20]
                lbl = 'Transit Tag' if ('transit tag' in dom['category'].lower() and not plotted_transit) else dom['name']
                if 'transit tag' in dom['category'].lower(): plotted_transit = True
                ax1.plot(x_vals[dom['start']-1:dom['end']], p['profile'][dom['start']-1:dom['end']] + (lvl*0.35 if lvl > 0 else 0), color=c, linewidth=8, zorder=4, alpha=0.5, label=lbl)

            # Draw external features if present
            plotted_ext_lbls = set()
            for idx, feat in enumerate(p.get('external_features', [])):
                s = feat['start']
                e = feat['end']
                if s is None or e is None or s > p['len'] or e > p['len']:
                    continue
                s_idx = max(0, s - 1)
                e_idx = min(p['len'], e)
                if s_idx >= e_idx:
                    continue
                
                overlap_anchor = any(not (e < anc_s or s > anc_e) for anc_s, anc_e in occupied_levels.get(0, []))
                lvl = get_overlap_level(s, e, occupied_levels) if overlap_anchor else 0
                y_offset = p['profile'][s_idx:e_idx] + (lvl * 0.35 if lvl > 0 else 0)
                
                c = '#D2691E'  # chocolate color for external features
                legend_lbl = "External Feature" if "External Feature" not in plotted_ext_lbls else ""
                plotted_ext_lbls.add("External Feature")
                
                ax1.plot(x_vals[s_idx:e_idx], y_offset, color=c, linewidth=8, zorder=4, alpha=0.6, label=legend_lbl)
                
                mid_x = (s + e) // 2
                y_pos = get_y_val(mid_x, p['profile']) + (lvl * 0.35 if lvl > 0 else 0)
                text_y = y_pos + 0.8
                
                ax1.text(mid_x, text_y, feat['name'], color='black', fontsize=7, ha='center', va='bottom',
                         bbox=dict(facecolor='white', edgecolor=c, alpha=0.8, boxstyle='round,pad=0.15'), zorder=10)
                ax1.plot([mid_x, mid_x], [y_pos, text_y], color=c, linestyle=':', linewidth=0.8)

            if p['cys']: ax1.scatter(p['cys'], [get_y_val(c, p['profile']) for c in p['cys']], color='red', s=25, zorder=5, label='Cysteine')

            all_y = p['profile'][~np.isnan(p['profile'])]
            y_min, y_max = (math.floor(np.min(all_y)) - 0.5, math.ceil(np.max(all_y)) + 0.5) if len(all_y) > 0 else (-4.5, 4.5)
            max_offset = max([abs(k) for k in occupied_levels.keys()] + [0]) * 0.35
            ax1.set_ylim(y_min - max_offset - 1.5, y_max + max_offset + 2.5)

            title_text = p['title']
            ax1.set_title(title_text, pad=25)
            ax1.set_ylabel('Hydropathy KD')
            
            y_base = 1.02
            is_tl_pred = len(p['tl_reasons']) > 0
            tl_color = 'green' if is_tl_pred else 'purple'
            if is_tl_pred:
                ax1.text(0.5, y_base, f"Overall Prediction: Translocator [{', '.join(p['tl_reasons'])}]", transform=ax1.transAxes, fontsize=10, fontweight='bold', color=tl_color, ha='center')
            else:
                ax1.text(0.5, y_base, "Overall Prediction: No Translocator Evidence", transform=ax1.transAxes, fontsize=10, fontweight='bold', color='black', ha='center')
            
            handles, labels = ax1.get_legend_handles_labels()
            unique_hl = dict(zip(labels, handles))
            if unique_hl: ax1.legend(unique_hl.values(), unique_hl.keys(), bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0.)

            # --- AIUPRED STRIPS (Tightly stacked, no inner x-labels) ---
            coord_offset = cdata['sp_len'] + cdata['pro_len']
            
            ax_dis = fig.add_subplot(gs_strips[0, 0], sharex=ax1)
            if p['aiupred'] and p['aiupred'].get('disorder'):
                arr = np.array(p['aiupred']['disorder'])
                pad_left = 0
                pad_right = max(0, p['len'] - len(arr) - pad_left)
                if len(arr) + pad_left > p['len']:
                    arr = arr[:p['len'] - pad_left]
                arr = np.pad(arr, (pad_left, pad_right))
                ax_dis.imshow(arr.reshape(1, -1), cmap='Reds', aspect='auto', extent=[1, p['len'], 0, 1], vmin=0, vmax=1)
            ax_dis.set_yticks([]); ax_dis.set_ylabel('Disorder', rotation=0, labelpad=30, va='center')
            plt.setp(ax_dis.get_xticklabels(), visible=False)
            
            ax_bind = fig.add_subplot(gs_strips[1, 0], sharex=ax1)
            if p['aiupred'] and p['aiupred'].get('binding'):
                arr = np.array(p['aiupred']['binding'])
                pad_left = 0
                pad_right = max(0, p['len'] - len(arr) - pad_left)
                if len(arr) + pad_left > p['len']:
                    arr = arr[:p['len'] - pad_left]
                arr = np.pad(arr, (pad_left, pad_right))
                ax_bind.imshow(arr.reshape(1, -1), cmap='Purples', aspect='auto', extent=[1, p['len'], 0, 1], vmin=0, vmax=1)
            ax_bind.set_yticks([]); ax_bind.set_ylabel('Binding', rotation=0, labelpad=30, va='center')
            plt.setp(ax_bind.get_xticklabels(), visible=False)

            ax_link = fig.add_subplot(gs_strips[2, 0], sharex=ax1)
            if p['aiupred'] and p['aiupred'].get('linker'):
                arr = np.array(p['aiupred']['linker'])
                pad_left = 0
                pad_right = max(0, p['len'] - len(arr) - pad_left)
                if len(arr) + pad_left > p['len']:
                    arr = arr[:p['len'] - pad_left]
                arr = np.pad(arr, (pad_left, pad_right))
                ax_link.imshow(arr.reshape(1, -1), cmap='Greens', aspect='auto', extent=[1, p['len'], 0, 1], vmin=0, vmax=1)
            ax_link.set_yticks([]); ax_link.set_ylabel('Linker', rotation=0, labelpad=30, va='center')
            plt.setp(ax_link.get_xticklabels(), visible=False)

            # --- CARGO STRIP ---
            ax_cargo = fig.add_subplot(gs_strips[3, 0], sharex=ax1)
            cargo_mat = np.zeros((1, p['len']))
            for cs, ce in p['cargo']: cargo_mat[0, cs-1:ce] = 1.0
            ax_cargo.imshow(cargo_mat, cmap='Blues', aspect='auto', extent=[1, p['len'], 0, 1], vmin=0, vmax=1)
            ax_cargo.set_yticks([]); ax_cargo.set_ylabel('Cargo', rotation=0, labelpad=30, va='center')
            ax_cargo.set_xlabel('Amino Acid Position')

            # --- BOTTOM LEFT: SUBCELLULAR MATRIX (0.51 Width) ---
            ax_sub = fig.add_subplot(gs_main[2, 0])
            tool_names = ['WoLF PSORT', 'DeepLoc2', 'MultiLoc2', 'ApoplastP / LOCALIZER', 'TargetP']
            tool_keys = ['wolf', 'deeploc', 'multiloc', 'apoplastp_localizer', 'targetp']
            
            mat = np.full((len(tool_names), len(std_locations)), np.nan)
            for i, t_key in enumerate(tool_keys):
                if t_key == 'apoplastp_localizer':
                    t_scores = {}
                    t_scores.update(p['subcell'].get('apoplastp', {}).get('scores', {}))
                    t_scores.update(p['subcell'].get('localizer', {}).get('scores', {}))
                else:
                    t_scores = p['subcell'].get(t_key, {}).get('scores', {})
                
                if not t_scores: continue
                if t_key in ['wolf', 'deeploc', 'multiloc']:
                    for wl in ['Extracellular', 'Plasma Membrane', 'Cytoplasm', 'Nucleus', 'Mitochondria', 'Chloroplast', 'Endoplasmic Reticulum', 'Golgi', 'Vacuole', 'Peroxisome']:
                        mapped_wl = loc_mappings.get(wl, wl)
                        if mapped_wl in std_locations:
                            mat[i, std_locations.index(mapped_wl)] = 0.0
                elif t_key == 'apoplastp_localizer':
                    if p['subcell'].get('apoplastp', {}).get('scores', {}):
                        for wl in ['Extracellular', 'Plasma Membrane']:
                            mapped_wl = loc_mappings.get(wl, wl)
                            if mapped_wl in std_locations:
                                mat[i, std_locations.index(mapped_wl)] = 0.0
                max_sc = 10.0 if t_key == 'wolf' else 1.0
                for loc_raw, score in t_scores.items():
                    std_loc = loc_mappings.get(loc_raw, loc_raw)
                    if std_loc in std_locations:
                        j = std_locations.index(std_loc)
                        val = score / max_sc
                        if val > 1.0: val = 1.0
                        if np.isnan(mat[i, j]): mat[i, j] = val
                        else: mat[i, j] = min(mat[i, j] + val, 1.0)
                        
            cmap_mat = plt.get_cmap('YlGnBu').copy()
            cmap_mat.set_bad('lightgrey')
            
            ax_sub.imshow(mat, cmap=cmap_mat, aspect='auto', vmin=0, vmax=1)
            ax_sub.set_xticks(np.arange(len(std_locations)))
            ax_sub.set_yticks(np.arange(len(tool_names)))
            ax_sub.set_xticklabels(std_locations, rotation=45, ha="right")
            ax_sub.set_yticklabels(tool_names)
            
            for i in range(len(tool_names)):
                for j in range(len(std_locations)):
                    val = mat[i, j]
                    if not np.isnan(val):
                        text_col = "white" if val > 0.6 else "black"
                        ax_sub.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=8)

            # --- BOTTOM MIDDLE: DEEPLOC2 MEMBRANE TYPES (0.025 Width) ---
            ax_memb = fig.add_subplot(gs_main[2, 2])
            memb_names = ["Peripheral", "Transmembrane", "Lipid Anchor", "Soluble"]
            memb_keys = ["Peripheral", "Transmembrane", "Lipid anchor", "Soluble"]
            memb_mat = np.full((len(memb_names), 1), np.nan)
            
            dl_scores = p['subcell'].get('deeploc', {}).get('scores', {})
            if dl_scores:
                for i, key in enumerate(memb_keys):
                    val = dl_scores.get(key)
                    if val is None:
                        val = dl_scores.get(key.lower())
                    if val is not None:
                        memb_mat[i, 0] = float(val)
            
            cmap_memb = plt.get_cmap('Oranges').copy()
            cmap_memb.set_bad('lightgrey')
            
            ax_memb.imshow(memb_mat, cmap=cmap_memb, aspect='auto', vmin=0, vmax=1)
            ax_memb.set_xticks([0])
            ax_memb.set_xticklabels(["Probability"])
            ax_memb.set_yticks(np.arange(len(memb_names)))
            ax_memb.set_yticklabels(memb_names)
            ax_memb.set_title("DeepLoc2 Membrane")
            
            for i in range(len(memb_names)):
                val = memb_mat[i, 0]
                if not np.isnan(val):
                    text_col = "white" if val > 0.6 else "black"
                    ax_memb.text(0, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=9, fontweight='bold' if val > 0.6 else 'normal')

            # --- BOTTOM RIGHT: PLICAT MATRIX (0.1 Width) ---
            ax_plic = fig.add_subplot(gs_main[2, 4])
            p_plicat = plicat_data.get(p['id'], {})
            
            lipid_names = ["NotLipidType", "Fatty Acyl", "Prenol Lipid", "Glycerophospholipid", "Sterol Lipid", "Polyketide", "Glycerolipid", "Sphingolipid", "Saccharolipid"]
            pl_mat = np.full((len(lipid_names), 1), np.nan)
            
            if p_plicat and p_plicat.get('scores'):
                for i, l_name in enumerate(lipid_names):
                    pl_mat[i, 0] = p_plicat['scores'].get(l_name, 0.0)
            
            cmap_pl = plt.get_cmap('YlOrRd').copy()
            cmap_pl.set_bad('lightgrey')
            
            ax_plic.imshow(pl_mat, cmap=cmap_pl, aspect='auto', vmin=0, vmax=1)
            ax_plic.set_xticks([0])
            ax_plic.set_xticklabels(["Probability"])
            ax_plic.set_yticks(np.arange(len(lipid_names)))
            ax_plic.set_yticklabels(lipid_names)
            ax_plic.set_title(f"PLiCat Binding" + (" (Truncated)" if p.get('plicat_trunc') else ""))
            
            for i in range(len(lipid_names)):
                val = pl_mat[i, 0]
                if not np.isnan(val):
                    text_col = "white" if val > 0.6 else "black"
                    ax_plic.text(0, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=9, fontweight='bold' if val > 0.6 else 'normal')

            # Force subplots alignment to smash AIUPred strips together
            # plt.subplots_adjust(hspace=0.0)
            fig.savefig(os.path.join(plots_dir, f"{sanitize_filename(p['id'])}.svg"), bbox_inches='tight')
            plt.close(fig)
        except Exception as e: print(f"Error plotting {p['id']}: {e}", flush=True)

    print("\n================ OVERALL TRANSLOCATOR EVALUATION METRICS ================", flush=True)
    for seq_type, stats in sorted(summary_stats.items()):
        total = stats['total']
        matches = stats['matches']
        acc = (matches / total) * 100 if total > 0 else 0
        print(f"[{seq_type}] Total Proteins: {total} | Predicted Translocators: {matches} | Percentage: {acc:.1f}%", flush=True)
        if matches > 0:
            print(f"  --- Supporting Evidence Breakdown ---", flush=True)
            for reason, r_count in sorted(stats['reasons'].items(), key=lambda x: x[1], reverse=True):
                r_perc = (r_count / matches) * 100
                print(f"      * {reason}: {r_count} ({r_perc:.1f}%)", flush=True)
        print("", flush=True) 
    print("=======================================================================", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help="Input protein FASTA file")
    parser.add_argument('--protease-motifs', default="kr/,rr/,kk/,rk/,[LIVMA].[KRTPLI]R/,R.L/.[EDQ]", help="Kex2/protease motifs fallback")
    parser.add_argument('--translocator-windowsize', type=int, default=21)
    parser.add_argument('--kep-identity', type=float, default=0.80, help="Identity threshold for collapsing KEP repeats (0.0 to 1.0)")
    parser.add_argument('--max-prodomain-percent', type=float, default=0.70, help="Max percent of protein allowed for prodomain cleavage site")
    parser.add_argument('--bin-dir', default="bin", help="Path to subfolder containing installed tools")
    parser.add_argument('--deeploc-model', choices=['Accurate', 'Fast'], default='Fast', help="DeepLoc2 model to use: 'Accurate' (ProtT5 - slow, high RAM) or 'Fast' (ESM1b - fast, lower RAM)")
    parser.add_argument('--cppsite-fasta', default="data/20240313_CPPsite2_peptide_seqeunces_cleaned.fasta")
    parser.add_argument('--cppsite-csv', default="data/20240312_CPPSite2_data.csv")
    parser.add_argument('--prev-data', help="Reuse completed data from a previous run to save CPU time. Format: folder:analysis1,analysis2 (e.g. outdir:signalp,plicat,deeptmhmm) or just folder to reuse all valid analyses.")
    parser.add_argument('--gff3', help="Path to input GFF3 file containing protein features to overlay on plots and GFF3 outputs")
    parser.add_argument('--bed', help="Path to input BED file containing protein features to overlay on plots and GFF3 outputs")
    args = parser.parse_args()
    process_proteins(args)