"""Orchestrates native Vx/CUDA programs; Python is outside every timed region."""
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
LAB = HERE.parents[1]
VXC_ROOT = Path(os.environ.get('VXC_ROOT', str(LAB/'Vx'))).resolve()
LLVM = Path(os.environ.get('MLIR_SYS_220_PREFIX') or subprocess.check_output(['llvm-config', '--prefix'], text=True).strip())
if os.environ.get('VX_DISPATCH_LIB'):
    OUT = Path(os.environ['VX_DISPATCH_LIB']).resolve().parent
else:
    OUT = max((VXC_ROOT/'target/debug/build').glob('vxc-*/out/libvx_cuda_dispatch.so'), key=lambda p: p.stat().st_mtime).parent
CASES = [(1,4), (4,256), (256,256), (4096,256)]

def command(args, env=None):
    proc = subprocess.run([str(x) for x in args], text=True, capture_output=True, env=env)
    if proc.returncode:
        raise RuntimeError(f'{args}\n{proc.stdout}\n{proc.stderr}')
    return proc

def source(rows, threads):
    original=(HERE/'hello_gpu.vx').read_text().split('// GPU kernel:')[0]
    original=original.replace('capacity: 64 MiB','capacity: 128 MiB')
    return original+f'''
import std::time;
extern {{ fn cudaDeviceSynchronize() -> i32; }}
pub fn add(result: &mut Tensor<f32, [{rows}, {threads}], Memory::HBM>,
           a: &Tensor<f32, [{rows}, {threads}], Memory::HBM>,
           b: &Tensor<f32, [{rows}, {threads}], Memory::HBM>) -> void {{
  spawn on(Topology::Device) {{
    for block in 0..{rows} across blocks {{
      for lane in 0..{threads} across lanes {{
        result[block][lane] = a[block][lane] + b[block][lane];
      }}
    }}
  }}
  return;
}}
fn sync() -> void {{
  let rc = unsafe {{ cudaDeviceSynchronize() }};
  assert(rc == 0, "CUDA synchronization failed");
  return;
}}
fn main() -> i32 {{
  let mut a = Tensor<f32, [{rows}, {threads}]>::fill(0.0);
  let mut b = Tensor<f32, [{rows}, {threads}]>::fill(0.0);
  let result = Tensor<f32, [{rows}, {threads}]>::fill(0.0);
  for row in 0..{rows} {{
    for lane in 0..{threads} {{
      let i = row * {threads} + lane;
      a[row][lane] = (i % 97) as f32;
      b[row][lane] = (i % 31) as f32;
    }}
  }}
  sync();
  let upload = now();
  let ag = transfer(a, Memory::HBM);
  let bg = transfer(b, Memory::HBM);
  let mut cg = transfer(result, Memory::HBM);
  sync();
  unsafe {{ bench_report("upload_ms", "ms", (now()-upload)*1000.0); }}
  for sample in 0..2000 {{ add(&mut cg, &ag, &bg); sync(); }}
  let start = now();
  for sample in 0..501 {{ add(&mut cg, &ag, &bg); sync(); }}
  let elapsed = now()-start;
  unsafe {{ bench_report("host_us", "us", elapsed*1000000.0/501.0); }}
  let download = now();
  let answer = transfer(cg, Memory::CPU_DRAM);
  sync();
  unsafe {{ bench_report("download_ms", "ms", (now()-download)*1000.0); }}
  for row in 0..{rows} {{
    for lane in 0..{threads} {{
      let i = row * {threads} + lane;
      let expected = ((i % 97) as f32) + ((i % 31) as f32);
      assert(answer[row][lane] == expected, "wrong sum");
    }}
  }}
  println!("PASS");
  return 0;
}}
'''

def build():
    cuda=Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    # CUDA development libraries must be present; never modify the toolkit.
    command(['clang++','-O3','cuda_bench.cpp','-I'+str(cuda/'include'),'-L'+str(cuda/'lib64'),
             '-L'+os.environ.get('CUDA_DRIVER_LIB_DIR', '/opt/nvidia/l4t-gpu-libs/nvgpu'),'-lcudart','-lcuda','-lnvrtc','-o','cuda_bench'])
    for rows,threads in CASES:
        name=f'vx_{rows*threads}'
        Path(name+'.vx').write_text(source(rows,threads))
        command([VXC_ROOT/'target/debug/vxc','--host','default','-O3','-c',name+'.vx','-o',name+'.o'])
        libs=[LLVM/'lib/libmlir_c_runner_utils.so',LLVM/'lib/libmlir_runner_utils.so',
              VXC_ROOT/'target/debug/libvx_std_core.so',OUT/'libvx_cuda_dispatch.so',OUT/'libvx_mlir_shims.so']
        command(['clang++',name+'.o',*libs,'-rdynamic','-lffi','-lm','-L'+str(cuda/'lib64'),
                 '-lcudart','-lcublas','-o',name])
        print(f'Built {name}',flush=True)

def metrics(stdout):
    data={}
    for line in stdout.splitlines():
        fields=line.split()
        if fields and fields[0]=='vx-bench': data[fields[1]]=float(fields[3])
        elif len(fields)==2 and fields[0] in ('host_us','upload_ms','download_ms'): data[fields[0]]=float(fields[1])
    return data

def run():
    results=[]
    base=os.environ.copy()
    for key in ('VX_CUDA_SYNC','VX_TIME_KERNEL','VX_DISPATCH_VERBOSE'): base.pop(key,None)
    for rows,threads in CASES:
        n=rows*threads
        for repetition in range(5):
            # Alternate order to reduce systematic clock/temperature drift.
            for lang in (('vx','cuda') if repetition%2==0 else ('cuda','vx')):
                for timed in (False,True):
                    env=base.copy()
                    if timed and lang=='vx': env['VX_TIME_KERNEL']='1'
                    args=[HERE/f'vx_{n}'] if lang=='vx' else [HERE/'cuda_bench',rows,threads,int(timed)]
                    started=time.perf_counter()
                    proc=command(args,env)
                    assert 'PASS' in proc.stdout
                    samples=re.findall(r'device time ([\d.]+) ms',proc.stderr) if lang=='vx' else re.findall(r'kernel_ms ([\d.]+)',proc.stderr)
                    if timed: assert len(samples)==2501, (lang,n,len(samples),proc.stderr)
                    entry=dict(lang=lang,n=n,rows=rows,threads=threads,repetition=repetition,timed=timed,
                               process_ms=(time.perf_counter()-started)*1000,**metrics(proc.stdout))
                    if timed: entry['kernel_us']=[float(x)*1000 for x in samples[2000:]]
                    results.append(entry)
                    Path(f'{lang}_{n}_{repetition}_{int(timed)}.log').write_text(proc.stdout+'\nSTDERR\n'+proc.stderr)
        Path('results.json').write_text(json.dumps(results,indent=2))
        print(f'Finished {n} elements',flush=True)
    summary=[]
    for rows,threads in CASES:
        n=rows*threads
        for lang in ('vx','cuda'):
            a=[r for r in results if r['n']==n and r['lang']==lang and not r['timed']]
            b=[r for r in results if r['n']==n and r['lang']==lang and r['timed']]
            row=dict(n=n,lang=lang,kernel_us=statistics.median(statistics.median(r['kernel_us']) for r in b))
            for key in ('host_us','upload_ms','download_ms'): row[key]=statistics.median(r[key] for r in a)
            summary.append(row)
    Path('summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_STACK,(64*1024*1024,resource.getrlimit(resource.RLIMIT_STACK)[1]))
    build()
    run()
