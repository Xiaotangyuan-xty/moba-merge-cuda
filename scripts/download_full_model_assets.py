"""CPU-only, serial, rate-limited evaluation assets. No executable model code."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION='060db6499f32faf8b98477b0a26969ef7d8b9987'


def fetch(url,path,rate):
    tmp=path.with_suffix(path.suffix+'.part')
    subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','15',
                    '--max-time','600','--retry','2','--limit-rate',rate,url,'--output',str(tmp)],check=True)
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--manifest')
    p.add_argument('--rate',default='8M')
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=Path(args.manifest) if args.manifest else out/'model_manifest.json'
    if not args.manifest:fetch(f'https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B/revision/{REVISION}?blobs=true',manifest,args.rate)
    meta=json.loads(manifest.read_text());revision=meta['sha']
    if revision!=REVISION:raise ValueError('unexpected model revision')
    allow=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','model.safetensors']
    files={x['rfilename']:x for x in meta['siblings']}
    model=out/'qwen2.5-0.5b';model.mkdir(exist_ok=True)
    verified={}
    for name in allow:
        info=files[name];path=model/name
        if not path.exists():
            print('download',name,info.get('size'),flush=True)
            fetch(f'https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/{revision}/{name}?download=true',path,args.rate)
        if path.stat().st_size!=info['size']:raise ValueError('size mismatch: '+name)
        sha=hashlib.sha256();git=hashlib.sha1(b'blob '+str(info['size']).encode()+b'\0')
        with path.open('rb') as f:
            while block:=f.read(4*1024*1024):sha.update(block);git.update(block)
        if 'lfs' in info:
            if sha.hexdigest()!=info['lfs']['sha256']:raise ValueError('LFS checksum mismatch: '+name)
        elif git.hexdigest()!=info['blobId']:raise ValueError('Git blob mismatch: '+name)
        verified[name]=dict(bytes=info['size'],sha256=sha.hexdigest())
        print('verified',name,flush=True)
    (out/'asset_manifest.json').write_text(json.dumps(dict(model='Qwen/Qwen2.5-0.5B',revision=revision,files=verified),indent=2)+'\n')


if __name__=='__main__':main()
