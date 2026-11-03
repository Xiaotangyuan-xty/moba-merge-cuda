"""Predeclared synthetic passkey tasks; no selection based on model answers."""
import hashlib
import random
import struct


def retrieval_cases(tokenizer,seed=20261102,lengths=(512,2048,4096)):
    rng=random.Random(seed)
    pad=tokenizer.encode('Ordinary memory notes discuss processors, buffers, queues, files and arithmetic.\n',add_special_tokens=False)
    cases=[]
    for length in lengths:
        for fraction in (0.25,0.75):
            key='K'+str(rng.randrange(10000,99999));answer=str(rng.randrange(1000,9999))
            if answer=='1290':answer='1291'
            head=tokenizer.encode('Example: Record A17 has access code 1290.\nQuestion: What is the access code for A17?\nAnswer: 1290.\n\nRead the database below. Answer the final question with only the four-digit code.\n',add_special_tokens=False)
            needle=tokenizer.encode(f'Record {key} has access code {answer}.\n',add_special_tokens=False)
            tail=tokenizer.encode(f'\nEnd of database.\nQuestion: What is the access code for {key}?\nAnswer:',add_special_tokens=False)
            remaining=length-len(head)-len(needle)-len(tail)
            if remaining<0 or not pad:raise ValueError('task template exceeds length')
            before=min(remaining,max(0,int(length*fraction)-len(head)))
            def filler(n):return (pad*(n//len(pad)+1))[:n]
            tokens=head+filler(before)+needle+filler(remaining-before)+tail
            assert len(tokens)==length
            cases.append(dict(id=f'passkey_{length}_{int(fraction*100)}',length=length,key=key,answer=answer,
                needle_token_position=len(head)+before,tokens=tokens,
                token_sha256=hashlib.sha256(b''.join(struct.pack('<I',t) for t in tokens)).hexdigest()))
    return cases
