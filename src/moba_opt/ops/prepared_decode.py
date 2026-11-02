"""Fixed-address kernels used by PreparedDecode, including CUDA Graph replay."""
import triton
import triton.language as tl


@triton.jit
def _metadata(TABLE,LENGTHS,IDS,POS,DELTA,WIDTH:tl.constexpr,B:tl.constexpr,
              APPEND:tl.constexpr,BP:tl.constexpr):
    row=tl.arange(0,BP)
    sid=tl.load(IDS+row,mask=row<B,other=0)
    logical=tl.load(DELTA+row,mask=row<B,other=-1)
    physical=tl.load(DELTA+B+row,mask=row<B,other=-1)
    tl.store(TABLE+sid.to(tl.int64)*WIDTH+logical,physical,
             mask=(row<B)&(logical>=0))
    length=tl.load(LENGTHS+sid,mask=row<B,other=0)
    if APPEND:
        tl.store(POS+row,length,mask=row<B)
        tl.store(LENGTHS+sid,length+1,mask=row<B)
    else:
        tl.store(POS+row,length-1,mask=row<B)


@triton.jit
def _step_means(KC,TABLE,MEANS,IDS,POS,HK:tl.constexpr,D:tl.constexpr,
                PAGE:tl.constexpr,WIDTH:tl.constexpr,C:tl.constexpr,CP:tl.constexpr,
                MC:tl.constexpr):
    tiles=tl.cdiv(D,32)
    item=tl.program_id(0)//(HK*tiles)
    head=(tl.program_id(0)//tiles)%HK
    ft=tl.program_id(0)%tiles
    sid,pos=tl.load(IDS+item),tl.load(POS+item)
    closed=(pos+1)%C==0
    chunk=pos//C
    token=chunk*C+tl.arange(0,CP)
    d=ft*32+tl.arange(0,32)
    valid=closed&(tl.arange(0,CP)<C)
    page=tl.load(TABLE+sid.to(tl.int64)*WIDTH+token//PAGE,mask=valid,other=0)
    address=((page.to(tl.int64)*PAGE+token%PAGE)[:,None]*HK+head)*D+d[None,:]
    keys=tl.load(KC+address,mask=valid[:,None]&(d[None,:]<D),other=0).to(tl.float32)
    mean=tl.sum(keys,axis=0)/C
    tl.store(MEANS+((sid.to(tl.int64)*MC+chunk)*HK+head)*D+d,mean,mask=closed&(d<D))


@triton.jit
def _selected(VALUES,INDICES,POS,SELECTED,HQ:tl.constexpr,C:tl.constexpr,
              REMOTE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    query=row//HQ
    tl.store(SELECTED+row*(REMOTE+1),tl.load(POS+query)//C)
    if REMOTE:
        slot=tl.arange(0,BLOCK)
        value=tl.load(VALUES+row*REMOTE+slot,mask=slot<REMOTE,other=-float('inf'))
        index=tl.load(INDICES+row*REMOTE+slot,mask=slot<REMOTE,other=-1)
        tl.store(SELECTED+row*(REMOTE+1)+slot+1,
                 tl.where(value>-float('inf'),index,-1).to(tl.int32),mask=slot<REMOTE)


def metadata(table,lengths,ids,positions,delta,append):
    b=ids.numel()
    _metadata[(1,)](table,lengths,ids,positions,delta,table.shape[1],b,append,
                    triton.next_power_of_2(b))


def step_means(kc,table,means,ids,positions,chunk):
    hk,d=kc.shape[2:]
    _step_means[(ids.numel()*hk*triton.cdiv(d,32),)](
        kc,table,means,ids,positions,hk,d,kc.shape[1],table.shape[1],chunk,
        triton.next_power_of_2(chunk),means.shape[1],num_warps=8 if chunk>=512 else 4)


def selected(values,indices,positions,out,chunk):
    remote=out.shape[-1]-1
    _selected[(out.shape[0]*out.shape[1],)](values,indices,positions,out,out.shape[1],
        chunk,remote,triton.next_power_of_2(max(1,remote)))
