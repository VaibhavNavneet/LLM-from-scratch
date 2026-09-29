import torch

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=1.0, top_k=None, top_p=None, device=None):
    model.eval(); device=device or next(model.parameters()).device
    ids=torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)],dtype=torch.long,device=device)
    for _ in range(max_new_tokens):
        logits,_=model(ids[:,-model.cfg.context_length:]); logits=logits[:,-1,:]/max(temperature,1e-8)
        if top_k:
            v,_=torch.topk(logits,min(top_k,logits.size(-1))); logits[logits<v[...,-1,None]]=-float("inf")
        probs=torch.softmax(logits,dim=-1)
        if top_p:
            sorted_p, sorted_i=torch.sort(probs,descending=True); remove=(torch.cumsum(sorted_p,dim=-1)-sorted_p)>top_p; probs.scatter_(1,sorted_i,sorted_p.masked_fill(remove,0)); probs/=probs.sum(dim=-1,keepdim=True)
        nxt=torch.argmax(probs,dim=-1,keepdim=True) if temperature<=0 else torch.multinomial(probs,1)
        ids=torch.cat([ids,nxt],dim=1)
        if nxt.item()==tokenizer.eos_id: break
    return tokenizer.decode(ids[0].tolist())
