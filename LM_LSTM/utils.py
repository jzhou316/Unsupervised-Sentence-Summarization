import time
import math
import torch


def logging(s, f_log=None, print_=True, log_=True):
    if print_:
        print(s)
    if log_ and f_log is not None:
        f_log.write(s + '\n')

        
def timeSince(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    if h == 0:
        return '%dm %ds' % (m, s)
    else:
        return '%dh %dm %ds' % (h, m, s)

def rand_subvocab(batch, vocab_size, subvocab_size=None):
    if subvocab_size is None or subvocab_size >= vocab_size:
        return None
    batch_ids = torch.cat([batch.text.view(-1), batch.target.view(-1)]).cpu().unique()
    subvocab = torch.cat([torch.randperm(vocab_size)[:subvocab_size], batch_ids]).unique(sorted=True)
    return subvocab

