import time
import math


def timeSince(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    if h == 0:
        return '%dm %.3fs' % (m, s)
    else:
        return '%dh %dm %.3fs' % (h, m, s)
