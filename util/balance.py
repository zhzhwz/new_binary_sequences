def balance_rate(s_sequence):
    total_diff = [0] * (len(s_sequence[0]) + 1)
    for seq in s_sequence:
        diff = 0
        for e in seq:
            if e == -1:
                diff -= 1
            else:
                diff += 1
        diff = abs(diff)
        total_diff[diff] += 1
    return total_diff

def distribution_calc(v):
    maxv = max(v)
    ret = [0] * int(maxv + 1)
    for e in v:
        ret[int(e)] += 1
    return ret