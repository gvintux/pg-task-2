from common.values import Values

if __name__ == '__main__':
    print('Testing task_1.common.values ...')
    v = Values()
    v_copy = v.copy()
    v_copy['h'] += 1
    assert v['h'] != v_copy['h'], 'Copy must be independent'
    prev_D = v['D']
    v['E'] += 1
    assert v['D'] != prev_D, 'D must be recalculated when E assigned'
    prev_D = v['D']
    v['h'] += 1
    assert v['D'] != prev_D, 'D must be recalculated when h assigned'
    prev_D = v['D']
    v['mu'] += 1
    assert v['D'] != prev_D, 'D must be recalculated when mu assigned'
    prev_v = v['v']
    v['g'] += 1
    assert v['v'] != prev_v, 'v must be recalculated when g assigned'
    prev_v = v['v']
    v['H'] += 1
    assert v['v'] != prev_v, 'v must be recalculated when H assigned'

    v = Values(forced=True)
    prev_D = v['D']
    v['E'] += 1
    assert v['D'] == prev_D, 'D must not be recalculated when E assigned in forced mode'
    v['h'] += 1
    assert v['D'] == prev_D, 'D must not be recalculated when h assigned in forced mode'
    v['mu'] += 1
    assert v['D'] == prev_D, 'D must not be recalculated when mu assigned in forced mode'
    prev_v = v['v']
    v['g'] += 1
    assert v['v'] == prev_v, 'v must not be recalculated when g assigned in forced mode'
    prev_v = v['v']
    v['H'] += 1
    assert v['v'] == prev_v, 'v must not be recalculated when H assigned in forced mode'

    print('Test finished')
