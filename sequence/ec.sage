from sage.all import *
import time
import logging
from util.debug import log_mem

class EC:
    def step1(self, n):
        logging.debug('Entering step 1...')
        q = 2 ^ n
        return q


    def step2(self, q, order_range, useAllEC=False):
        logging.debug('Entering step 2...')
        lbound, rbound = order_range
        F.<g> = GF(q, 'g')
        for a2 in F:
            for a6 in F:
                try:
                    E = EllipticCurve(F, [1, 0, a2, 0, a6])
                except ArithmeticError:
                    continue
                if lbound <= E.order() and E.order() <= rbound:
                    return F, g, E
        # for a3 in F:
        #     for a4 in F:
        #         for a6 in F:
        #             try:
        #                 E = EllipticCurve(F, [0, a3, 0, a4, a6])
        #             except ArithmeticError:
        #                 continue
        #             if E.order() == q:
        #                 return F, g, E
        if useAllEC:
            for a1 in F:
                 for a2 in F:
                     for a3 in F:
                          for a4 in F:
                                for a6 in F:
                                    try:
                                        E = EllipticCurve(F, [a1, a3, a2, a4, a6])
                                    except ArithmeticError:
                                        continue
                                    if lbound <= E.order() and E.order() <= rbound:
                                        return F, g, E
        raise ValueError("Can't find a elliptic curve with exactly q points!")


    def step3(self, E):
        logging.debug('Entering step 3...')
        P = None
        for e in E:
            if e.order() == E.order():
                P = e
        if P is None:
            raise ValueError("Can't find a generator P!")
        P_list = [E(0), P]
        now = P
        for _ in range(2, E.order()):
            now = now + P
            P_list.append(now)
        return P_list


    def step4(self):
        logging.debug('Skipping step 4...')


    def step5(self, F, a2, a6):
        logging.debug('Entering step 5...')
        R.<x> = F['x']
        for a in F:
            for b in F:
                p = x ^ 2 + a * x + b
                if not p.is_irreducible():
                    continue
                F2.<alpha> = F.extension(p)
                R2.<y> = F2['y']
                func = y ^ 2 + alpha * y - alpha ^ 3 - a2 * alpha - a6
                if func.is_irreducible():
                    continue
                root = func.roots()
                beta = root[0][0]
                beta2 = root[1][0]
                if beta == beta2:
                    continue
                return R, x, p, a, b, F2, alpha, beta, beta2
        print('Error in step 5!')
        exit()


    def step6(self, beta):
        logging.debug('Entering step 6...')
        beta_coe = beta.list()
        u = 1
        v = -beta_coe[1]
        w = -beta_coe[0]
        return u, v, w


    def step7(self):
        logging.debug('Skipping step 7...')


    def step8(self, F, P_list, E, u, v, w, a, b):
        logging.debug('Entering step 8...')
        log_mem('ecc 8 begin')
        s_sequence = []
        for alpi in F:
            if alpi == 0:
                continue
            tmp_list = []
            for P in P_list:
                if P == E(0):
                    tmp_list.append(1)
                else:
                    x, y = P.xy()
                    tmp = alpi * (u * y + v * x + w) * (x ^ 2 + a * x + b) ^ (-1)
                    if tmp.trace() == 0: 
                        tmp_list.append(1r)
                    else:
                        tmp_list.append(-1r)
            s_sequence.append(tmp_list)
        log_mem('ecc 8 end')
        return s_sequence


    def gen_of_order(self, n, order_range, file=None, useAllEC=False):
        logging.debug('Start geneating sequence using ecc method...')
        time_start = time.time()
        q = self.step1(n)
        if file: print('Field order:', q, file=file)
        F, g, E = self.step2(q, order_range, useAllEC)
        if file: print(E, file=file)
        if file: print('Elliptic curve order:', E.order(), file=file)
        P_list = self.step3(E)
        if file: print('rational points:', file=file)
        if file: print('\n'.join([str(e) for e in P_list]), file=file)
        if file: print('generator point:', P_list[1], file=file)
        self.step4()
        R, x, p, a, b, F2, alpha, beta, beta2 = self.step5(F, E.a2(), E.a6())
        if file: print('irreducible polynomial over F(step 5):', p, file=file)
        if file: print(F2, file=file)
        u, v, w = self.step6(beta)
        if file: print('step6:', file=file)
        if file: print('u =', u, file=file)
        if file: print('v =', v, file=file)
        if file: print('w =', w, file=file)
        self.step7()
        s_sequence = self.step8(F, P_list, E, u, v, w, a, b)
        if file: print('sequence:', file=file)
        if file: print('\n'.join(['\t'.join([str(x) for x in e]) for e in s_sequence]), file=file)
        time_end = time.time()
        log_mem('ecc end')
        logging.debug('Time used to generate ecc sequence of n = {}: {}'.format(n, time_end - time_start))
        return s_sequence, E.order()
    
    def gen(self, n):
        q = 2 ^ n
        s_sequence, _ = self.gen_of_order(n, [q, q])
        return s_sequence
