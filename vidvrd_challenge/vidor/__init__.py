# import time
# def p(i):
#     print(i)
#     time.sleep(2)
#     return i
#
#
# from multiprocessing.pool import Pool as Pool
# pool = Pool()
# results = [pool.apply_async(p, args=(v,)) for v in range(100)]
# pool.close()
# pool.join()
#
# v = [res.get() for res in results]
# print(v)