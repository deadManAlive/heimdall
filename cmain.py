from cmainlib import run
import multiprocessing as mp
from main import server

if __name__ == "__main__":
    total_down = mp.Value("i", 0)
    total_up = mp.Value("i", 0)

    p1 = mp.Process(target=run, args=(total_down, total_up, ))
    p2 = mp.Process(target=server, args=(total_down, total_up, ))

    p1.start()
    p2.start()
