taskset -c 0 python writer_flatcurve.py --idx 0 &
taskset -c 1 python writer_flatcurve.py --idx 1 &
taskset -c 2 python writer_flatcurve.py --idx 2 &
taskset -c 3 python writer_flatcurve.py --idx 3 &
taskset -c 4 python writer_flatcurve.py --idx 4 &
taskset -c 5 python writer_flatcurve.py --idx 5 &
wait
taskset -c 0 python writer_flatcurve.py
