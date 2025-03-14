# sorting olympics

yet another sorting analysis toolkit. because the world needed more sorting analysis.


## tools

### benchmark.py
measures how painfully slow your sorting algorithm really is.
```bash
TIMEOUT=5 python benchmark.py
```
<img width="1488" alt="tournament example" src="https://github.com/user-attachments/assets/1739ac4e-6a28-443f-8905-91208949e4ad" />


### visualize.py
watch sorting happen in real-time. riveting entertainment.
```bash
python DELAY=10 SIZE=100 visualize.py all
```
```bash
python visualize.py bubble_sort
```

### run.py
pits algorithms against each other. may the least awful one win.
```bash

SIZE=100 TIMEOUT=5 python run.py

# with benchmark and visualisation
BENCH=1 VIZ=1 DATA_TYPE=random TEST_RANGE=1000 python run.py
```
