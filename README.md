To train on a fresh clone:

1. `pip install -r requirements.txt`
2. `python scripts/fetch_dataset.py`
3. `g++ -std=c++20 -O3 -march=native -ffast-math -fopenmp main.cpp -o main.exe`
4. `./main.exe`
