To train on a fresh clone:

1. `pip install -r requirements.txt`
2. `python scripts/fetch_dataset.py`
3. `g++ -std=c++20 -O3 -march=native -ffast-math -fopenmp -static-libgcc -static-libstdc++ main.cpp -o main.exe`
   (the `-static-lib*` flags matter on this machine: `PATH` has Git for Windows' bundled `libstdc++-6.dll` ahead of MSYS2 UCRT64's, and the two are ABI-incompatible -- without static linking the binary loads the wrong DLL at runtime and segfaults inside iostream internals, non-deterministically, well past the point where the actual bug would be.)
4. `./main.exe`
