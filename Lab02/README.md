# Task 3 - Echo video

**How to run**
1. `pip install -r ../requirements.txt`
2. Open `realtime_echo.ipynb` from this folder and run all cells.
3. A window opens with raw (left) and enhanced (right). Press `q` to quit.
4. If your PC has no display, run with `HEADLESS=1` and check the `output/` folder.

Each frame goes through: histogram equalization -> JET -> color balance -> log -> gamma 0.6, then it is joined with the raw frame using `np.hstack`. Screenshots and a saved video are in `output/`.
