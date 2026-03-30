import os
import shutil

def fix_missing_bb_table():
    """
    The agent looks for 'bb_discard_table.bin', but the provided file is 'flop_table.bin'.
    This script copies 'flop_table.bin' to 'bb_discard_table.bin' if needed.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'submission67', 'data')
    src = os.path.join(data_dir, 'flop_table.bin')
    dst = os.path.join(data_dir, 'bb_discard_table.bin')
    
    if os.path.exists(src) and not os.path.exists(dst):
        print(f"Copying {src} to {dst} to fix missing table dependency...")
        shutil.copy2(src, dst)
        print("Done.")
    elif os.path.exists(dst):
        print(f"File {dst} already exists. No fix needed.")
    else:
        print(f"Error: Source file {src} not found. Cannot apply fix.")

if __name__ == '__main__':
    fix_missing_bb_table()
