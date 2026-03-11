import os, glob
files = glob.glob('data/movienet_subset/annotation/*.json')
print('total', len(files))
print('nonempty', len([f for f in files if os.path.getsize(f) > 1000]))
print([os.path.basename(f) for f in files if os.path.getsize(f) > 1000])
