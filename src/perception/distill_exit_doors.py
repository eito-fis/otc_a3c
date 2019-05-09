import os

dir = '/Volumes/Storage'
exit_dir = dir+'/exit_door'
not_exit_dir = dir+'/not_exit_door'

os.makedirs(not_exit_dir, exist_ok=True)

files = os.listdir(exit_dir)

good = 0
bad = 0
all = 0

for f in files:
    passed, floor, steps, episode, step_i = f.replace('.png', '').split('_')
    passed = passed == 'pass'
    floor = int(floor.replace('floor',''))
    steps = int(steps.replace('steps',''))
    episode = int(episode.replace('episode',''))
    step_i = int(step_i)

    if steps - step_i > 10 or not passed:
        os.rename(exit_dir+'/'+f,not_exit_dir+'/'+f)
        bad += 1
    else:
        good += 1
    all += 1

    print('\r{}/{} {:.2f}% {} {:.2f}% good {} {:.2f}% bad'.format(
        all,len(files),all*100./len(files),
        good, good*100./all, bad, bad*100./all,
    ),end='')
print('\nDone')
