import json
import glob
import numpy as np

turns_files = glob.glob("data/turns/*.json")
caller_durations = []

for f in turns_files:
    with open(f, 'r') as file:
        data = json.load(file)
        for t in data.get('turns', []):
            if t['channel'] == 0:
                caller_durations.append(t['end'] - t['start'])

if caller_durations:
    durations = np.array(caller_durations)
    print("Percentiles (10, 25, 50, 75, 90):")
    print(np.percentile(durations, [10, 25, 50, 75, 90]))
    print("\nMenores a:")
    for thresh in [0.5, 1.0, 1.5, 2.0]:
        count = np.sum(durations < thresh)
        pct = (count / len(durations)) * 100
        print(f"  < {thresh}s: {count} ({pct:.1f}%)")
else:
    print("No se encontraron archivos turns.json o turnos del caller")
