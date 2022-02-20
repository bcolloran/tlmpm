from re import sub
import subprocess, sys, os

path = os.path.abspath(__file__ + "/..")

quality = 6
time = 20

file_names = sorted([f for f in os.listdir(path) if f[0] == "v"])
print(file_names)
results = []

for file_name in file_names:
    completed_process = subprocess.run(
        ["python", f"{path}/{file_name}", "-q", str(quality), "-t", str(time)],
        capture_output=True,
        text=True,
    )

    stdout_lines = completed_process.stdout.split("\n")
    last_valid_line = [l for l in stdout_lines if l != ""][-1]
    n_particles_line = [l for l in stdout_lines if l.startswith("n_particles")][0]

    results.append((n_particles_line, last_valid_line))

    print("\n-------")
    print(file_name)
    print(n_particles_line)
    print(last_valid_line)
    # print(completed_process.stdout.split("\n"))
    # print(completed_process.stdout.split("\n")[-1])


for file_name, result in zip(file_names, results):
    print("\n-------")
    print(file_name)
    print(result)
