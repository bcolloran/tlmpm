from re import sub
import subprocess, sys, os

path = os.path.abspath(__file__ + "/..")

print(f"{path}/v1_from_paper.py")

quality = 2
time = 6

file_names = ["v0_mls-mpm128_jelly_spinning_beam.py", "v1_from_paper.py"]
results = []

for file_name in file_names:
    completed_process = subprocess.run(
        ["python", f"{path}/{file_name}", "-q", str(quality), "-t", str(time)],
        capture_output=True,
        text=True,
    )

    stdout_lines = completed_process.stdout.split("\n")
    last_valid_line = [l for l in stdout_lines if l != ""][-1]

    results.append(last_valid_line)

    print("-------")
    print(file_name)
    print(last_valid_line)
    # print(completed_process.stdout.split("\n"))
    # print(completed_process.stdout.split("\n")[-1])


for file_name, result in zip(file_names, results):
    print(file_name)
    print(result)
