import os
import re
import matplotlib.pyplot as plt

log_directory = "./build/logs"

data = {}


for filename in os.listdir(log_directory):
    if filename.endswith(".log"):
        file_path = os.path.join(log_directory, filename)
        
        with open(file_path, 'r') as file:
            file_data = []
            for line in file:
                match = re.search(r"gflops:\s*([\d.]+)", line)
                if match:
                    file_data.append(float(match.group(1)))
            data[filename] = file_data

plt.figure(figsize=(10, 6))
print(data)
for file, file_data in data.items():
    plt.plot(file_data, label=f'{file}')

plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('gflops')
plt.title('gemm performance')

output_filename = 'gemm_performance.png'
plt.savefig(output_filename)

print(f"performance saved as {output_filename}")