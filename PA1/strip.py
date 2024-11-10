import sys

file = sys.argv[1]
striped_lines = []
with open(file, 'r') as f:
    while line:=f.readline():
        split_line = line.split()
        if len(split_line) > 0:
            if split_line[0] in ["Process", "Running", "MPI_Init", "Main"]:
                continue
            if "pass" in split_line:
                continue
        striped_lines.append(line)
with open(file, 'w') as f:
    f.writelines(striped_lines)
