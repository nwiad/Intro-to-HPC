import re
import argparse

# Mapping of dataset names to the number of non-zero elements.
# nzz = {
#     "arxiv": 169343,
#     "collab": 235868,
#     "citation": 2927963,
#     "ddi": 4267,
#     "protein": 132534,
#     "ppa": 576289,
#     "reddit.dgl": 232965,
#     "products": 2449029,
#     "youtube": 1138499,
#     "amazon_cogdl": 1569960,
#     "yelp": 716847,
#     "wikikg2": 2500604,
#     "am": 881680
# }

nzz = {
    "arxiv": 1166243,
    "collab": 2358104,
    "citation": 30387995,
    "ddi": 2135822,
    "protein": 79122504,
    "ppa": 42463862,
    "reddit.dgl": 114615891,
    "products": 123718280,
    "youtube": 5980886,
    "amazon_cogdl": 264339468,
    "yelp": 13954819,
    "wikikg2": 16109182,
    "am": 5668682
}

# Hardcoded cusparse_info data for k=32.
cusparse_info_32 = {
    "arxiv": 350,
    "collab": 620,
    "citation": 8900,
    "ddi": 250,
    "protein": 8200,
    "ppa": 8800,
    "reddit.dgl": 17000,
    "products": 32000,
    "youtube": 2800,
    "amazon_cogdl": 44000,
    "yelp": 3400,
    "wikikg2": 4400,
    "am": 2300
}

# Hardcoded cusparse_info data for k=256.
cusparse_info_256 = {
    "arxiv": 2500,
    "collab": 4500,
    "citation": 70000,
    "ddi": 1500,
    "protein": 130000,
    "ppa": 80000,
    "reddit.dgl": 160000,
    "products": 250000,
    "youtube": 16000,
    "amazon_cogdl": 400000,
    "yelp": 27000,
    "wikikg2": 25000,
    "am": 13000
}

def process_file(file_path, k_value):
    with open(file_path, 'r') as file:
        content = file.readlines()

    results = []
    throughputs = []
    pattern = re.compile(r'\[\.\.A3/test/test_spmm\.cu:71 \(TestBody\)\] time = (\d+\.\d+) \(double\)')
    dset_pattern = re.compile(r'dset = "(.*?)" \(std::string\)')

    current_dset = None

    for line in content:
        dset_match = dset_pattern.search(line)
        if dset_match:
            current_dset = dset_match.group(1)

        match = pattern.search(line)
        if match and current_dset:
            time_value = float(match.group(1))  # Time in seconds
            nnz_count = nzz.get(current_dset, 0)  # Get the number of non-zero elements for the current dataset
            throughput = nnz_count / time_value if time_value > 0 else 0  # Calculate throughput in nnz/s
            throughputs.append(throughput)  # Collect throughput for average calculation
            time_value *= 1000000
            time_value = int(time_value)
            cusparse_value = cusparse_info_32.get(current_dset, 'N/A') if k_value == 32 else cusparse_info_256.get(current_dset, 'N/A')
            results.append((current_dset, cusparse_value, time_value))

    # Calculate and print the average throughput
    average_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
    print(f'Average Throughput (nnz/s): {average_throughput:.2f}')

    # Print original results without changes
    for result in results:
        print(f'dataset: {result[0]}, cusparse_{k_value}: {result[1]}, time: {result[2]}')

def process_file_cusparse(file_path, k_value):
    with open(file_path, 'r') as file:
        content = file.readlines()

    results = []
    throughputs = []
    pattern = re.compile(r'\[\.\.A3/test/test_spmm\.cu:62 \(TestBody\)\] time = (\d+\.\d+) \(double\)')
    dset_pattern = re.compile(r'dset = "(.*?)" \(std::string\)')

    current_dset = None

    for line in content:
        dset_match = dset_pattern.search(line)
        if dset_match:
            current_dset = dset_match.group(1)

        match = pattern.search(line)
        if match and current_dset:
            time_value = float(match.group(1))  # Time in seconds
            nnz_count = nzz.get(current_dset, 0)  # Get the number of non-zero elements for the current dataset
            throughput = nnz_count / time_value if time_value > 0 else 0  # Calculate throughput in nnz/s
            throughputs.append(throughput)  # Collect throughput for average calculation
            time_value *= 1000000
            time_value = int(time_value)
            cusparse_value = cusparse_info_32.get(current_dset, 'N/A') if k_value == 32 else cusparse_info_256.get(current_dset, 'N/A')
            results.append((current_dset, cusparse_value, time_value))

    # Calculate and print the average throughput
    average_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
    print(f'Average Throughput (nnz/s): {average_throughput:.2f}')

    # Print original results without changes
    for result in results:
        print(f'dataset: {result[0]}, cusparse_{k_value}: {result[1]}, time: {result[2]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a file to extract and multiply time values.')
    parser.add_argument('-f', type=str, help='Path to the input file')
    parser.add_argument('-k', type=int, choices=[32, 256], help='Choose the k value to use (32 or 256)')
    args = parser.parse_args()
    print(f'Processing file: {args.f}, k value: {args.k}')
    print("spmm_opt.cu: ")
    process_file(args.f, args.k)
    print("spmm_cusparse.cu: ")
    process_file_cusparse(args.f, args.k)
