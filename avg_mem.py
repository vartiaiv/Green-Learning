from glob import glob

if __name__ == "__main__":
    filenames = glob("mprofile_*")
    newest_file = sorted(filenames)[-1]
    mems_sum = 0
    count = 0
    print("opening:", newest_file)
    with open(newest_file) as f:
        lines = f.readlines()[1:] # exclude header
        for ll in lines:
            parts = ll.split(" ")
            mems_sum += float(parts[1])
            count += 1
    avg_mem = mems_sum/count
    print(avg_mem)
    
