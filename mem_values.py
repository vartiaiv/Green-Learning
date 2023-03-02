from glob import glob

if __name__ == "__main__":
    filenames = glob("mprofile_*.dat")
    newest_file = sorted(filenames)[-1]
    for filename in filenames:
        mems = []
        with open(filename) as f:
            lines = f.readlines()[1:] # exclude header
            for ll in lines:
                parts = ll.split(" ")
                mems.append(float(parts[1]))
        avg_mem = sum(mems)/len(mems)
        max_mem = max(mems)
        print(f"{filename}")
        # print(f"- RAM_peak: {max_mem:.2f} MiB\nRAM_avg: {avg_mem:.2f} MiB")
        conversion = 2**20 / 10**6 # 1 MiB is equal to this amount of MB
        print(f"- RAM_peak: {conversion*max_mem:.2f} MB\n- RAM_avg: {conversion*avg_mem:.2f} MB")
        
    
