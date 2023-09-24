import torch
import numpy as np

# entry point for the program
if __name__ == '__main__':
    # create the process pool

    cos = torch.nn.CosineSimilarity(dim=0)
    krum_folder = 'krum'
    flag_folders = ['0.0','0.5', '1.0', '10.0', '100.0']
    krum_file_arr = []
    flag_file_arr = []
    cos = torch.nn.CosineSimilarity(dim=0)
    for flag_folder in flag_folders:
        similarities = []
        for i in range(100):
            krum_file = "Y_krum_" + str(i) + ".pt"
            flag_file = "Y_flag_median_" + flag_folder  + "_" +str(i) + ".pt"
            krum_path = krum_folder + "/" + krum_file
            flag_path = flag_folder + "/" + flag_file
            krum_tensor = torch.load(krum_path)
            flag_tensor = torch.load(flag_path)
            flag_tensor = flag_tensor.flatten()
            output = cos(flag_tensor, krum_tensor)
            similarity = output.detach().cpu().numpy()
            similarities.append(similarity)
        np.save('similarities_' + flag_folder + '.npy', similarities)


