import zstandard as zstd
import pickle
import numpy as np
import os

def load_fft_data(path_to_files, each_file_seconds = 10, num_channels = 3764, freq_range = [0,5_000], verbose=False):
    handler = ZstdArrayHandler(verbose=verbose)
    
    
    num_files=len(os.listdir(path_to_files))
    data_fft = np.zeros((each_file_seconds*num_files, num_channels, freq_range[1]))
    
    count = 0
    for file in os.listdir(path_to_files):
        file_loaded = handler.load(path_to_files+file)[:, :, freq_range[0]:freq_range[1]]
        data_fft[count:count+each_file_seconds] = file_loaded
        count+=each_file_seconds

    return data_fft

def load_fft_data_2dim(path_to_files, each_file_seconds = 1, num_channels = 3764, freq_range = [0,5_000], verbose=False):
    handler = ZstdArrayHandler(verbose=verbose)
    
    
    num_files=len(os.listdir(path_to_files))
    data_fft = np.zeros((num_files, num_channels, freq_range[1]))
    
    count = 0
    for file in os.listdir(path_to_files):
        file_loaded = handler.load(path_to_files+file)[:, freq_range[0]:freq_range[1]]
        data_fft[count:count+each_file_seconds] = file_loaded
        count+=each_file_seconds

    return data_fft

class ZstdArrayHandler:
    def __init__(self, verbose=False, compression_level=3):
        """
        Initializes the ZstdArrayHandler with a specific compression level.
        Args:
            compression_level (int): Compression level for Zstandard (1–22). Default is 3.
        """
        self.compressor = zstd.ZstdCompressor(level=compression_level)
        self.decompressor = zstd.ZstdDecompressor()
        self.verbose = verbose

    def save(self, array, filepath):
        """
        Compresses and saves a numpy array to a file.
        Args:
            array (np.ndarray): The numpy array to be saved.
            filepath (str): The path to the file where the array will be saved.
        """
        # Serialize and compress
        compressed_data = self.compressor.compress(pickle.dumps(array))
        # Write to file
        with open(filepath, 'wb') as f:
            f.write(compressed_data)
        print(f"Array saved and compressed to {filepath}")

    def load(self, filepath):
        """
        Loads and decompresses a numpy array from a file.
        Args:
            filepath (str): The path to the file from which to load the array.
        Returns:
            np.ndarray: The decompressed numpy array.
        """
        # Read from file
        with open(filepath, 'rb') as f:
            compressed_data = f.read()
        # Decompress and deserialize
        array = pickle.loads(self.decompressor.decompress(compressed_data))
        if self.verbose:
            print(f"Array loaded and decompressed from {filepath}")
        return array