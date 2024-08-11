"""!
@file
@brief The EDF to HDF5 file converter. we generate EDF files,
        Extract data and information and then save them in HDF5 Format.
@author Mahdi
"""
try:
    import mne
    from mne._fiff.meas_info import Info
    import os
    import h5py
    import numpy as np
    import datetime
    from joblib import Memory
    import pyedflib as pf
    from tqdm import tqdm
    import os
except ImportError as e:
    print(f"Error: {e}")
    exit(1)

class ExtractTimeOfRecords:
    """!
    @brief Generates timestapms dataset.

    Methods:
    - `__init__(self)`: Initializes the ExtractTimeOfRecords class.
    - `run(self, ch_data, sf : int, ch_number : int, start_time)`: Executes the preparing function for timestamps.
    """
        
    def __init__(self):
        """!
        @brief Initializes the ExtractTimeOfRecords class.

        @param timestamp_type (np.dtype): The defination of the data type that we want to use to save our timestamps dataset.
        """
        self.timestamp_type = np.dtype([('index', np.int64), ('value', np.float64)]) #, ('timestamp', h5py.string_dtype(encoding='utf-8')

    def run(self, ch_data, sf : int, ch_number : int, start_time):
        """!
        @brief Generating the timetamps data set and return with the datatype that we defined before.

        @param ch_data (EDF file): The array of the values of the specific signal.
        @param sf (int): The sample frequency. It means how many sample we have for each second.
        @param ch_number (int): The index of the signal in EDF raw file.
        @param start_time (datetime): The start time of recording. we just use it when we want to generate the exact date and time of each sample.
        """

        print(f"\n number of samples (timestamps) in channel({ch_number}): {len(ch_data)} ")

        timestamps_dataset = []
        time_for_each_sample = 1 / sf
        sample_time = 0
        for i in range(len(ch_data)):  # len(ch_data)
            sample_time += time_for_each_sample
            #date_record = start_time + datetime.timedelta(seconds=sample_time)
            #time_string = date_record.strftime("%Y-%m-%d %H:%M:%S.%f")
            timestamps_dataset.append((i, sample_time)) #(i, sample_time, time_string)

        structured_data = np.array(timestamps_dataset, dtype=self.timestamp_type)
        return structured_data
    
class ExtractInfoFromEDF:
    """!
    @brief Extract information from EDF headers and export it as a descriptor dataset.

    Methods:
    - `__init__(self)`: Initializes the ExtractInfoFromEDF class.
    - `run(self, edf_info)`: Executes the extracting function and returns the descriptor values based on the defined data type.
    """
    def __init__(self) -> None:
        """!
        @brief Initializes the ExtractTimeOfRecords class.

        @param main_header_ds_type (np.dtype): The defination of the data type that we want to use to save the descriptor of the EDF header.
        @param signal_header_ds_type (np.dtype): The defination of the data type that we want to use to save our Signals header.
       
        """
        # Define main header data type
        self.main_header_ds_type = np.dtype([
            ('channel_names', h5py.string_dtype(encoding='utf-8')),            
            ('channel_numbers', np.uint16),
            ('channel_types', h5py.string_dtype(encoding='utf-8')),          
            ('sample_freq', np.uint16),
            ('lowpass', np.uint16),
            ('highpass', np.uint16),
            ('date_of_record', h5py.string_dtype(encoding='utf-8')),          
            ('patient_information', h5py.string_dtype(encoding='utf-8'))  # adjusted variable name
        ])
        
        # Define signal header data type
        self.signal_header_ds_type = np.dtype([
            ('label', h5py.string_dtype(encoding='utf-8')),  
            ('dimension', h5py.string_dtype(encoding='utf-8')),     
            ('sample_rate', np.uint16),
            ('sample_frequency', np.uint16),
            ('physical_max', np.uint16),
            ('physical_min', np.uint16),
            ('digital_max', np.uint16),
            ('digital_min', np.uint16),
            ('prefilter', h5py.string_dtype(encoding='utf-8')),     
            ('transducer', h5py.string_dtype(encoding='utf-8'))     
        ])

    def run(self, edf_info):
        """!
        @brief Executes the extracting functions and returns the descriptor values based on the defined data type.

        @param edf_info (EDF.info): The information of the EDF file that contains the Headers. maybe EDF header or signals header.       
        """
        if isinstance(edf_info, Info):
            non_empty_attr_edf = {}
            ch_types = set()

            for k, v in edf_info.items():
                if k == "ch_names": 
                    if v:
                        non_empty_attr_edf['channel_names'] = ','.join(name for name in v)
                    else:
                        non_empty_attr_edf['channel_names'] = "[]"  # always show
                elif k == "nchan":
                    if v:
                        non_empty_attr_edf['channel_numbers'] = v
                    else:
                        non_empty_attr_edf['channel_numbers'] = 0
                elif k == "chs":
                    if v:
                        for i in range(len(v)):
                            ch_type = str(edf_info['chs'][i]['kind'])
                            space_index = ch_type.find(' ')
                            ch_type = ch_type[space_index + 1:]
                            ch_types.add(ch_type)
                        non_empty_attr_edf['channel_types'] = ','.join(name for name in ch_types)
                    else:
                        non_empty_attr_edf['channel_types'] = ['']
                elif k in ["lowpass", "highpass"]:
                    non_empty_attr_edf[k] = v
                elif k == "sfreq":
                    non_empty_attr_edf["sample_freq"] = v
                elif k == "meas_date":
                    if v is None:
                        non_empty_attr_edf['date_of_record'] = "unspecified"
                    else:
                        non_empty_attr_edf['date_of_record'] =  v.strftime("%Y-%m-%d %H:%M:%S %Z")
                elif k == "subject_info":
                    if v:
                        non_empty_attr_edf['patient_information'] = v
                    else:
                        non_empty_attr_edf['patient_information'] = "unspecified"
        
            structured_data = np.array([
                (
                    non_empty_attr_edf["channel_names"],
                    non_empty_attr_edf["channel_numbers"],
                    non_empty_attr_edf["channel_types"],
                    non_empty_attr_edf["sample_freq"],
                    non_empty_attr_edf["lowpass"],
                    non_empty_attr_edf["highpass"],
                    non_empty_attr_edf["date_of_record"],
                    str(non_empty_attr_edf['patient_information'])
                )
            ], dtype=self.main_header_ds_type)
         
            return structured_data 

        elif isinstance(edf_info, dict):
            structured_data_signal = np.array([
                (
                    edf_info['label'],
                    edf_info['dimension'],
                    edf_info['sample_rate'],
                    edf_info['sample_frequency'],
                    edf_info['physical_max'],
                    edf_info['physical_min'],
                    edf_info['digital_max'],
                    edf_info['digital_min'],
                    edf_info['prefilter'],
                    edf_info['transducer']
                )
            ], dtype=self.signal_header_ds_type)

            return structured_data_signal


class EDFToH5Converter:
    """!
    @brief Extract information from EDF file and generate the HDF5 file that contains velues and descriptors.

    Methods:
    - `__init__(self)`: Initializes the EDFToH5Converter class.
    - `load_data(self, edf_file)`: Data is loaded into the cache to reduce runtime.
    - `convert_edf_to_hdf5(self, edf_file, hdf5_file_path)`: Executes the extracting functions and creat HDF5 file with datasets and groups.
    """

    def __init__(self) -> None:
        
        pass

    memory = Memory(location='./cache', verbose=0)
    @memory.cache
    def load_data(self, edf_file_path):
        """!
        @brief Data is loaded into the cache to reduce runtime.

        @param edf_file_path (string): The path of the EDF file.       
        """
        raw_edf_mne = mne.io.read_raw_edf(edf_file_path)
        raw_edf_py = pf.EdfReader(edf_file_path)
        return raw_edf_mne, raw_edf_py

    def convert_edf_to_hdf5(self, edf_file_path, hdf5_file_path):
        """!
        @brief Executes the extracting functions and creat HDF5 file with datasets and groups.

        @param edf_file_path (string): The path of the EDF file.   
        @param hdf5_file_path (string): The destination path where we intend to save the HDF5 file.    
        """
        # Load the EDF file
        raw_edf_mne, raw_edf_py = self.load_data(self,edf_file_path)
        description_extractor = ExtractInfoFromEDF()
        timestamps_extractor = ExtractTimeOfRecords()
        start_time = raw_edf_py.getStartdatetime()
        # Create new HDF5 file
        with h5py.File(hdf5_file_path, "w") as new_file:
            # Add EDF info to HDF5 file as an attribute
            extracted_data = description_extractor.run(
                raw_edf_mne.info.copy())
            edf_description = new_file.create_dataset(
                'EDF_description', data=extracted_data)

            # Create group for each signal name and then export the raw data into the specific signal group.
            for sig_number in tqdm(range(raw_edf_py.signals_in_file)):
                ch_physical_values = raw_edf_py.readSignal(sig_number)
                ch_digital_values = raw_edf_py.readSignal(sig_number,digital=True)
                ch_name = "channel_{:04d}".format(sig_number)
                ch_description = description_extractor.run(
                    raw_edf_py.getSignalHeader(sig_number))
                """ch_times = timestamps_extractor.run(         # for calculating timestamps
                    ch_data, raw_edf_py.getSampleFrequency(sig_number), sig_number, start_time)"""

                ch_ph_datasets = new_file.create_dataset(
                    f'channels/{ch_name}/physical_values', data=ch_physical_values)
                ch_di_datasets = new_file.create_dataset(
                    f'channels/{ch_name}/digital_values', data=ch_digital_values)
                ch_descriptions = new_file.create_dataset(
                    f'channels/{ch_name}/descriptor', data=ch_description)
                """ch_timestamps = new_file.create_dataset(     # for calculating timestamps
                    f'channels/{ch_name}/timestamps', data=ch_times)"""

if __name__ == "__main__":
    # Provide paths for your EDF and HDF5 files and call the methods.

    folder_edf_path = input("\nEDF foder path that contains EDF files: ") #/home/mt/Projects/internship/a"
    folder_hdf5_path = input("\nHDF5 foder path to export HDF5 files: ") #"/home/mt/Projects/internship"

    for file_name in tqdm(os.listdir(folder_edf_path)):
        if file_name.endswith('.edf'):
            edf_file_path = os.path.join(folder_edf_path,file_name)
            print("\n======================================")
            print("\nProcessing file : ", edf_file_path)
            print("\n======================================\n")
            
            hdf5_file_name = file_name + ".h5"
            hdf5_path = os.path.join(folder_hdf5_path,hdf5_file_name)

            edf_h5 = EDFToH5Converter()
            edf_h5.convert_edf_to_hdf5(edf_file_path, hdf5_path)
