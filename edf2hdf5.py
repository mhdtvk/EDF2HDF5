"""!
@file
@brief The EDF to HDF5 file converter. we generate EDF files,
        Extract data and information and then save them in HDF5 Format.
@author Mahdi
"""
try:
    import os
    import mne
    import h5py
    import datetime
    import numpy as np
    import pyedflib as pf
    from mne._fiff.meas_info import Info
    from joblib import Memory
    from tqdm import tqdm
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
        self.timestamp_type = np.dtype([('index', np.int64), ('value', np.float64), ('timestamp', h5py.string_dtype(encoding='utf-8'))])

    def run(self, smp_number, smp_freq : int, ch_number : int, start_time : datetime = None):
        """!
        @brief Generating the timetamps data set and return with the datatype that we defined before.
        @param ch_data (EDF file): The array of the values of the specific signal.
        @param sf (int): The sample frequency. It means how many sample we have for each second.
        @param ch_number (int): The index of the signal in EDF raw file.
        @param start_time (datetime): The start time of recording. we just use it when we want to add the exact date and time of each sample is recorded to timestamp dataset.
        """
        print(f"\n number of samples (timestamps) in channel({ch_number}): {smp_number} ")
        tstamp_dset = []
        smp_duration = 1 / smp_freq
        smp_time = 0                    # Recording time of each sample

        if not start_time :
            for i in range(smp_number):  # smp_number: Is the number of samples that we add to our dataset.
                smp_time += smp_duration
                tstamp_dset.append((i, smp_time)) 
        else :
            for i in range(smp_number):  
                smp_time += smp_duration
                date_record = start_time + datetime.timedelta(seconds=smp_time)
                time_string = date_record.strftime("%Y-%m-%d %H:%M:%S.%f")
                tstamp_dset.append((i, smp_time, time_string)) 

        structured_data = np.array(tstamp_dset, dtype=self.timestamp_type)
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
        @param edf_info (EDF.info or dict): The information of the EDF file that contains the Headers.       
        """
        if isinstance(edf_info, Info):
            return self.process_main_header(edf_info)
        elif isinstance(edf_info, dict):
            return self.process_signal_header(edf_info)

    def process_main_header(self, edf_info):
        edf_attr = {}
        ch_types = set()
        for k, v in edf_info.items():
                if k == "ch_names": 
                    if v:
                        edf_attr['channel_names'] = ','.join(name for name in v)
                    else:
                        edf_attr['channel_names'] = "[]"  # always show
                elif k == "nchan":
                    if v:
                        edf_attr['channel_numbers'] = v
                    else:
                        edf_attr['channel_numbers'] = 0
                elif k == "chs":
                    if v:
                        for i in range(len(v)):
                            ch_type = str(edf_info['chs'][i]['kind'])
                            space_index = ch_type.find(' ')
                            ch_type = ch_type[space_index + 1:]
                            ch_types.add(ch_type)
                        edf_attr['channel_types'] = ','.join(name for name in ch_types)
                    else:
                        edf_attr['channel_types'] = ['']
                elif k in ["lowpass", "highpass"]:
                    edf_attr[k] = v
                elif k == "sfreq":
                    edf_attr["sample_freq"] = v
                elif k == "meas_date":
                    if v is None:
                        edf_attr['date_of_record'] = "unspecified"
                    else:
                        edf_attr['date_of_record'] =  v.strftime("%Y-%m-%d %H:%M:%S %Z")
                elif k == "subject_info":
                    if v:
                        edf_attr['patient_information'] = v
                    else:
                        edf_attr['patient_information'] = "unspecified"
        
        structured_main_header = np.array([
                (
                    edf_attr["channel_names"],
                    edf_attr["channel_numbers"],
                    edf_attr["channel_types"],
                    edf_attr["sample_freq"],
                    edf_attr["lowpass"],
                    edf_attr["highpass"],
                    edf_attr["date_of_record"],
                    str(edf_attr['patient_information'])
                )
            ], dtype=self.main_header_ds_type)

        return structured_main_header

    def process_signal_header(self, edf_info):
        structured_signal_header = np.array([
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

        return structured_signal_header


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
        @param edf_file_path (string): The path of the EDF file that we want to load into our program.      
        """
        try:
            raw_edf_mne = mne.io.read_raw_edf(edf_file_path)
            raw_edf_py = pf.EdfReader(edf_file_path)
            return raw_edf_mne, raw_edf_py
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {edf_file_path}. Error: {e}")

    def run(self, edf_file_path, hdf5_file_path, datetime = False):
        """!
        @brief Executes the extracting functions and creat HDF5 file with datasets and groups.
        @param edf_file_path (string): The path of the EDF file.   
        @param hdf5_file_path (string): The destination path where we intend to save the HDF5 file.    
        """
        # Load the EDF file
        try : 
            raw_edf_mne, raw_edf_py = self.load_data(self,edf_file_path)
            dscrpt_ext = ExtractInfoFromEDF()
            tstamp_ext = ExtractTimeOfRecords()
            start_time = raw_edf_py.getStartdatetime()

        except RuntimeError as e:
            print(e)
        # Create new HDF5 file
        with h5py.File(hdf5_file_path, "w") as new_file:
            # Add EDF info to HDF5 file as an attribute
            extracted_data = dscrpt_ext.run(
                raw_edf_mne.info.copy())
            edf_descriptor = new_file.create_dataset(
                'EDF_descriptor', data=extracted_data)
            # Sanity check
            self.sanity_check_H5(edf_descriptor, len(extracted_data))

            # Create group for each signal_name and then export the raw data into the specific signal group.
            for sig_number in tqdm(range(raw_edf_py.signals_in_file)):
                ch_phys_values = raw_edf_py.readSignal(sig_number)
                ch_digit_values = raw_edf_py.readSignal(sig_number,digital=True)
                ch_name = "channel_{:04d}".format(sig_number)
                ch_descriptor = dscrpt_ext.run(
                    raw_edf_py.getSignalHeader(sig_number))
                # writing data in HDF5 file:
                ch_phys_datasets = new_file.create_dataset(
                    f'channels/{ch_name}/physical_values', data=ch_phys_values)
                # Sanity check
                self.sanity_check_H5(ch_phys_datasets, len(ch_phys_values))

                ch_digit_datasets = new_file.create_dataset(
                    f'channels/{ch_name}/digital_values', data=ch_digit_values)
                # Sanity check
                self.sanity_check_H5(ch_digit_datasets, len(ch_digit_values))

                ch_dscriptor = new_file.create_dataset(
                    f'channels/{ch_name}/descriptor', data=ch_descriptor)
                # Sanity check
                self.sanity_check_H5(ch_dscriptor, len(ch_descriptor))
                
                if datetime == True :
                    ch_times = tstamp_ext.run(
                        len(ch_phys_values), raw_edf_py.getSampleFrequency(sig_number), sig_number, start_time)
                    ch_timestamps = new_file.create_dataset(     
                    f'channels/{ch_name}/timestamps', data=ch_times)
                    # Sanity check
                    self.sanity_check_H5(ch_timestamps, len(ch_times))
    
    def sanity_check_H5(self, h5_dset, act_dset_len) :
        """!
        @brief Performs sanity checks on an HDF5 dataset.

        @param h5_dset (h5py.Dataset): The HDF5 dataset to be checked.
        @param act_dset_len (int): The expected length of the dataset.
        """
        assert h5_dset is not None, "The dataset is not correctly created."
        assert len(h5_dset) == act_dset_len, f"Unexpected data size. Expected length: {act_dset_len}."


if __name__ == "__main__":
    # Provide paths for your EDF and HDF5 files and call the methods.
    folder_edf_path = input("\nEDF foder path that contains EDF files: ")
    folder_hdf5_path = input("\nHDF5 foder path to export HDF5 files: ") 

    for file_name in tqdm(os.listdir(folder_edf_path)):
        if file_name.endswith('.edf'):
            edf_file_path = os.path.join(folder_edf_path,file_name)
            print("\n======================================")
            print("\nProcessing file : ", edf_file_path)
            print("\n======================================\n")
            
            hdf5_file_name = file_name + ".h5"
            hdf5_path = os.path.join(folder_hdf5_path,hdf5_file_name)

            edf_h5 = EDFToH5Converter()
            edf_h5.run(edf_file_path, hdf5_path)