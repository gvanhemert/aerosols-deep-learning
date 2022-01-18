#Imports
import tqdm
import glob
import numpy as np
import pandas as pd
import xarray as xr

import tensorflow as tf

from sklearn.model_selection import train_test_split

#%%
path = 'C:/Users/Guus van Hemert/Desktop/TW/TW5/Thesis/Data/'

# Reading datasets
ctl = xr.open_dataset(path + 'CTL_AOD550.nc')
ctl_masked = xr.open_dataset(path + 'CTL_AOD550_MASKED.nc')
nat = xr.open_dataset(path + 'NAT1_AOD550.nc')
spex_one = xr.open_dataset(path + 'SPEXone_Mask.nc')


#%% Functions
def create_input_images(df, input_label):
    '''Input a dataframe and an label for the column to convert
     returns an array containing the reshaped arrays from the chosen column containing arrays'''
    input_image = []
    for i in df.index:
        input_image.append(df[input_label][i].reshape(96, 192, 1))
    return np.array(input_image)

def create_output(df, output_label):
    '''Input a dataframe and an label for the column to convert
     returns an array containing the reshaped arrays from the chosen column containing arrays'''
    output_image = []
    for i in df.index:
        output_image.append(df[output_label][i].reshape(96, 192, 1))
    return np.array(output_image)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_combined_data(spex_day, ctl_lat_grad, ctl_lon_grad, ctl_day):

    # define the dictionary -- the structure -- of our single example
    data = {
        'height': _int64_feature(spex_day.shape[0]),
        'width': _int64_feature(spex_day.shape[1]),
        'depth': _int64_feature(spex_day.shape[2]),
        'spex_day': _bytes_feature(serialize_array(spex_day)),
        'ctl_lat_grad': _bytes_feature(serialize_array(ctl_lat_grad)),
        'ctl_lon_grad': _bytes_feature(serialize_array(ctl_lon_grad)),
        'ctl_day': _bytes_feature(serialize_array(ctl_day))
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def write_data(spex_day, ctl_lat_grad, ctl_lon_grad, ctl_day,
               filename, max_files, out_dir):
    '''Writes the data to multiple tfrecord files each containing max_files examples'''
    splits = (len(spex_day)//max_files) + 1
    if len(spex_day) % max_files == 0:
        splits -= 1

    print(
        f"\nUsing {splits} shard(s) for {len(spex_one)} files,\
            with up to {max_files} samples per shard")

    file_count = 0

    for i in tqdm.tqdm(range(splits)):
        current_shard_name = "{}{}_{}{}_{}.tfrecords".format(
            out_dir, i+1, splits, filename, max_files)
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files:
            index = i*max_files + current_shard_count
            if index == len(spex_day):
                break

            current_spex_day = spex_day[index]
            current_ctl_lat_grad = ctl_lat_grad[index]
            current_ctl_lon_grad = ctl_lon_grad[index]
            current_ctl_day = ctl_day[index]

            out = parse_combined_data(spex_day=current_spex_day,
                                      ctl_lat_grad=current_ctl_lat_grad,
                                      ctl_lon_grad=current_ctl_lon_grad,
                                      ctl_day = current_ctl_day)

            writer.write(out.SerializeToString())
            current_shard_count += 1
            file_count += 1

        writer.close()

    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count


def get_dataset_large(tfr_dir=path+'train_data/', 
                      pattern: str = "*train_data.tfrecords"):
    '''Loads the tfrecord files and returns a tfrecord dataset''' 
    files = glob.glob(tfr_dir+pattern, recursive=False)

    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(
        tf_parse)

    return dataset


def tf_parse(eg):
    """parse an example (or batch of examples, not quite sure...)"""

    # here we re-specify our format
    # you can also infer the format from the data using tf.train.Example.FromString
    # but that did not work
    example = tf.io.parse_example(
        eg[tf.newaxis],
        {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'spex_day': tf.io.FixedLenFeature([], tf.string),
            'ctl_lat_grad': tf.io.FixedLenFeature([], tf.string),
            'ctl_lon_grad': tf.io.FixedLenFeature([], tf.string),
            'ctl_day': tf.io.FixedLenFeature([], tf.string),
        },
    )
    spex_day = tf.io.parse_tensor(example["spex_day"][0], out_type="float32")
    ctl_lat_grad = tf.io.parse_tensor(example["ctl_lat_grad"][0], out_type="float32")
    ctl_lon_grad = tf.io.parse_tensor(example["ctl_lon_grad"][0], out_type="float32")
    ctl_day = tf.io.parse_tensor(example["ctl_day"][0], out_type="float32")
    input_data = tf.concat([spex_day, ctl_lat_grad, ctl_lon_grad], axis=-1)
    return input_data, ctl_day

#%%

n_days = int(ctl.time.data.shape[0] / 8)
shape = (n_days, ctl.lat.shape[0], ctl.lon.shape[0])

ctl_day_avg = {}
spex_day_avg = {}
ctl_day_lat_grad = {}
ctl_day_lon_grad = {}

for n in range(n_days):
    ctl_day_avg[n] = np.mean(ctl.TAU_2D_550nm.data[n*8:(n+1)*8], axis=0)
    ctl_day_avg[n] = (ctl_day_avg[n] - np.min(ctl_day_avg[n])) / (
        np.max(ctl_day_avg[n]) - np.min(ctl_day_avg[n]))
    ctl_day_lat_grad[n] = np.gradient(ctl_day_avg[n], axis=0)
    ctl_day_lon_grad[n] = np.gradient(ctl_day_avg[n], axis=1)
    spex_day_avg[n] = np.nanmean(spex_one.Count.data[n*8:(n+1)*8], axis=0)
    spex_day_avg[n] = np.nan_to_num(spex_day_avg[n], nan=0)
    
# Assign values to dataframe
dataframe = pd.DataFrame(ctl_day_avg.items(), columns=["Day", "CTL daily avg"])
dataframe = dataframe.assign(CTL_lat_grad = ctl_day_lat_grad.values(),
                             CTL_lon_grad = ctl_day_lon_grad.values(),
                             spex_day_avg = spex_day_avg.values())

# Create train and test data
(train_data, test_data) = train_test_split(dataframe, test_size=0.3,
                                           random_state=0)

#Input and output test data
ctl_lat_grad_test = create_input_images(test_data, 'CTL_lat_grad')
ctl_lon_grad_test = create_input_images(test_data, 'CTL_lon_grad')
spex_day_avg_test = create_input_images(test_data, 'spex_day_avg')
spex_day_avg_test = spex_day_avg_test.astype("float32")

ctl_test = create_output(test_data, "CTL daily avg")

#Input and output train data
ctl_lat_grad_train = create_input_images(train_data, 'CTL_lat_grad')
ctl_lon_grad_train = create_input_images(train_data, 'CTL_lon_grad')
spex_day_avg_train = create_input_images(train_data, 'spex_day_avg')
spex_day_avg_train = spex_day_avg_train.astype("float32")

ctl_train = create_output(train_data, "CTL daily avg")

#%% Write to tfrecord
write_data(spex_day_avg_test, ctl_lat_grad_test, ctl_lon_grad_test,
           ctl_test, max_files=100,
           filename='test_data', out_dir=path+"test_data/")

write_data(spex_day_avg_train, ctl_lat_grad_train, ctl_lon_grad_train,
           ctl_train, max_files=100,
           filename='train_data', out_dir=path+"train_data/")

#%% Testing
dataset = get_dataset_large()

for sample in dataset.take(1):
    print(repr(sample))
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2].shape)

