import pydicom
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

LENGTH_OF_PER_TOPIC = 80
PART2_LOAD_SLICES = 25
FIRST_PATIENT_FOLDER = 'CT_chest_scans/0a0c32c9e08cc2ea76a71649de56be6d/'
FIRST_PATIENT_FIRST_DICOM_PATH = FIRST_PATIENT_FOLDER + '0a67f9edb4915467ac16a565955898d3.dcm'


def topic_log(topic_str, length):
    num_of_dash = length - (len(topic_str) + 2)
    left_num_of_dash = num_of_dash // 2
    right_num_of_dash = num_of_dash - left_num_of_dash
    print('\n\n' + '-' * left_num_of_dash + " " + topic_str + " " + '-' * right_num_of_dash)


def get_dicom_dataFields(file_path, log=False):
    ds = pydicom.read_file(file_path)
    if log:
        print(ds)
    return ds


def trans_pixels2housefield(ds):
    housefield_values = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    return housefield_values


def normalize(data):
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm_data


def get_data_statistics(data, data_type='raw_data', log=False):
    data = data.reshape(-1)
    min_data = np.min(data)
    max_data = np.max(data)
    mean_data = np.mean(data)
    std_data = np.std(data)
    statistics = dict(
        min=min_data,
        max=max_data,
        mean=mean_data,
        std=std_data
    )
    if log:
        print(f'{data_type:20} |   min: {min_data:<10} max: {max_data:<10} mean: {mean_data:<10.3f} std: '
              f'{std_data:<10.3f}')
    return statistics


def load_slices(dir_path, get_slices_num, log=False):
    all_slices = [pydicom.read_file(dir_path + '/' + file_name) for file_name in os.listdir(dir_path)]
    all_slices.sort(key=lambda x: x.ImagePositionPatient[2])
    slices_data = list()
    for counts, slice in list(enumerate(all_slices, start=1)):
        slice_data = trans_pixels2housefield(slice)
        slice_data = normalize(slice_data)
        slices_data.append(slice_data)
        if counts == get_slices_num:
            break
    if log:
        topic_log('part 2-1 [ read slices ]', LENGTH_OF_PER_TOPIC)
        print(f'get {get_slices_num} slices from {dir_path}')
        topic_log('part 2-2 [ sort slices ]', LENGTH_OF_PER_TOPIC)
        print(f'sort {get_slices_num} slices')
        topic_log('part 2-2 [ norm slices ]', LENGTH_OF_PER_TOPIC)
        print(f'normalize {get_slices_num} slices')
    return np.array(slices_data, dtype=np.float32)


def plot_slices(slices, save_path):
    num_of_plot_col = 5
    num_of_plot_row = slices.shape[0] // num_of_plot_col
    if slices.shape[0] % num_of_plot_col != 0:
        num_of_plot_row += 1
    f, plots = plt.subplots(num_of_plot_row, num_of_plot_col, figsize=(20, 20))
    for i in range(num_of_plot_row):
        for j in range(num_of_plot_col):
            idx = i * num_of_plot_col + j
            plots[i, j].axis('off')
            plots[i, j].imshow(slices[idx], cmap='gray')
            if idx == slices.shape[0]-1:
                plt.savefig(save_path)
                print(f'save figure to "{save_path}"')
                plt.close()
                return


def get_segent_slice(slice, thres_method='mean'):
    if thres_method == 'mean' or thres_method == 'median':
        block_size = 21
        thresholds = threshold_local(slice, block_size, method=thres_method)
        seg_slice = slice > thresholds
        threshold = np.mean(thresholds.reshape(-1))
    else:
        raise ValueError('thres_method must be "mean" or "median"')
    return seg_slice, threshold


def plot_segment_slice(thres_method='mean'):
    first_slice = load_slices(FIRST_PATIENT_FOLDER, 1)[0]
    seg_first_slice, threshold = get_segent_slice(first_slice, thres_method=thres_method)
    flat_first_slice = first_slice.flatten()

    f, plots = plt.subplots(3, 1, figsize=(70, 70))
    font_size = 100

    plots[0].set_title('histogram', fontsize=font_size)
    plots[0].hist(flat_first_slice, bins=100, color='steelblue')
    plots[0].axvline(threshold, min(flat_first_slice), max(flat_first_slice), lw=25, color='red')
    plt.sca(plots[0])
    plt.xticks(fontsize=font_size-20)
    plt.yticks(fontsize=font_size-20)

    plots[1].set_title('raw pixels', fontsize=font_size)
    plots[1].axis('off')
    plots[1].imshow(first_slice, cmap='gray')

    plots[2].set_title('threshold pixels', fontsize=font_size)
    plots[2].axis('off')
    plots[2].imshow(seg_first_slice, cmap='gray')

    save_path = './part3_' + thres_method + '.png'
    plt.savefig(save_path)
    plt.close()
    print(f'save figure to "{save_path}"')


def plot_3d(image, threshold):
    p = image.transpose(2, 1, 0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def part1():
    topic_log('part 1-1 [ print dicom dataFields ]', LENGTH_OF_PER_TOPIC)
    ds = get_dicom_dataFields(FIRST_PATIENT_FIRST_DICOM_PATH, log=True)

    topic_log('part 1-2  [ print data statistics ]', LENGTH_OF_PER_TOPIC)
    raw_data = ds.pixel_array
    housefield_data = trans_pixels2housefield(ds)
    _ = get_data_statistics(raw_data, data_type='raw data', log=True)
    _ = get_data_statistics(housefield_data, data_type='housefield data', log=True)


def part2():
    slices = load_slices(FIRST_PATIENT_FOLDER, PART2_LOAD_SLICES, log=True)
    topic_log('part 2-2 [ plot slices ]', LENGTH_OF_PER_TOPIC)
    plot_slices(slices, './part2.png')


def part3():
    topic_log('part 3-1 [ plot segment slice - mean ]', LENGTH_OF_PER_TOPIC)
    plot_segment_slice(thres_method='mean')

    topic_log('part 3-2 [ plot segment slice - median ]', LENGTH_OF_PER_TOPIC)
    plot_segment_slice(thres_method='median')


def bonus():
    scan = load_slices(FIRST_PATIENT_FOLDER, None)
    threshold = np.mean(np.array([get_segent_slice(slice, thres_method='mean')[1] for slice in scan]))
    # threshold = None
    topic_log('bonus part[ plot scan ]', LENGTH_OF_PER_TOPIC)
    plot_3d(scan, threshold)


def main():
    part1()
    part2()
    part3()
    bonus()


if __name__ == '__main__':
    main()




