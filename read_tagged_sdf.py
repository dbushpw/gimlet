# =============================================================================
# imports
# =============================================================================
import gin
import lime
import pandas as pd
import tensorflow as tf

# =============================================================================
# utility functions
# =============================================================================
def read_tagged_sdf(file_path):
    # read the file
    text = tf.io.read_file(
        file_path)

    lines = tf.strings.split(
        tf.expand_dims(text, 0),
        '\n').values

    starts = tf.where(tf.strings.regex_full_match(
        lines,
        '.*END.*')) + 1

    # get the starts and the ends
    ends = tf.where(tf.strings.regex_full_match(
        lines,
        '.*\$\$\$\$.*')) - 1

    print(ends)

    attr_idxs = tf.map_fn(
        lambda x: tf.range(x[0], x[1]),
        tf.squeeze(tf.stack(
            [
                starts,
                ends
            ],
            axis=1)),
        dtype=tf.int64)

    id_idxs = attr_idxs[:, 1]
    pka_idxs = attr_idxs[:, 4]
    pka_cx_idxs = attr_idxs[:, 7]
    _atom_idxs = attr_idxs[:, 10]

    ids = tf.strings.to_number(tf.gather(lines, id_idxs), tf.float32)
    pkas = tf.strings.to_number(tf.gather(lines, pka_idxs), tf.float32)
    pka_cxs = tf.strings.to_number(tf.gather(lines, pka_cx_idxs), tf.float32)
    _atoms = tf.strings.to_number(tf.gather(lines, _atom_idxs), tf.float32)


    attr = tf.stack(
        [
            ids,
            pkas,
            pka_cxs,
            _atoms
        ],
        axis=1)

    ds_attr = tf.data.Dataset.from_tensor_slices(attr)

    new_sdf_mask = tf.tensor_scatter_nd_update(
        tf.constant(
            True,
            shape=tf.TensorShape([lines.shape[0],])),
        tf.reshape(attr_idxs, [-1, 1]),
        tf.constant(
            False,
            shape=[tf.shape(tf.reshape(attr_idxs, [-1]))[0], ]))

    tf.io.write_file(
        'temp.sdf',
        tf.strings.join(
            tf.boolean_mask(
                lines,
                new_sdf_mask),
            separator='\n'))

    mol_ds = gin.i_o.from_sdf.to_ds('temp.sdf')

    ds = tf.data.Dataset.zip((mol_ds, ds_attr))

    return ds

# read_tagged_sdf('/Users/yuanqingwang/Downloads/mp_df_only_normal.sdf')
