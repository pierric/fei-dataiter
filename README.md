# mxnet-dataiter

Here is an example of making a Conduit from MNIST dataset.

```haskell
mnistIter (add @"image" "data/train-images-idx3-ubyte" $ 
           add @"label" "data/train-labels-idx1-ubyte" $
           add @"batch_size" 128 
           nil) :: ConduitData IO (NDArray Float, NDArray Float)
```

The first argument is provides named parameters for the MXNet Data Iterators. Detailed specification can be 
found in MXNet API 's python [document](https://mxnet.incubator.apache.org/api/python/io/io.html).

Below is a snapshot of current support in this package.

```haskell
type CSVIter_Args = 
    '[ "data_csv" := String, "data_shape" := [Int], "label_csv" := String, "label_shape" := [Int]
     , "batch_size" := Int, "round_batch" := Bool, "prefetch_buffer" := Integer, "dtype" := String] 

type MNISTIter_Args = 
    '[ "image" := String, "label" := String, "batch_size" := Int, "shuffle" := Bool, "flat" := Bool
     , "seed" := Int, "silent" := Bool, "num_parts" := Int, "part_index" := Int
     , "prefetch_buffer" := Integer, "dtype" := String]

type ImageRecordIter_Args = 
    '[ "path_imglist" := String, "path_imgrec" := String, "path_imgidx" := String, "aug_seq" := String
     , "label_width" := Int, "data_shape" := [Int], "preprocess_threads" := Int, "verbose" := Bool
     , "num_parts" := Int, "part_index" := Int, "shuffle_chunk_size" := Integer
     , "shuffle_chunk_seed" := Int, "shuffle" := Bool, "seed" := Int, "batch_size" := Int
     , "round_batch" := Bool, "prefetch_buffer" := Integer, "dtype" := String, "resize" := Int
     , "rand_crop" := Bool, "max_rotate_angle" := Int, "max_aspect_ratio" := Float
     , "max_shear_ratio" := Float, "max_crop_size" := Int, "min_crop_size" := Int
     , "max_random_scale" := Float, "min_random_scale" := Float, "max_img_size" := Float
     , "min_img_size" := Float, "random_h" := Int, "random_s" := Int, "random_l" := Int, "rotate" := Int
     , "fill_value" := Int, "inter_method" := Int, "pad" := Int, "mirror" := Bool, "rand_mirror" := Bool
     , "mean_img" := String, "mean_r" := Float, "mean_g" := Float, "mean_b" := Float, "mean_a" := Float
     , "std_r" := Float, "std_g" := Float, "std_b" := Float, "std_a" := Float, "scale" := Float
     , "max_random_contrast" := Float, "max_random_illumination" := Float]
     
type ImageDetRecordIter_Args = 
    '[ "path_imglist" := String, "path_imgrec" := String, "aug_seq" := String, "label_width" := Int
     , "data_shape" := [Int], "preprocess_threads" := Int, "verbose" := Bool, "num_parts" := Int
     , "part_index" := Int, "shuffle_chunk_size" := Integer, "shuffle_chunk_seed" := Int
     , "label_pad_width" := Int, "label_pad_value" := Float, "shuffle" := Bool, "seed" := Int
     , "batch_size" := Int, "round_batch" := Bool, "prefetch_buffer" := Integer, "dtype" := String
     , "resize" := Int, "rand_crop_prob" := Float, "min_crop_scales" := [Float]
     , "max_crop_scales" := [Float], "min_crop_aspect_ratios" := [Float]
     , "max_crop_aspect_ratios" := [Float], "min_crop_overlaps" := [Float], "max_crop_overlaps" := [Float]
     , "min_crop_sample_coverages" := [Float], "max_crop_sample_coverages" := [Float]
     , "min_crop_object_coverages" := [Float], "max_crop_object_coverages" := [Float]
     , "num_crop_sampler" := Int, "crop_emit_mode" := String, "emit_overlap_thresh" := Float
     , "max_crop_trials" := [Int], "rand_pad_prob" := Float, "max_pad_scale" := Float
     , "max_random_hue" := Int, "random_hue_prob" := Float, "max_random_saturation" := Int
     , "random_saturation_prob" := Float, "max_random_illumination" := Int
     , "random_illumination_prob" := Float, "max_random_contrast" := Float, "random_contrast_prob" := Float
     , "rand_mirror_prob" := Float, "fill_value" := Int, "inter_method" := Int, "resize_mode" := String
     , "mean_img" := String, "mean_r" := Float, "mean_g" := Float, "mean_b" := Float, "mean_a" := Float
     , "std_r" := Float, "std_g" := Float, "std_b" := Float, "std_a" := Float, "scale" := Float]

type ImageRecordUInt8Iter_Args = 
    '[ "path_imglist" := String, "path_imgrec" := String, "path_imgidx" := String, "aug_seq" := String
     , "label_width" := Int, "data_shape" := [Int], "preprocess_threads" := Int, "verbose" := Bool
     , "num_parts" := Int, "part_index" := Int, "shuffle_chunk_size" := Integer
     , "shuffle_chunk_seed" := Int, "shuffle" := Bool, "seed" := Int, "batch_size" := Int
     , "round_batch" := Bool, "prefetch_buffer" := Integer, "dtype" := String, "resize" := Int
     , "rand_crop" := Bool, "max_rotate_angle" := Int, "max_aspect_ratio" := Float
     , "max_shear_ratio" := Float, "max_crop_size" := Int, "min_crop_size" := Int
     , "max_random_scale" := Float, "min_random_scale" := Float, "max_img_size" := Float
     , "min_img_size" := Float, "random_h" := Int, "random_s" := Int, "random_l" := Int, "rotate" := Int
     , "fill_value" := Int, "inter_method" := Int, "pad" := Int]

type LibSVMIter_Args = 
    '[ "data_libsvm" := String, "data_shape" := [Int], "label_libsvm" := String, "label_shape" := [Int]
     , "num_parts" := Int, "part_index" := Int, "batch_size" := Int, "round_batch" := Bool
     , "prefetch_buffer" := Integer, "dtype" := String]
```
