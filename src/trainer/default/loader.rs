mod direct;
mod montybinpack;
mod sfbinpack;

use bulletformat::BulletFormat;
pub use direct::{CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader};
pub use montybinpack::MontyBinpackLoader;
pub use sfbinpack::SfBinpackLoader;

use super::{inputs::SparseInputType, outputs::OutputBuckets};

use crate::{tensor::Shape, trainer::DataPreparer};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub trait LoadableDataType: Sized {
    fn score(&self) -> i16;

    fn result(&self) -> GameResult;
}

impl<T: BulletFormat + 'static> LoadableDataType for T {
    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result_idx()]
    }

    fn score(&self) -> i16 {
        <Self as BulletFormat>::score(self)
    }
}

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, batch_size: usize, f: F);
}

#[derive(Clone)]
pub struct DefaultDataLoader<I, O, D> {
    input_getter: I,
    output_getter: O,
    wdl: bool,
    scale: f32,
    loader: D,
    dense_inputs: bool,
}

impl<I, O, D> DefaultDataLoader<I, O, D> {
    pub fn new(input_getter: I, output_getter: O, wdl: bool, scale: f32, loader: D, dense_inputs: bool) -> Self {
        Self { input_getter, output_getter, wdl, scale, loader, dense_inputs }
    }
}

impl<I, O, D> DataPreparer for DefaultDataLoader<I, O, D>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    D: DataLoader<I::RequiredDataType>,
{
    type DataType = I::RequiredDataType;
    type PreparedData = DefaultDataPreparer<I, O>;

    fn get_data_file_paths(&self) -> &[String] {
        self.loader.data_file_paths()
    }

    fn try_count_positions(&self) -> Option<u64> {
        self.loader.count_positions()
    }

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(&self, batch_size: usize, f: F) {
        self.loader.map_batches(batch_size, f);
    }

    fn prepare(&self, data: &[Self::DataType], threads: usize, blend: f32) -> Self::PreparedData {
        DefaultDataPreparer::prepare(
            self.input_getter.clone(),
            self.output_getter,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
            self.dense_inputs,
        )
    }
}

pub(crate) struct DenseInput {
    pub shape: Shape,
    pub value: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct SparseInput {
    pub shape: Shape,
    pub value: Vec<i32>,
    pub max_active: usize,
}

impl Default for SparseInput {
    fn default() -> Self {
        Self { shape: Shape::new(0, 0), value: Vec::new(), max_active: 0 }
    }
}

/// A batch of data, in the correct format for the GPU.
pub struct DefaultDataPreparer<I, O> {
    input_getter: I,
    output_getter: O,
    pub(crate) batch_size: usize,
    pub(crate) stm: SparseInput,
    pub(crate) nstm: SparseInput,
    pub(crate) dstm: DenseInput,
    pub(crate) dnstm: DenseInput,
    pub(crate) buckets: SparseInput,
    pub(crate) targets: DenseInput,
}

impl<I: SparseInputType, O: OutputBuckets<I::RequiredDataType>> DefaultDataPreparer<I, O> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        input_getter: I,
        output_getter: O,
        wdl: bool,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
        dense: bool,
    ) -> Self {
        let rscale = 1.0 / scale;
        let batch_size = data.len();
        let max_active = input_getter.max_active();
        let chunk_size = batch_size.div_ceil(threads);

        let input_size = input_getter.num_inputs();

        let output_size = if wdl { 3 } else { 1 };

        let shape = Shape::new(input_size, batch_size);
        let sparse_size = if dense { batch_size } else { max_active * batch_size };
        let dense_size = if dense { shape.size() } else { batch_size };

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm: SparseInput { shape, max_active, value: vec![0; sparse_size] },
            nstm: SparseInput { shape, max_active, value: vec![0; sparse_size] },
            dstm: DenseInput { shape, value: vec![0.0; dense_size] },
            dnstm: DenseInput { shape, value: vec![0.0; dense_size] },
            buckets: SparseInput {
                shape: Shape::new(O::BUCKETS, batch_size),
                max_active: 1,
                value: vec![0; batch_size],
            },
            targets: DenseInput {
                shape: Shape::new(output_size, batch_size),
                value: vec![0.0; output_size * batch_size],
            },
        };

        let sparse_chunk_size = if dense { 1 } else { max_active * chunk_size };
        let dense_chunk_size = if dense { input_size * chunk_size } else { 1 };

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.value.chunks_mut(sparse_chunk_size))
                .zip(prep.nstm.value.chunks_mut(sparse_chunk_size))
                .zip(prep.dstm.value.chunks_mut(dense_chunk_size))
                .zip(prep.dnstm.value.chunks_mut(dense_chunk_size))
                .zip(prep.buckets.value.chunks_mut(chunk_size))
                .zip(prep.targets.value.chunks_mut(output_size * chunk_size))
                .for_each(
                    |(
                        (((((data_chunk, stm_chunk), nstm_chunk), dstm_chunk), dnstm_chunk), buckets_chunk),
                        results_chunk,
                    )| {
                        let inp = &prep.input_getter;
                        let out = &prep.output_getter;
                        s.spawn(move || {
                            let chunk_len = data_chunk.len();

                            for i in 0..chunk_len {
                                let pos = &data_chunk[i];
                                let mut j = 0;
                                let sparse_offset = max_active * i;
                                let dense_offset = input_size * i;

                                inp.map_features(pos, |our, opp| {
                                    if dense {
                                        dstm_chunk[dense_offset + our] = 1.0;
                                        dnstm_chunk[dense_offset + opp] = 1.0;
                                    } else {
                                        stm_chunk[sparse_offset + j] = our as i32;
                                        nstm_chunk[sparse_offset + j] = opp as i32;
                                    }

                                    j += 1;
                                });

                                if !dense && j < max_active {
                                    stm_chunk[sparse_offset + j] = -1;
                                    nstm_chunk[sparse_offset + j] = -1;
                                }

                                assert!(j <= max_active, "More inputs provided than the specified maximum!");

                                buckets_chunk[i] = i32::from(out.bucket(pos));

                                if wdl {
                                    results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                                } else {
                                    let score = 1. / (1. + (-rscale * f32::from(pos.score())).exp());
                                    let result = f32::from(pos.result() as u8) / 2.0;
                                    results_chunk[i] = blend * result + (1. - blend) * score;
                                }
                            }
                        });
                    },
                );
        });

        prep
    }
}
