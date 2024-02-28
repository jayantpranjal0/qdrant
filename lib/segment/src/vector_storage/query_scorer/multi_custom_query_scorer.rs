use std::marker::PhantomData;

use common::types::{PointOffsetType, ScoreType};

use super::score_multivector;
use crate::data_types::vectors::MultiVector;
use crate::spaces::metric::Metric;
use crate::vector_storage::query::{Query, TransformInto};
use crate::vector_storage::query_scorer::QueryScorer;
use crate::vector_storage::MultiVectorStorage;

pub struct CustomQueryScorer<
    'a,
    TMetric: Metric,
    TVectorStorage: MultiVectorStorage,
    TQuery: Query<MultiVector>,
> {
    vector_storage: &'a TVectorStorage,
    query: TQuery,
    metric: PhantomData<TMetric>,
}

impl<
        'a,
        TMetric: Metric,
        TVectorStorage: MultiVectorStorage,
        TQuery: Query<MultiVector> + TransformInto<TQuery>,
    > CustomQueryScorer<'a, TMetric, TVectorStorage, TQuery>
{
    #[allow(dead_code)]
    pub fn new(query: TQuery, vector_storage: &'a TVectorStorage) -> Self {
        let query = query
            .transform(|vector| Ok(TMetric::preprocess(vector)))
            .unwrap();

        Self {
            query,
            vector_storage,
            metric: PhantomData,
        }
    }
}

impl<'a, TMetric: Metric, TVectorStorage: MultiVectorStorage, TQuery: Query<MultiVector>>
    QueryScorer<MultiVector> for CustomQueryScorer<'a, TMetric, TVectorStorage, TQuery>
{
    #[inline]
    fn score_stored(&self, idx: PointOffsetType) -> ScoreType {
        let stored = self.vector_storage.get_multi(idx);
        self.score(stored)
    }

    #[inline]
    fn score(&self, against: &MultiVector) -> ScoreType {
        self.query
            .score_by(|example| score_multivector::<TMetric>(example, against))
    }

    fn score_internal(&self, _point_a: PointOffsetType, _point_b: PointOffsetType) -> ScoreType {
        unimplemented!("Custom scorer can compare against multiple vectors, not just one")
    }
}
