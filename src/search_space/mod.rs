mod group_decomposed;
mod intersection;
pub mod transform;

pub use group_decomposed::{GroupDecomposedSearchSpace, SearchSpaceGroup};
pub use intersection::{intersection_search_space, IntersectionSearchSpace};
pub use transform::{SearchSpaceTransform, round_ties_even};
