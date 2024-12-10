use serde::{Deserialize, Serialize};

use crate::events::{memory::MemoryWriteRecord, LookupId, MemoryLocalEvent};

/// Sqr Event.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct SqrEvent {
    /// The lookup identifier.
    pub lookup_id: LookupId,
    /// The shard number.
    pub shard: u32,
    /// The clock cycle.
    pub clk: u32,

    /// The pointer to the value to square.
    pub x_ptr: u32,
    /// The value to square as a list of words.
    pub x: Vec<u32>,

    /// The memory records for the value to square.
    /// NOTE: we read from x and write back to x
    pub x_memory_records: Vec<MemoryWriteRecord>,
    /// The local memory access records.
    pub local_mem_access: Vec<MemoryLocalEvent>,
}
