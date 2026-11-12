# Discard queued optional persistence

Add `cache::PersistenceQueue::DiscardPending(CheckpointId)`. It releases a
queued write's source pins, counts it as superseded and reports false completion
outside the queue lock before returning. Active writes are unaffected. `Drain`
waits for these completion callbacks as well as queued and active writes.
