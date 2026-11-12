# Cache cost estimates and durable use history

Add optional `cache::Adapter::Cost()` with the `cache::CostEstimate` interval
prefill and RAM/disk restore estimates. The default returns no calibration;
extensions can rebuild without overriding it. Estimates are diagnostic and do
not alter retention or capture thresholds.

Add advisory `cache::DiskStore::LastUse` and `TryRecordUse` for independently
checksummed, budgeted access history. Missing or invalid history returns zero;
a busy store refuses a write without waiting. Restore eligibility and immutable
checkpoint payloads do not depend on this history.
