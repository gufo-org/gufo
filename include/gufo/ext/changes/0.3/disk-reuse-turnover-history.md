# Durable reuse turnover history

Adds `DiskUseHistory`, `DiskStore::UseHistory` and the structured
`TryRecordUse` overload. History includes the admission-byte clock and reuse
stamp. Legacy 16-byte recency records remain readable; new checksummed records
are budgeted and advisory. Existing scalar accessors retain their behavior.
