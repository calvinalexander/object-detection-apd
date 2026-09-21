# Device events mapped onto the 17 Sep recordings

source: `data/recording/17/event.db` (to 2026-09-18 02:41), clip windows from
`data/recording/17/db` (the recorder's own index, exact start/end per clip).

| | |
|---|---|
| events in the database | 10971 |
| inside one of the 351 copied recordings | **167** |
| clips with at least one event | 55 |
| violations | gloves 82, headcap 92, mask 41 |
| placed by picture | 49 |
| placed by timestamp minus median lag | 118 |
| device lag behind the filmed frame | median 16.4s, range 0.9..44.4 |

Snapshots: the device's own image where it was copied
(120), otherwise the frame from the
recording with the person box drawn, marked "rekonstruksi"
(47).

Timestamps in apd_events.db are FILMED time, so they line up with a replay of the
same clips.
