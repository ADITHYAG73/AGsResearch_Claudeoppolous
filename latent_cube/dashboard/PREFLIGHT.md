# Dashboard build preflight and completion record

Opening figure: AWS month-to-date net USD 0.00 for 2026-10 at 2026-10-30 23:20 IST, Cost Explorer, credits offset charges, may lag 24h; budget 15 GB S3 reads, about USD 2 maximum

- Read-only S3 prefix and environment-only temporary credentials; never read ~/.aws or write S3.
- 15,000,000,000-byte persistent reservation cap; each attempted/retried GET reserved before dispatch. Final reserved 13,463,130,313 bytes; successfully accounted payload 13,455,579,081 bytes. Listing/HTTP overhead is outside this payload ledger.
- Watchdog every 30 seconds; stop subsequent reads after 600 seconds without progress. Finite connection/read timeouts and at most two explicit transient retries. No active read process remains.
- Timestamped per-capture progress and phase logs; atomic incremental files; resumes reuse completed summaries. Temporary unrounded readouts are recreated on resume and rereads count toward budget.
- Four numerical threads, 16 network workers, lens batches of 16 vectors. Raw weights/readouts stored only in a /tmp TemporaryDirectory and deleted on exit. Final raw cache count: zero.
- Gate first, unchanged: max logit error <=0.5, RMSE <=0.05, relative max <=1%. Actual 0.06486892700195312 / 0.005661397064948961 / 0.20511913676506915%; six identical passes.
- Full export complete; lossless packing fixed the initial size failure. Site 54,568,952 bytes; largest 741,008 bytes. All retention limits and precision are listed on the page and in STATUS.md.
- Final validator, syntax, six browser-layout cases, 676-field preservation check and screenshot inspection passed. Hosting on claude.ai and actual settled AWS charges are not verified.
