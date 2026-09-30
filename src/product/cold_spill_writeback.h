#pragma once

// Cold spill WRITE-BACK: the disk tier's stage-out, off the caller's thread,
// with ONE named completion point.
//
// ===========================================================================
// WHAT THIS REPLACES, AND WHERE THE THING IT REPLACES STOOD
// ===========================================================================
//
// `src/targets/qwen3_6/impl/runtime/program_impl.h`, the Disk / HostThenDisk
// mirror (the "Mirror every layer's slot bytes into its spill file" block, which
// sat around :12562-12590 on the revision this header lands beside -- the remark
// below names the SHAPE, and REPORT.md carries the sha16 of the file this was
// read on, because a line number in an 18k-line header is not an identity):
//
//     for (layer = 0 .. layers-1)
//         cudaMemcpyAsync(cold_disk_staging[0], k_slot, bytes, D2H, device.stream);
//         cudaStreamSynchronize(device.stream);          // ONE FULL STREAM DRAIN PER LAYER
//         fseek(f, file_slot * bytes, SEEK_SET);
//         fwrite(cold_disk_staging[0], 1, bytes, f);
//     fflush(nullptr);                                   // EVERY OPEN STREAM IN THE PROCESS
//
// Three things are wrong with that, and all three are named by the reference
// material this line was ordered to read (sources table in REPORT.md):
//
//   (1) THE CALLER BLOCKS. The writing thread owns the decode/compaction flow,
//       so a page's spill is on the critical path. This is exactly KVMI-009's
//       surviving half, `kvmem_known_issues.md:331`, verbatim:
//
//           「NVMe stage-in reads 已重做，但 stage-out writes 仍同步。」
//           ("NVMe stage-in reads have been redone, but stage-out writes are
//            still synchronous")
//
//       The reference implementation fixed the READ half and LEFT THE WRITE HALF
//       SYNCHRONOUS. Not repeating that is what this file is for.
//   (2) IT DRAINS THE COMPUTE STREAM ONCE PER LAYER. `cudaStreamSynchronize` on
//       the engine's own stream is a fence on ALL outstanding work, not on one
//       copy, so 16 layers cost 16 full drains of the compute stream per page.
//   (3) IT FLUSHES EVERY STREAM IN THE PROCESS. `fflush(nullptr)` is not "flush
//       this page's file"; it walks the whole open-stream list. The per-page cost
//       is proportional to the number of open streams and the collateral is every
//       other component's buffering. `kvmem_nvme_ssd_architecture.md:249` states
//       the rule the other way round, verbatim: "no per-block `fflush` or
//       `fsync`" -- and :251-253 gives the reason: "KVMem storage is an ephemeral
//       inference cache. Durability after power loss is not required, so
//       per-block flushes provide no correctness benefit."
//
// ===========================================================================
// WHAT THIS DOES NOT CHANGE: NOT ONE BYTE, AND NOT ONE OFFSET
// ===========================================================================
//
// The unit of work is the same unit, the offset is the same formula
// (`file_slot * <that layer's slot stride>`), the buffer holds the same bytes the
// same D2H produced, and the file is the same file. This header moves WHEN a byte
// reaches the file and WHICH thread issues the write. The acceptance criterion is
// behavioural, not aesthetic: the same argv must produce byte-identical generated
// ids before and after, and REPORT.md prints the id digests on both arms.
//
// ===========================================================================
// THE COMPLETION POINT IS NAMED, AND A PAGE IS NOT "SPILLED" UNTIL IT IS TAKEN
// ===========================================================================
//
// THE FAILURE THIS DESIGN IS BUILT TO PREVENT is "I thought it was written". An
// async write is only safe if something names the moment the bytes are in the
// file, and if every reader of that region takes that moment before reading. So:
//
//   * `submit()` writes a `seq` into the caller's token and REFUSES -- naming
//     which way it refused -- rather than growing an unbounded queue. A refusal
//     is not a drop: the caller takes its own synchronous path.
//   * `drain_through(seq)` blocks until every write with `seq <= seq` completed
//     AND the files those writes touched were flushed. That is the named
//     completion point. It returns false, at that same point, if any of those
//     writes failed -- so a full disk, a short write, a bad descriptor or a flush
//     error surfaces as a REFUSAL where the caller already handles one (it
//     releases the file slot and the device slot and keeps the page resident),
//     never as a silent hole in a spill file.
//   * `drain_all()` is `drain_through()` at the highest seq ever issued. Every
//     read-back of a spill region takes it first.
//   * `stop()` drains everything and joins. The caller MUST call it before it
//     `fclose`s the files: a worker holding a FILE* the caller has closed is the
//     other way this design fails.
//
// The reference implementation's own vocabulary for the two halves is
// `kvmem_tiered_io_design.md:74` ("issue, don't wait") and :84 ("join and land"),
// and its overlap claim is the same one: the I/O runs "concurrently with the
// ... compute". This header is that `issue, don't wait` half plus a
// `join and land` that is a single call at a place the caller already has.
//
// ===========================================================================
// WHY ONE WORKER AND NOT A POOL (a deliberate, stated limit)
// ===========================================================================
//
// The reference design asks for io_uring with a calibrated queue depth
// (`kvmem_nvme_ssd_architecture.md:243-248`). THIS TREE HAS NO io_uring: a
// whole-tree scan for `io_uring`, `posix_fadvise` and `fdatasync` returns nothing
// on the cold path, and the tree's one `O_DIRECT` is in `src/artifact/reader.cpp`
// (a weight reader, a different subsystem). An io_uring backend would be a new
// dependency on the kernel and on the file system under `--cold-disk-path`, and it
// would move the write path onto a different syscall family than every existing
// reading of this tier was taken on. So the queue depth here is ONE WORKER and the
// concurrency comes from the caller not waiting, not from parallel submission.
//
// That ceiling is stated rather than hidden: what this file buys is REMOVAL FROM
// THE CRITICAL PATH, not a deeper queue. A second submission channel is the next
// step and is named as NOT DONE in REPORT.md.
//
// ===========================================================================
// THE BOUND, AND WHY A BOUND IS NOT OPTIONAL
// ===========================================================================
//
// Every queued write pins the slab it reads from. Unbounded queueing would let the
// spill pipeline allocate without limit, which is the opposite of what a tier that
// exists to bound memory may do. So the queue is bounded twice -- in ENTRIES (one
// slab per outstanding write, so the entry bound IS the slab bound) and in BYTES --
// and the bound is a REFUSAL, counted, never a drop.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
namespace ninfer::product {

// One unit of spill I/O. `source` must stay untouched from the moment it is
// accepted until `drain_through(seq)` has returned for the `seq` the submit call
// wrote into the caller's token -- that is the whole contract, and it is why the
// buffer is a slab of its own rather than a shared scratch.
struct ColdSpillWrite {
    std::size_t file_index = 0; // index into the caller's file table
    std::uint64_t offset   = 0; // absolute byte offset in that file
    std::size_t bytes      = 0;
    const std::uint8_t* source = nullptr;
};

// Every way `submit()` can answer. A caller that does not handle all three is the
// bug this enum exists to make impossible: the alternative to `Accepted` is never
// "your write happened anyway".
enum class ColdSpillSubmit : std::uint8_t {
    Accepted,   // queued; `out_seq` carries the completion token
    QueueFull,  // refused at the bound -- the caller takes its synchronous path
    NotRunning, // no worker: not started, or already stopped
};

[[nodiscard]] inline const char* cold_spill_submit_name(ColdSpillSubmit s) noexcept {
    switch (s) {
    case ColdSpillSubmit::Accepted: return "accepted";
    case ColdSpillSubmit::QueueFull: return "queue-full";
    case ColdSpillSubmit::NotRunning: return "not-running";
    }
    return "?";
}

struct ColdSpillStats {
    std::uint64_t submitted           = 0; // accepted into the queue
    std::uint64_t completed           = 0; // the worker issued the write
    std::uint64_t bytes               = 0; // bytes the worker actually wrote
    std::uint64_t refused_queue_full  = 0; // submit() returned QueueFull
    std::uint64_t refused_not_running = 0; // submit() returned NotRunning
    std::uint64_t write_failures      = 0; // fseek/fwrite failed on a queued write
    std::uint64_t flush_failures      = 0; // fflush failed while landing a drain
    std::uint64_t flush_calls         = 0; // one per (drain, file) -- NOT one per write
    std::uint64_t drains              = 0; // drain_through()/drain_all() calls
    std::uint64_t drains_that_waited  = 0; // of those, the ones that actually blocked
    std::uint64_t max_outstanding     = 0; // high-water mark of queued entries
    double        drain_ms            = 0.0; // total wall clock inside a drain
    double        submit_stall_ms     = 0.0; // wall clock the CALLER spent inside submit()

    [[nodiscard]] bool any() const noexcept {
        return submitted != 0 || completed != 0 || refused_queue_full != 0 ||
               refused_not_running != 0 || write_failures != 0 || flush_failures != 0;
    }

    // One line, printed by the caller where it already prints one. `mode` is the
    // caller's word for what it did with a refusal, so the counter and the
    // behaviour travel together.
    [[nodiscard]] std::string describe(const std::string& mode = "sync-fallback") const {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "[cold-spill] mode=async submitted=%llu completed=%llu bytes=%llu "
                      "refused_queue_full=%llu refused_not_running=%llu write_failures=%llu "
                      "flush_failures=%llu flush_calls=%llu drains=%llu drains_that_waited=%llu "
                      "max_outstanding=%llu drain_ms=%.3f submit_stall_ms=%.3f refusal_mode=%s",
                      static_cast<unsigned long long>(submitted),
                      static_cast<unsigned long long>(completed),
                      static_cast<unsigned long long>(bytes),
                      static_cast<unsigned long long>(refused_queue_full),
                      static_cast<unsigned long long>(refused_not_running),
                      static_cast<unsigned long long>(write_failures),
                      static_cast<unsigned long long>(flush_failures),
                      static_cast<unsigned long long>(flush_calls),
                      static_cast<unsigned long long>(drains),
                      static_cast<unsigned long long>(drains_that_waited),
                      static_cast<unsigned long long>(max_outstanding), drain_ms,
                      submit_stall_ms, mode.c_str());
        return std::string(buf);
    }
};

// A refusal has to say WHICH refusal, because the two want different fixes:
// `queue-full` means the caller is outrunning one worker (raise the bound or drain
// sooner); `not-running` means the engine was never started or is already stopped
// (a lifecycle bug, not a load problem).
[[nodiscard]] inline std::string cold_spill_refusal_line(ColdSpillSubmit s,
                                                         std::uint64_t file_index,
                                                         std::uint64_t offset,
                                                         std::uint64_t bytes) {
    return "[cold-spill] REFUSED (" + std::string(cold_spill_submit_name(s)) +
           ") file_index=" + std::to_string(file_index) + " offset=" + std::to_string(offset) +
           " bytes=" + std::to_string(bytes) +
           ": this write is NOT queued; the caller must take the synchronous path for it "
           "(and if that path fails too the page stays resident, it is never a silent hole)";
}

// A failed write is a NAMED refusal at the completion point, not a silent hole.
[[nodiscard]] inline std::string cold_spill_failure_line(std::uint64_t file_index,
                                                         std::uint64_t offset,
                                                         std::uint64_t bytes, int err) {
    return "[cold-spill] WRITE FAILED file_index=" + std::to_string(file_index) +
           " offset=" + std::to_string(offset) + " bytes=" + std::to_string(bytes) +
           " errno=" + std::to_string(err) + " (" +
           (err != 0 ? std::strerror(err) : "no errno recorded") +
           "): the spill file does not hold this page's bytes, so the page must NOT be "
           "treated as spilled";
}

class ColdSpillWriteback {
public:
    struct Config {
        // One slab per outstanding write, so this IS the slab count the caller must
        // supply. The call site uses 2 x layers: one page's worth in flight while the
        // next page's D2H is being enqueued.
        std::size_t max_outstanding_entries = 32;
        std::uint64_t max_outstanding_bytes = 0; // 0 = entries x slab_bytes
        std::size_t slab_bytes              = 0;
    };

    ColdSpillWriteback() = default;
    ColdSpillWriteback(const ColdSpillWriteback&)            = delete;
    ColdSpillWriteback& operator=(const ColdSpillWriteback&) = delete;
    ~ColdSpillWriteback() { stop(); }

    // `files` is the caller's file table and is NOT owned here: this engine never
    // opens, closes or truncates a file. The caller shares `io_mutex()` with its
    // read path -- an fseek+fwrite pair and an fseek+fread pair on one FILE* are
    // not atomic against each other.
    bool start(const std::vector<FILE*>& files, const Config& cfg) {
        // raceclose H5 -- DISCLOSED EXTRA, NOT one of C1..C5. This function publishes FOUR members
        // (`files_`, `cfg_`, `stopping_`, `running_`) and used to do it with NO lock held. TSan saw
        // the consequence on the lifecycle POC, at three distinct pairs: :262 (`running_ = true`) vs
        // :277 (`if (!running_ || stopping_)` in `submit`, which HOLDS `m_`), and :255 (`cfg_ = cfg`)
        // vs :283 (`q_.size() >= cfg_.max_outstanding_entries`) and :284 (`cfg_.max_outstanding_bytes`).
        // Taking `m_` here orders the publication against every reader that already takes it. The
        // worker is created while the lock is held, which is what makes `files_`/`cfg_` ordered for
        // `run()` and `write_one()` too: thread creation is the release that publishes them.
        std::lock_guard<std::mutex> lk(m_);
        if (running_) { return true; }
        files_ = &files;
        cfg_   = cfg;
        if (cfg_.max_outstanding_bytes == 0 && cfg_.slab_bytes != 0) {
            cfg_.max_outstanding_bytes =
                static_cast<std::uint64_t>(cfg_.max_outstanding_entries) *
                static_cast<std::uint64_t>(cfg_.slab_bytes);
        }
        stopping_ = false;
        running_  = true;
        worker_   = std::thread([this] { run(); });
        return true;
    }

    [[nodiscard]] bool running() const noexcept { return running_; }

    // The lock the read path takes around its own fseek+fread pair.
    [[nodiscard]] std::mutex& io_mutex() noexcept { return io_mutex_; }

    // Hand one write to the worker. On `Accepted`, `out_seq` is the token the caller
    // passes to `drain_through()` before touching `w.source` again.
    ColdSpillSubmit submit(const ColdSpillWrite& w, std::uint64_t& out_seq) {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(m_);
        if (!running_ || stopping_) {
            stats_.refused_not_running += 1;
            stats_.submit_stall_ms += ms_since(t0);
            return ColdSpillSubmit::NotRunning;
        }
        const std::uint64_t bytes = static_cast<std::uint64_t>(w.bytes);
        if (q_.size() >= cfg_.max_outstanding_entries ||
            (q_bytes_ + bytes) > cfg_.max_outstanding_bytes) {
            // THE BOUND IS A REFUSAL. Growing the queue would let the spill pipeline
            // allocate without limit, and dropping the write would publish a spill
            // file that does not hold the page.
            stats_.refused_queue_full += 1;
            stats_.submit_stall_ms += ms_since(t0);
            return ColdSpillSubmit::QueueFull;
        }
        Job job;
        job.seq        = ++seq_;
        job.file_index = w.file_index;
        job.offset     = w.offset;
        job.bytes      = w.bytes;
        job.source     = w.source;
        q_.push_back(job);
        q_bytes_ += bytes;
        stats_.submitted += 1;
        if (q_.size() > stats_.max_outstanding) { stats_.max_outstanding = q_.size(); }
        out_seq = job.seq;
        // ⚠ THIS STATEMENT IS ABOVE THE UNLOCK **ON PURPOSE**, and the placement is a
        // correctness requirement, not style. The worker thread writes `stats_`
        // (completed / bytes / write_failures) under this same `m_`, and `stats()` READS
        // the whole struct under it. An unlocked `+=` on a `double` here would be a data
        // race: usually invisible, UB always, and the field it corrupts is one of the ones
        // that carries this line's asynchrony claim. (Found by a readability re-read of
        // this header AFTER the poc was green -- "the tests pass" and "the code is
        // race-free" are different statements. Ledger F28; the first version of this file
        // had it below the unlock and the fix needed a rebuild the box could not afford
        // at the moment it was found.)
        stats_.submit_stall_ms += ms_since(t0);
        lk.unlock();
        cv_.notify_all();
        return ColdSpillSubmit::Accepted;
    }

    // THE NAMED COMPLETION POINT. True when every write with `seq <= seq` completed
    // AND the files those writes touched were flushed; false when one of them
    // failed, in which case the caller must NOT treat the page as spilled.
    // `seq == 0` means "nothing was ever submitted" and is answered immediately.
    //
    // THE WAIT IS ON PROCESSING ONLY, never on success. `processed_seq_` advances for
    // EVERY job the worker takes off the queue, success or failure; `completed_seq_`
    // advances only for the ones that landed. Waiting on `completed_seq_` would hang
    // forever on a failing write -- and a full disk fails EVERY write, so the first
    // full disk would hang the run instead of reporting it. (This is a defect the
    // line's own test caught: the wait predicate and the success counter were the same
    // variable. See REPORT.md section F.)
    bool drain_through(std::uint64_t seq) {
        if (seq == 0) { return true; }
        const auto t0 = std::chrono::steady_clock::now();
        bool waited = false;
        {
            std::unique_lock<std::mutex> lk(m_);
            stats_.drains += 1;
            if (processed_seq_ < seq) { waited = true; }
            cv_.wait(lk, [this, seq] {
                return processed_seq_ >= seq || !running_ || stopping_;
            });
            // raceclose (dl/_orch/landq/raceclose) H1 = closure criterion C1. This mutation used to sit
            // BELOW the unlock, four lines down, and so did the `drain_ms` one after `flush_dirty()`.
            // Every member of this struct is read under `m_` by `stats()`, so every mutation of it has
            // to be written under `m_` too -- the alternative is a data race between two caller threads,
            // and between a caller and the worker's own writes in `run()`. TSan reported exactly that,
            // at :343 and :345, for three distinct caller threads.
            if (waited) { stats_.drains_that_waited += 1; }
        }
        flush_dirty();
        bool spilled = true;
        {
            std::lock_guard<std::mutex> lk(m_);
            // H1, second half. This stays at the SAME point in the program -- after `flush_dirty()`, so
            // `drain_ms` still means "total wall clock inside a drain" and the number it reports does
            // not move; only the discipline around it does. Cost: one extra lock/unlock per drain.
            stats_.drain_ms += ms_since(t0);
            // H2 = closure criterion C2, and this one is the CORRECTNESS one. `failed_seq_` is written
            // by the worker under `m_` and read under `m_` by `drain_range` and the accessors; this read
            // was the single one that was not, and its value is one of the two inputs to the answer
            // about whether a page MAY BE TREATED AS SPILLED.
            spilled = failed_seq_ == kNoSeq || failed_seq_ > seq;
        }
        return spilled;
    }

    // A PAGE IS SEVERAL WRITES, so its completion point is a RANGE, not a point.
    // Waits for `last_seq`, then answers whether any write in [first_seq, last_seq]
    // failed. The range is what keeps a failure LOCAL: a full disk on an earlier
    // page leaves `failed_seq_` below every later page's `first_seq`, so later pages
    // whose own writes are all home are still reported spilled. `last_seq == 0`
    // (nothing submitted for this page) is answered true without waiting.
    bool drain_range(std::uint64_t first_seq, std::uint64_t last_seq) {
        if (last_seq == 0) { return true; }
        const bool reached = drain_through(last_seq);
        std::lock_guard<std::mutex> lk(m_);
        if (failed_seq_ == kNoSeq) { return reached; }
        return failed_seq_ < first_seq;
    }

    // The seq of the FIRST failed write, or `kNoSeq` when nothing failed. Exposed so
    // a caller can name which page a failure belongs to rather than only that one
    // happened.
    [[nodiscard]] std::uint64_t first_failed_seq() const {
        std::lock_guard<std::mutex> lk(m_);
        return failed_seq_;
    }
    [[nodiscard]] static constexpr std::uint64_t no_seq() noexcept { return kNoSeq; }

    // Every read-back of a spill region takes this first.
    // raceclose H4 -- DISCLOSED EXTRA, NOT one of C1..C5. `seq_` is guarded by `m_` at every other
    // site; this bare read was the one lock-free one. `last_seq()` takes the lock and returns the same
    // value, so this is behaviour-identical and the change is separable from the three criteria hunks.
    bool drain_all() { return drain_through(last_seq()); }

    // Drain everything and join. MUST be called before the caller `fclose`s the
    // files: a worker holding a FILE* the caller has closed is the second way this
    // design fails.
    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!running_) { return; }
            stopping_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
        running_ = false;
        flush_dirty();
    }

    // Copy, so a caller can print the line after it has already joined.
    [[nodiscard]] ColdSpillStats stats() const {
        std::lock_guard<std::mutex> lk(m_);
        return stats_;
    }

    // The last seq issued, so a caller that kept no token can still say "all".
    [[nodiscard]] std::uint64_t last_seq() const {
        std::lock_guard<std::mutex> lk(m_);
        return seq_;
    }

    [[nodiscard]] bool any_failure() const {
        std::lock_guard<std::mutex> lk(m_);
        return failed_seq_ != kNoSeq;
    }

    [[nodiscard]] int last_errno() const {
        std::lock_guard<std::mutex> lk(m_);
        return last_errno_;
    }

private:
    static constexpr std::uint64_t kNoSeq = ~static_cast<std::uint64_t>(0);

    struct Job {
        std::uint64_t seq          = 0;
        std::size_t file_index     = 0;
        std::uint64_t offset       = 0;
        std::size_t bytes          = 0;
        const std::uint8_t* source = nullptr;
    };

    static double ms_since(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    }

    void run() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return !q_.empty() || stopping_; });
                if (q_.empty()) { return; } // stopping_ and drained
                job = q_.front();
            }
            const bool ok = write_one(job);
            {
                std::lock_guard<std::mutex> lk(m_);
                q_.pop_front();
                q_bytes_ -= static_cast<std::uint64_t>(job.bytes);
                // PROCESSED advances for every job, landed or failed: it is what a
                // drain waits on, so a failing write must not stall the queue.
                if (job.seq > processed_seq_) { processed_seq_ = job.seq; }
                if (ok) {
                    stats_.completed += 1;
                    stats_.bytes += job.bytes;
                    if (job.seq > completed_seq_) { completed_seq_ = job.seq; }
                } else {
                    stats_.write_failures += 1;
                    if (job.seq < failed_seq_) { failed_seq_ = job.seq; }
                }
            }
            cv_.notify_all();
        }
    }

    bool write_one(const Job& job) {
        if (files_ == nullptr || job.file_index >= files_->size()) {
            set_errno(EBADF);
            return false;
        }
        FILE* f = (*files_)[job.file_index];
        if (f == nullptr) {
            set_errno(EBADF);
            return false;
        }
        // The same lock the read path takes: an fseek+fwrite pair and an
        // fseek+fread pair on one FILE* must not interleave.
        std::lock_guard<std::mutex> io(io_mutex_);
        // `static_cast<long>` is the SAME expression the synchronous site used, so
        // the offset that reaches the file is the same offset by the same rule. On
        // the 64-bit hosts this tier is read on, `long` is 64-bit; a 32-bit `long`
        // would truncate above 2 GiB and that limitation is the caller's, unchanged
        // by this header.
        if (std::fseek(f, static_cast<long>(job.offset), SEEK_SET) != 0) {
            set_errno(errno);
            return false;
        }
        if (std::fwrite(job.source, 1, job.bytes, f) != job.bytes) {
            set_errno(errno);
            return false;
        }
        std::lock_guard<std::mutex> lk(m_);
        dirty_.push_back(job.file_index);
        return true;
    }

    void set_errno(int e) {
        std::lock_guard<std::mutex> lk(m_);
        last_errno_ = e;
    }

    // ONE flush per (drain, FILE), never one per write, and never the process-wide
    // `fflush(nullptr)` the synchronous path used. The dedup is the whole point and it
    // is not cosmetic: a page is `layers` writes across `layers` files, so a drain that
    // flushed per ENTRY would flush as often as the layer count while claiming to flush
    // once per file. (Caught by this line's own test: `C5_four_flushes_for_sixteen_writes`
    // read 16 before the `seen` mask below existed. See REPORT.md section F.)
    void flush_dirty() {
        std::vector<std::size_t> files;
        // raceclose H6 -- DISCLOSED EXTRA, NOT one of C1..C5. `files_` is published by `start()` and
        // never changed again, so this is not a new invariant -- but the READ used to happen outside
        // `m_`, and `flush_dirty()` is called by CALLER threads (through `drain_through`), not only by
        // the worker. TSan saw it: :254 (`files_ = &files` in `start()`) against :506, on the lifecycle
        // POC. Capturing the pointer under the same lock `start()` now writes it under costs one store.
        const std::vector<FILE*>* fl = nullptr;
        {
            std::lock_guard<std::mutex> lk(m_);
            files.swap(dirty_);
            fl = files_;
        }
        if (files.empty() || fl == nullptr) { return; }
        std::uint64_t calls = 0;
        std::uint64_t fails = 0;
        {
            std::lock_guard<std::mutex> io(io_mutex_);
            std::vector<std::uint8_t> seen(fl->size(), 0);
            for (const std::size_t idx : files) {
                if (idx >= fl->size() || seen[idx] != 0) { continue; }
                seen[idx] = 1;
                FILE* f = (*fl)[idx];
                if (f == nullptr) { continue; }
                ++calls;
                if (std::fflush(f) != 0) { ++fails; }
            }
        }
        std::lock_guard<std::mutex> lk(m_);
        stats_.flush_calls += calls;
        stats_.flush_failures += fails;
    }

    // `m_` is MUTABLE so the const accessors below can take it: the alternative --
    // reading `stats_` without the lock -- would make the census a torn read, and the
    // census is what proves the timing claim.
    mutable std::mutex m_;
    // The lock shared with the caller's READ path. Separate from `m_` on purpose:
    // holding the queue lock across a blocking fwrite would let one slow write stall
    // every submit, which is the opposite of the point.
    std::mutex io_mutex_;
    std::condition_variable cv_;
    std::deque<Job> q_;
    std::vector<std::size_t> dirty_;
    const std::vector<FILE*>* files_ = nullptr;
    Config cfg_{};
    std::thread worker_;
    // raceclose (dl/_orch/landq/raceclose) H3 = closure criterion C3. These two were PLAIN `bool`:
    // written by `start()` at :262 and by `stop()` at :386 with no lock held, and read at :253, :267
    // (no lock), inside the wait predicates at :340, and under `m_` at :277 and :383. `std::atomic`
    // is the first of the two alternatives the criterion names (the other is "brought under `m_` at
    // all six sites"), and it is the right one here: `stop()` joins the worker while holding no lock,
    // and `running()` must stay callable from a thread that may not touch the queue lock. `m_` is
    // deliberately NOT used for these two -- see the comment on `m_` about not stalling submits.
    std::atomic<bool> running_{false};
    std::atomic<bool> stopping_{false};
    std::uint64_t seq_           = 0;
    // `processed_seq_` = every job the worker took off the queue (what a drain waits
    // on); `completed_seq_` = the ones that landed (what the success counter reads).
    // Keeping them separate is what makes a failing write terminate instead of hang.
    std::uint64_t processed_seq_ = 0;
    std::uint64_t completed_seq_ = 0;
    std::uint64_t failed_seq_    = kNoSeq;
    std::uint64_t q_bytes_       = 0;
    int last_errno_              = 0;
    ColdSpillStats stats_{};
};

} // namespace ninfer::product
