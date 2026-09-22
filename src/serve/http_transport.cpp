#include "serve/http_transport.h"

#include "serve/request_validation.h"

#if defined(_WIN32)
// httplib.h has already pulled <winsock2.h> on this path and defined NOMINMAX itself; <mstcpip.h>
// is what actually declares SIO_KEEPALIVE_VALS and the tcp_keepalive argument structure it takes.
#    include <mstcpip.h>
#elif defined(__linux__)
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#endif

#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

#if defined(_WIN32)
// Windows has SO_KEEPALIVE but NOT the Linux TCP_KEEP* knobs. MEASURED against the installed SDK
// (C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0): um\ws2tcpip.h contains no
// occurrence of TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT or TCP_USER_TIMEOUT. What exists is one
// WSAIoctl with SIO_KEEPALIVE_VALS (shared\mstcpip.h:186) carrying a tcp_keepalive, and that struct
// has fields for on/off, idle time and probe interval -- and NO FIELD FOR A PROBE COUNT. What that
// costs is stated where the probes are configured, in configure_http_server_socket.
constexpr unsigned long kKeepAliveIdleMilliseconds     = 10 * 1000; // TCP_KEEPIDLE  10 s
constexpr unsigned long kKeepAliveIntervalMilliseconds = 3 * 1000;  // TCP_KEEPINTVL  3 s

// Same shape as the Linux twin below; the difference is the cast, because Windows declares the
// value parameter as const char* where POSIX declares const void*. This cast spelling is httplib's
// own (third_party/cpp-httplib/httplib.h:2047-2048), so it is the tree's dependency's convention
// rather than a new one.
template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, reinterpret_cast<const char*>(&value), sizeof(value));
}
#elif defined(__linux__)
constexpr int kKeepAliveIdleSeconds                = 10;
constexpr int kKeepAliveIntervalSeconds            = 3;
constexpr int kKeepAliveProbeCount                 = 3;
constexpr unsigned int kTcpUserTimeoutMilliseconds = 15000;

template <class T>
void set_socket_option(socket_t socket, int level, int option, const T& value) noexcept {
    (void)::setsockopt(socket, level, option, &value, sizeof(value));
}
#endif

} // namespace

nlohmann::json parse_json_body(const httplib::Request& request) {
    try {
        return nlohmann::json::parse(request.body);
    } catch (const std::exception&) { bad_request("request body is not valid JSON"); }
}

bool client_disconnected(const httplib::Request& request) {
    return request.is_connection_alive && !request.is_connection_alive();
}

void prepare_sse_response(httplib::Response& response) {
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
}

SseTransport::SseTransport(httplib::DataSink& sink, std::atomic<bool>& cancelled,
                           Clock::duration heartbeat_interval, Clock::time_point now)
    : sink_(sink), cancelled_(cancelled), heartbeat_interval_(heartbeat_interval),
      last_write_(now) {
    if (heartbeat_interval_ <= Clock::duration::zero()) {
        throw std::invalid_argument("SSE heartbeat interval must be positive");
    }
}

bool SseTransport::mark_cancelled() noexcept {
    cancelled_.store(true, std::memory_order_release);
    return true;
}

void SseTransport::write(std::string_view item, Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire) || !sink_.write(item.data(), item.size())) {
        mark_cancelled();
        throw ClientDisconnected();
    }
    last_write_ = now;
}

void SseTransport::write(const std::vector<std::string>& items, Clock::time_point now) {
    for (const std::string& item : items) { write(item, now); }
}

bool SseTransport::poll(Clock::time_point now) {
    if (cancelled_.load(std::memory_order_acquire)) { return true; }
    if (sink_.is_writable && !sink_.is_writable()) { return mark_cancelled(); }
    if (now - last_write_ < heartbeat_interval_) { return false; }
    if (!sink_.write(kHeartbeatComment.data(), kHeartbeatComment.size())) {
        return mark_cancelled();
    }
    last_write_ = now;
    return false;
}

void configure_http_server_socket(socket_t socket) noexcept {
    httplib::default_socket_options(socket);
#if defined(_WIN32)
    const int enabled = 1;
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    tcp_keepalive keepalive{};
    keepalive.onoff             = 1;
    keepalive.keepalivetime     = kKeepAliveIdleMilliseconds;
    keepalive.keepaliveinterval = kKeepAliveIntervalMilliseconds;
    DWORD returned              = 0;
    (void)::WSAIoctl(socket, SIO_KEEPALIVE_VALS, &keepalive, sizeof(keepalive), nullptr, 0, &returned,
                     nullptr, nullptr);

    // WHAT THIS ARM DOES NOT REPRODUCE, named rather than dropped silently:
    //   1. TCP_KEEPCNT. Windows has no per-socket probe count; the stack uses its own. The direction
    //      of the difference is that a Windows server keeps probing PAST three unanswered probes
    //      before it gives up, so a dead peer is noticed LATER here than on Linux, never never.
    //   2. TCP_USER_TIMEOUT (15 s on the Linux arm). Windows exposes no per-socket equivalent, so
    //      the "how long may unacknowledged data sit" bound is not set from here at all.
    // Both are losses of TIMING GRANULARITY, not of the disconnect-detection mechanism itself: the
    // SSE heartbeat in SseTransport::poll is what actually clears a stalled stream, and it is
    // unchanged and portable. A Windows deployment should expect a wider worst-case detection
    // window than a Linux one -- that is a statement about this file, not a support claim about any
    // Windows version.
#elif defined(__linux__)
    const int enabled = 1;
    set_socket_option(socket, SOL_SOCKET, SO_KEEPALIVE, enabled);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPIDLE, kKeepAliveIdleSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPINTVL, kKeepAliveIntervalSeconds);
    set_socket_option(socket, IPPROTO_TCP, TCP_KEEPCNT, kKeepAliveProbeCount);
    set_socket_option(socket, IPPROTO_TCP, TCP_USER_TIMEOUT, kTcpUserTimeoutMilliseconds);
#endif
}

void set_owned_json_content(httplib::Response& response, std::string body,
                            std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

} // namespace ninfer::serve
