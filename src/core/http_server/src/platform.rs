//! OS-specific socket setup and listen-address parsing.
//!
//! All platform-specific networking code for `gufo-http-server` lives in this
//! module so the rest of the crate stays portable.

use std::io;
use std::net::{IpAddr, Ipv6Addr, SocketAddr, SocketAddrV6, TcpListener};

use socket2::{Domain, Protocol, Socket, Type};

/// Ignores `SIGPIPE` on POSIX so writing to a closed client socket returns
/// `EPIPE` instead of terminating the process.
pub(crate) fn ignore_sigpipe() {
    #[cfg(unix)]
    {
        // SAFETY: `signal(SIGPIPE, SIG_IGN)` is async-signal-safe and standard
        // POSIX process-wide configuration for network servers.
        #[allow(unsafe_code)]
        unsafe {
            const SIGPIPE: i32 = 13;
            const SIG_IGN: usize = 1;
            unsafe extern "C" {
                fn signal(sig: i32, handler: usize) -> usize;
            }
            let _ = signal(SIGPIPE, SIG_IGN);
        }
    }
}

/// Parses a numeric IPv4 or IPv6 literal (with optional `%<scope>` on IPv6)
/// into a `SocketAddr`. Host names are never resolved.
pub(crate) fn parse_listen_address(host: &str, port: u16) -> Option<SocketAddr> {
    if host.is_empty() {
        return None;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(SocketAddr::new(ip, port));
    }
    let (addr_part, scope_part) = host.split_once('%')?;
    if scope_part.is_empty() {
        return None;
    }
    let ipv6: Ipv6Addr = addr_part.parse().ok()?;
    let scope_id = parse_scope_id(scope_part)?;
    Some(SocketAddr::V6(SocketAddrV6::new(ipv6, port, 0, scope_id)))
}

fn parse_scope_id(scope: &str) -> Option<u32> {
    if scope.is_empty()
        || scope.contains('[')
        || scope.contains(']')
        || scope.bytes().any(|b| b.is_ascii_whitespace())
    {
        return None;
    }
    if let Ok(id) = scope.parse::<u32>() {
        return Some(id);
    }
    #[cfg(unix)]
    {
        let c_scope = std::ffi::CString::new(scope).ok()?;
        // SAFETY: `c_scope` is a valid NUL-terminated C string.
        #[allow(unsafe_code)]
        let index = unsafe {
            unsafe extern "C" {
                fn if_nametoindex(ifname: *const std::ffi::c_char) -> u32;
            }
            if_nametoindex(c_scope.as_ptr())
        };
        Some(index)
    }
    #[cfg(not(unix))]
    {
        Some(0)
    }
}

/// Binds a non-blocking TCP listener with platform-appropriate reuse semantics
/// and dual-stack IPv6 (`IPV6_V6ONLY = false`).
pub(crate) fn bind_tcp_listener(addr: SocketAddr, backlog: i32) -> io::Result<TcpListener> {
    ignore_sigpipe();
    let domain = match addr {
        SocketAddr::V4(_) => Domain::IPV4,
        SocketAddr::V6(_) => Domain::IPV6,
    };
    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;

    // Exclusive on Windows so a second server cannot bind an active port;
    // SO_REUSEADDR on POSIX so restarts bind through TIME_WAIT.
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawSocket;
        const SOL_SOCKET: i32 = 0xffff;
        const SO_EXCLUSIVEADDRUSE: i32 = -5;
        let optval: i32 = 1;
        // SAFETY: `socket` is a valid Winsock handle and `optval` is a valid 4-byte i32.
        #[allow(unsafe_code)]
        let rc = unsafe {
            unsafe extern "system" {
                fn setsockopt(
                    s: usize,
                    level: i32,
                    optname: i32,
                    optval: *const i32,
                    optlen: i32,
                ) -> i32;
            }
            setsockopt(
                socket.as_raw_socket() as usize,
                SOL_SOCKET,
                SO_EXCLUSIVEADDRUSE,
                &optval,
                std::mem::size_of::<i32>() as i32,
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(not(windows))]
    {
        socket.set_reuse_address(true)?;
    }

    if addr.is_ipv6() {
        socket.set_only_v6(false)?;
    }

    socket.bind(&addr.into())?;
    socket.listen(backlog)?;
    socket.set_nonblocking(true)?;
    Ok(socket.into())
}

/// Formats a peer address for `HttpRequest::client_id`, normalizing
/// IPv4-mapped IPv6 addresses (`::ffff:a.b.c.d`) to IPv4.
pub(crate) fn format_peer_addr(addr: SocketAddr) -> String {
    match addr.ip() {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                v4.to_string()
            } else {
                v6.to_string()
            }
        }
    }
}

/// Returns true if an `io::Error` indicates the address is already in use.
pub(crate) fn is_addr_in_use(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::AddrInUse
}
