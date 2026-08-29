//! Startup claims on the unix socket path. The file at that path is a live
//! daemon's front door: evicting it cuts that daemon off from every
//! `bridge-client` while its HTTP health keeps answering ok, which is the one
//! failure an operator's monitoring cannot see. A second daemon must
//! therefore die without touching the path, and only a genuinely stale file
//! may be replaced.

use functional_tests::{wire_call, Daemon, Request, Response, TempDir};

fn embed_stub() -> Request {
    Request::Embed {
        model: "stub".to_string(),
        texts: vec!["hello".to_string()],
    }
}

fn assert_still_serving(daemon: &Daemon) {
    let Response::Embed { dim, .. } = wire_call(daemon.socket(), &embed_stub()) else {
        panic!("the surviving daemon no longer answers on its socket");
    };
    assert_eq!(dim, 384);
}

#[test]
fn a_second_daemon_refuses_to_evict_a_live_socket() {
    let daemon = Daemon::stub();

    let logs = Daemon::builder()
        .without_socket()
        .args(["--socket", daemon.socket().to_str().unwrap()])
        .start_expect_failure();
    assert!(
        logs.contains("already serves this socket"),
        "the refusal must say who owns the path:\n{logs}"
    );

    assert_still_serving(&daemon);
}

#[test]
fn a_taken_http_port_stops_the_daemon_before_it_touches_the_socket() {
    let daemon = Daemon::stub();

    let scratch = TempDir::new("socket-claim");
    let socket = scratch.child("second.sock");
    let logs = Daemon::builder()
        .listen(daemon.port)
        .without_socket()
        .args(["--socket", socket.to_str().unwrap()])
        .start_expect_failure();
    assert!(
        logs.contains(&format!("binding 127.0.0.1:{}", daemon.port)),
        "the error must name the taken address:\n{logs}"
    );
    assert!(
        !socket.exists(),
        "the doomed daemon got far enough to create its socket file"
    );

    assert_still_serving(&daemon);
}

#[test]
fn a_stale_socket_file_is_replaced_on_startup() {
    let scratch = TempDir::new("stale-socket");
    let socket = scratch.child("bridge.sock");
    // A daemon that died without cleanup: bind the path, abandon the file.
    drop(std::os::unix::net::UnixListener::bind(&socket).unwrap());
    assert!(
        socket.exists(),
        "the stale file must be there to be replaced"
    );

    let _daemon = Daemon::builder()
        .without_socket()
        .args(["--socket", socket.to_str().unwrap()])
        .start();

    let Response::Embed { dim, .. } = wire_call(&socket, &embed_stub()) else {
        panic!("the daemon did not serve on the reclaimed socket");
    };
    assert_eq!(dim, 384);
}
