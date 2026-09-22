#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().map(String::as_str) == Some("screenshot") {
        args.next();
        if let Err(e) = texture_graph_ui::screenshot::run(args) {
            eprintln!("{e}");
            std::process::exit(2);
        }
        return Ok(());
    }
    texture_graph_ui::run_native()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
