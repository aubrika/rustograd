//! rustograd REPL — placeholder.
//!
//! This binary only reads lines and echoes them back. A real evaluator (build
//! values, run ops, call `backward`, print grads) is an unchecked item on the
//! learning roadmap: ROADMAP.md → "Interactive REPL".
//!
//! For interactive use today, `evcxr` works with the library as it is:
//!     cargo install evcxr_repl   # then run `evcxr`
//! and `:dep rustograd = { path = "." }` to load this crate live.

use std::io::{self, Write};

fn main() {
    println!("rustograd REPL (placeholder: echoes input, evaluates nothing). Type `quit` to exit.");
    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        print!("rustograd> ");
        io::stdout().flush().expect("flush failed");

        line.clear();
        if stdin.read_line(&mut line).expect("read failed") == 0 {
            break; // EOF (Ctrl-Z on Windows, Ctrl-D elsewhere)
        }
        let input = line.trim();
        if input == "quit" || input == "exit" {
            break;
        }

        // Not implemented: parse `input`, build the graph, eval, optionally backward.
        println!("(not implemented) you typed: {input}");
    }

    println!("bye");
}
