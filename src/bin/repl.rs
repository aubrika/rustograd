//! rustograd REPL — an interactive prompt for experimenting with `Value`s and
//! networks, the way you'd poke at micrograd in a Python shell or notebook.
//!
//! This is a *starting skeleton*: it reads lines and echoes them. Your job
//! (ROADMAP.md → "Interactive REPL") is to grow it into something that can
//! build Values, run ops, call `.backward()`, and print grads.
//!
//! Two paths, pick per your goals:
//!   • Build it out here — add `rustyline` for history/editing, then a small
//!     expression parser/evaluator over `rustograd::engine::Value`.
//!   • Or, for zero-effort interactive Rust, use `evcxr`:
//!         cargo install evcxr_repl   # then run `evcxr`
//!     and `:dep rustograd = { path = "." }` to load this crate live.
//!     (evcxr also ships a Jupyter kernel — handy for the plotting items.)

use std::io::{self, Write};

fn main() {
    println!("rustograd REPL (skeleton). Type `quit` to exit.");
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

        // TODO: parse `input`, build the Value graph, eval, optionally backward,
        // and print the result. For now we just echo.
        println!("(todo) you typed: {input}");
    }

    println!("bye");
}
