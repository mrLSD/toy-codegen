clippy:
	@cargo fmt --all -- --check
	@cargo clippy -- -D warnings -D clippy::pedantic -D clippy::nursery

run:
	@cargo run

build:
	@cargo build --release
