clippy:
	@find ~/.cargo -name "*semantic-a*" -exec rm -f {} \;
	@cargo fmt --all -- --check
	@cargo clippy -- -D warnings -D clippy::pedantic -D clippy::nursery

run:
	@find ~/.cargo -name "*semantic-a*" -exec rm -f {} \;
	@cargo run

build:
	@find ~/.cargo -name "*semantic-a*" -exec rm -f {} \;
	@cargo build --release
