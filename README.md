# implicit-fn.rs

<!-- cargo-rdme start -->

A macro that adds support for implicit closures to Rust.

This provides a concise alternative to regular closure syntax
for when each parameter is used at most once
and is not deeply nested.

This feature [has been suggested before](https://github.com/rust-lang/rfcs/issues/2554),
but this macro mostly exists for fun.

## Examples

Using implicit closures to concisely sum a list:

```rust
#[implicit_fn]
fn main() {
    let n = [1, 2, 3].into_iter().fold(0, _ + _);
    assert_eq!(n, 6);
}
```

Copying all the elements of an array:

```rust
#[implicit_fn]
fn main() {
    let array: [&u32; 3] = [&1, &2, &3];
    let array: [u32; 3] = array.map(*_);
    assert_eq!(array, [1, 2, 3]);
}
```

Running a fallible function in an iterator:

```rust
#[implicit_fn]
fn main() -> Result<(), Box<dyn Error>> {
    let names = fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))?
        .map(_?.file_name().into_string().map_err(|_| "file not UTF-8")?)
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    assert_eq!(names, ["lib.rs"]);
    Ok(())
}
```

Running a match on an array of options:

```rust
#[implicit_fn]
fn main() {
    let options = [Some(16), None, Some(2)];
    let numbers = options.map(match _ {
        Some(x) => x + 1,
        None => 0,
    });
    assert_eq!(numbers, [17, 0, 3]);
}
```

Printing the elements of an iterator:

```rust
#[implicit_fn]
fn main() {
    [1, 2, 3].into_iter().for_each(println!("{}", _));
}
```

<!-- cargo-rdme end -->

## License

MIT.
